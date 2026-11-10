import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import agent_otel
from agent_otel import agent, model_call, tool, tool_call
from agent_otel import conventions as C

PRICES = {"currency": "USD", "models": {"gpt-x*": {"input": 1.0, "output": 2.0, "cache_read": 0.1}}}

# Every attribute a metric may carry. Ids (conversation, response, tool call, agent) and content must
# never be here: each value would become its own time series.
ALLOWED = {C.OPERATION_NAME, C.PROVIDER_NAME, C.REQUEST_MODEL, C.RESPONSE_MODEL, C.ERROR_TYPE, C.TOKEN_MODALITY,
           C.TOOL_NAME, C.TOOL_TYPE, C.AGENT_NAME, C.COST_CURRENCY}


class Telemetry:
    def __init__(self, reader, exporter):
        self.reader, self.exporter = reader, exporter
        self._data = None

    def collect(self):
        self._data = {}
        data = self.reader.get_metrics_data()
        for rm in data.resource_metrics if data else ():
            for sm in rm.scope_metrics:
                for m in sm.metrics:
                    self._data[m.name] = m
        return self._data

    def points(self, name):
        data = self._data if self._data is not None else self.collect()
        return list(data[name].data.data_points) if name in data else []

    def point(self, name, **attrs):
        found = [p for p in self.points(name) if all(p.attributes.get(k) == v for k, v in attrs.items())]
        assert len(found) == 1, (name, attrs, [dict(p.attributes) for p in self.points(name)])
        return found[0]

    def span(self, name):
        found = [s for s in self.exporter.get_finished_spans() if s.name == name]
        assert len(found) == 1
        return found[0]


@pytest.fixture
def tel():
    reader = InMemoryMetricReader()
    meters = MeterProvider(metric_readers=[reader])
    exporter = InMemorySpanExporter()
    tracer = TracerProvider()
    tracer.add_span_processor(SimpleSpanProcessor(exporter))

    def make(prices=None, capture_content=False):
        agent_otel.configure(tracer, prices=prices, capture_content=capture_content, meter_provider=meters)
        return Telemetry(reader, exporter)

    yield make
    agent_otel.configure(None)
    meters.shutdown()
    tracer.shutdown()


def test_a_model_call_gives_the_inference_metrics_with_the_spans_numbers(tel):
    t = tel(PRICES)
    with agent("support", conversation_id="conv-1", agent_id="a-1"):
        with model_call("gpt-x", provider="openai") as m:
            m.record(input_tokens=1000, output_tokens=50, cache_read_input_tokens=400, reasoning_output_tokens=20,
                     response_model="gpt-x-2026", response_id="resp-1")
    base = {C.OPERATION_NAME: "chat", C.PROVIDER_NAME: "openai", C.REQUEST_MODEL: "gpt-x",
            C.RESPONSE_MODEL: "gpt-x-2026"}
    span = t.span("chat gpt-x")
    d = t.point(C.M_INFERENCE_DURATION)
    assert dict(d.attributes) == base
    assert d.count == 1 and d.sum == (span.end_time - span.start_time) / 1e9  # the same clock, exactly
    assert tuple(d.explicit_bounds) == C.SECONDS_BUCKETS
    assert t.point(C.M_INFERENCE_INPUT_TOKENS).sum == 1000
    assert tuple(t.point(C.M_INFERENCE_INPUT_TOKENS).explicit_bounds) == C.TOKEN_BUCKETS
    assert t.point(C.M_INFERENCE_OUTPUT_TOKENS).sum == 50
    unknown = dict(base, **{C.TOKEN_MODALITY: "unknown"})
    assert dict(t.point(C.M_USAGE_INPUT_TOKENS).attributes) == unknown
    assert t.point(C.M_USAGE_INPUT_TOKENS).value == 1000  # cached tokens included, as the span says
    assert t.point(C.M_USAGE_OUTPUT_TOKENS).value == 50
    assert t.point(C.M_USAGE_CACHE_READ_INPUT_TOKENS).value == 400
    assert t.points(C.M_USAGE_CACHE_WRITE_INPUT_TOKENS) == []  # nothing written, nothing added
    reasoning = t.point(C.M_USAGE_REASONING_OUTPUT_TOKENS)
    assert reasoning.value == 20 and dict(reasoning.attributes) == base  # no modality on this one
    cost = t.point(C.COST)
    assert cost.value == pytest.approx(span.attributes[C.COST])
    assert cost.attributes[C.COST_CURRENCY] == "USD"


def test_metrics_never_carry_ids_or_content(tel):
    t = tel(PRICES, capture_content=True)
    with agent("support", conversation_id="conv-secret", agent_id="agent-secret"):
        with model_call("gpt-x", provider="openai") as m:
            m.record(input_tokens=10, output_tokens=5, response_id="resp-secret", finish_reasons=["stop"])
        with tool_call("lookup", call_id="call-secret", arguments={"email": "a@example.com"}) as c:
            c.result({"order": "secret-order"})
    seen = set()
    for metric in t.collect().values():
        for p in metric.data.data_points:
            seen |= set(p.attributes)
            assert not any("secret" in str(v) or "example.com" in str(v) for v in p.attributes.values())
    assert seen <= ALLOWED, seen - ALLOWED


def test_a_failed_call_records_zero_tokens_with_the_error_and_adds_nothing(tel):
    t = tel(PRICES)
    with pytest.raises(TimeoutError):
        with model_call("gpt-x", provider="openai"):
            raise TimeoutError("slow")
    d = t.point(C.M_INFERENCE_DURATION)
    assert d.attributes[C.ERROR_TYPE] == "TimeoutError"
    assert t.point(C.M_INFERENCE_INPUT_TOKENS, **{C.ERROR_TYPE: "TimeoutError"}).sum == 0
    assert t.point(C.M_INFERENCE_OUTPUT_TOKENS).sum == 0
    for counter in (C.M_USAGE_INPUT_TOKENS, C.M_USAGE_OUTPUT_TOKENS, C.COST, C.M_UNPRICED_CALLS):
        assert t.points(counter) == []


def test_usage_from_a_call_that_then_failed_still_counts(tel):
    t = tel()
    with pytest.raises(ConnectionError):
        with model_call("gpt-x", provider="openai") as m:
            m.record(input_tokens=300, output_tokens=7)
            raise ConnectionError("stream cut")
    assert t.point(C.M_USAGE_INPUT_TOKENS).value == 300
    assert t.point(C.M_INFERENCE_OUTPUT_TOKENS).attributes[C.ERROR_TYPE] == "ConnectionError"


def test_record_called_twice_counts_the_final_numbers_once(tel):
    t = tel(PRICES)
    with agent("streamer"):
        with model_call("gpt-x", provider="openai") as m:
            m.record(input_tokens=100)
            m.record(output_tokens=40)  # usage reported in two parts, as some streams do
    span = t.span("chat gpt-x")
    assert span.attributes[C.USAGE_INPUT_TOKENS] == 100 and span.attributes[C.USAGE_OUTPUT_TOKENS] == 40
    run_span = t.span("invoke_agent streamer")
    assert run_span.attributes[C.TOTAL_INPUT_TOKENS] == 100 and run_span.attributes[C.TOTAL_OUTPUT_TOKENS] == 40
    assert span.attributes[C.COST] == pytest.approx((100 * 1.0 + 40 * 2.0) / 1e6)
    assert run_span.attributes[C.TOTAL_COST] == pytest.approx(span.attributes[C.COST])
    assert t.point(C.M_INFERENCE_INPUT_TOKENS).count == 1  # one value per call, not one per record()
    assert t.point(C.M_USAGE_INPUT_TOKENS).value == 100
    assert t.point(C.M_USAGE_OUTPUT_TOKENS).value == 40
    assert t.point(C.COST).value == pytest.approx(span.attributes[C.COST])


def test_cache_tokens_reported_apart_are_counted_like_the_span(tel):
    t = tel()
    with pytest.warns(UserWarning):
        with model_call("claude-x", provider="anthropic") as m:
            m.record(input_tokens=20, output_tokens=300, cache_read_input_tokens=50_000)
    assert t.span("chat claude-x").attributes[C.USAGE_INPUT_TOKENS] == 50_020
    assert t.point(C.M_USAGE_INPUT_TOKENS).value == 50_020
    assert t.point(C.M_INFERENCE_INPUT_TOKENS).sum == 50_020


def test_embeddings_get_their_own_metrics(tel):
    t = tel()
    with model_call("embed-x", provider="openai", operation="embeddings") as m:
        m.record(input_tokens=64)
    assert t.point(C.M_EMBEDDINGS_INPUT_TOKENS).sum == 64
    span = t.span("embeddings embed-x")
    assert t.point(C.M_EMBEDDINGS_DURATION).sum == (span.end_time - span.start_time) / 1e9
    for name in (C.M_INFERENCE_DURATION, C.M_INFERENCE_INPUT_TOKENS, C.M_USAGE_INPUT_TOKENS):
        assert t.points(name) == []


def test_unpriced_calls_are_counted_beside_the_cost(tel):
    t = tel(PRICES)
    with model_call("gpt-x", provider="openai") as m:
        m.record(input_tokens=10, output_tokens=1)
    with model_call("mystery-model", provider="other") as m:
        m.record(input_tokens=10, output_tokens=1)
    assert t.point(C.M_UNPRICED_CALLS).value == 1
    assert t.point(C.M_UNPRICED_CALLS).attributes[C.REQUEST_MODEL] == "mystery-model"
    assert t.point(C.COST).attributes[C.REQUEST_MODEL] == "gpt-x"


def test_no_price_table_means_no_cost_metrics(tel):
    t = tel()
    with model_call("gpt-x", provider="openai") as m:
        m.record(input_tokens=10, output_tokens=1)
    assert t.points(C.COST) == [] and t.points(C.M_UNPRICED_CALLS) == []


def test_tool_duration_names_the_agent_and_the_error(tel):
    t = tel()

    @tool()
    def lookup(order_id):
        return {"id": order_id}

    with agent("support"):
        lookup("A1")
        with pytest.raises(KeyError):
            with tool_call("refund"):
                raise KeyError("no such order")
    lookup("A2")  # outside any agent
    span = [s for s in t.exporter.get_finished_spans() if s.name == "execute_tool refund"][0]
    refund = t.point(C.M_EXECUTE_TOOL_DURATION, **{C.TOOL_NAME: "refund"})
    assert dict(refund.attributes) == {C.TOOL_NAME: "refund", C.TOOL_TYPE: "function", C.AGENT_NAME: "support",
                                       C.ERROR_TYPE: "KeyError"}
    assert refund.sum == (span.end_time - span.start_time) / 1e9
    assert t.point(C.M_EXECUTE_TOOL_DURATION, **{C.TOOL_NAME: "lookup", C.AGENT_NAME: "support"}).count == 1
    outside = [p for p in t.points(C.M_EXECUTE_TOOL_DURATION)
               if p.attributes[C.TOOL_NAME] == "lookup" and C.AGENT_NAME not in p.attributes]
    assert len(outside) == 1 and outside[0].count == 1


def test_each_call_is_counted_once_across_nested_agents(tel):
    t = tel()
    with agent("router"):  # only hands work to the two below: an entrypoint
        with agent("planner"):
            for _ in range(3):
                with model_call("gpt-x", provider="openai") as m:
                    m.record(input_tokens=1, output_tokens=1)
            with tool_call("search"):
                pass
        with agent("writer"):
            with model_call("gpt-x", provider="openai") as m:
                m.record(input_tokens=1, output_tokens=1)
    with agent("idle"):
        pass
    calls = {p.attributes[C.AGENT_NAME]: p for p in t.points(C.M_INVOKE_AGENT_INFERENCE_CALLS)}
    tools = {p.attributes[C.AGENT_NAME]: p for p in t.points(C.M_INVOKE_AGENT_TOOL_CALLS)}
    assert {k: p.sum for k, p in calls.items()} == {"planner": 3, "writer": 1, "idle": 0}
    assert {k: p.sum for k, p in tools.items()} == {"planner": 1, "writer": 0, "idle": 0}
    assert tuple(calls["planner"].explicit_bounds) == C.CALL_COUNT_BUCKETS
    # The span totals still cover everything inside, nested runs included.
    assert t.span("invoke_agent router").attributes[C.TOTAL_MODEL_CALLS] == 4
    router = t.point(C.M_INVOKE_AGENT_DURATION, **{C.AGENT_NAME: "router"})
    span = t.span("invoke_agent router")
    assert router.sum == (span.end_time - span.start_time) / 1e9
    assert tuple(router.explicit_bounds) == C.AGENT_SECONDS_BUCKETS


def test_a_failed_agent_run_carries_the_error(tel):
    t = tel()
    with pytest.raises(RuntimeError):
        with agent("support"):
            raise RuntimeError("gave up")
    assert t.point(C.M_INVOKE_AGENT_DURATION).attributes[C.ERROR_TYPE] == "RuntimeError"
    assert C.ERROR_TYPE not in t.point(C.M_INVOKE_AGENT_INFERENCE_CALLS).attributes


def test_metrics_are_off_without_a_meter_provider(spans):
    spans()
    assert agent_otel.core._config.instruments is None
    with model_call("gpt-x") as m:
        m.record(input_tokens=1, output_tokens=1)


def test_metrics_never_raise_into_the_app(tel):
    t = tel()

    class Broken:
        def __getattr__(self, name):
            raise RuntimeError("meter is broken")

    agent_otel.core._config.instruments.inference_duration = Broken()
    agent_otel.core._config.instruments.tool_duration = Broken()
    with agent("support"):
        with model_call("gpt-x", provider="openai") as m:
            m.record(input_tokens=5, output_tokens=1)
        with tool_call("lookup") as c:
            assert c.result(41) + 1 == 42
    assert t.span("chat gpt-x").end_time is not None  # spans still end and export
    assert t.span("execute_tool lookup").end_time is not None
