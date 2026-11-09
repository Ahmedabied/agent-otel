import asyncio

import pytest

import agent_otel as ao
from agent_otel import conventions as C

PRICES = {"currency": "USD", "models": {"local-*": {"input": 0, "output": 0},
                                        "hosted-small": {"input": 1.0, "output": 4.0, "cache_read": 0.1}}}


def by_name(exporter):
    return {s.name: s for s in exporter.get_finished_spans()}


def test_names_kinds_and_genai_attributes(spans):
    exp = spans()
    with ao.agent("helper", conversation_id="c-1"):
        with ao.model_call("local-8b", provider="ollama", max_tokens=256, temperature=0.2) as m:
            m.record(input_tokens=100, output_tokens=20, finish_reasons=["stop"], response_model="local-8b-q4")
        with ao.tool_call("search", call_id="t-1"):
            pass
    s = by_name(exp)
    chat, tool, run = s["chat local-8b"], s["execute_tool search"], s["invoke_agent helper"]
    assert chat.kind.name == "CLIENT" and tool.kind.name == "INTERNAL"
    assert chat.attributes[C.OPERATION_NAME] == "chat"
    assert chat.attributes[C.PROVIDER_NAME] == "ollama"
    assert chat.attributes[C.REQUEST_MODEL] == "local-8b"
    assert chat.attributes[C.RESPONSE_MODEL] == "local-8b-q4"
    assert chat.attributes[C.USAGE_INPUT_TOKENS] == 100 and chat.attributes[C.USAGE_OUTPUT_TOKENS] == 20
    assert chat.attributes[C.REQUEST_MAX_TOKENS] == 256 and chat.attributes[C.REQUEST_TEMPERATURE] == 0.2
    assert tuple(chat.attributes[C.RESPONSE_FINISH_REASONS]) == ("stop",)
    assert tool.attributes[C.TOOL_NAME] == "search" and tool.attributes[C.TOOL_CALL_ID] == "t-1"
    assert run.attributes[C.CONVERSATION_ID] == "c-1"
    assert chat.parent.span_id == run.context.span_id == tool.parent.span_id
    assert "gen_ai.system" not in chat.attributes  # the deprecated name is not used


def test_totals_and_cost_roll_up(spans):
    exp = spans(prices=PRICES)
    with ao.agent("outer"):
        with ao.model_call("hosted-small") as m:
            m.record(input_tokens=1_000_000, output_tokens=250_000, cache_read_input_tokens=400_000)
        with ao.agent("inner"):
            with ao.model_call("local-8b") as m:
                m.record(input_tokens=500, output_tokens=50)
            with ao.tool_call("x"):
                pass
        with ao.model_call("unknown-model") as m:
            m.record(input_tokens=10, output_tokens=1)
    s = by_name(exp)
    # 600k fresh input at 1.0 + 400k cached at 0.1 + 250k output at 4.0, per million
    assert s["chat hosted-small"].attributes[C.COST] == pytest.approx(0.6 + 0.04 + 1.0)
    assert C.COST not in s["chat unknown-model"].attributes  # no price, no made-up number
    outer, inner = s["invoke_agent outer"].attributes, s["invoke_agent inner"].attributes
    assert outer[C.TOTAL_INPUT_TOKENS] == 1_000_510 and outer[C.TOTAL_OUTPUT_TOKENS] == 250_051
    assert outer[C.TOTAL_MODEL_CALLS] == 3 and outer[C.TOTAL_TOOL_CALLS] == 1
    assert outer[C.TOTAL_COST] == pytest.approx(1.64) and outer[C.COST_CURRENCY] == "USD"
    assert inner[C.TOTAL_INPUT_TOKENS] == 500 and inner[C.TOTAL_COST] == 0
    assert inner[C.TOTAL_MODEL_CALLS] == 1
    assert outer[C.TOTAL_UNPRICED_CALLS] == 1 and inner[C.TOTAL_UNPRICED_CALLS] == 0  # the total says what it left out


def test_no_price_table_means_no_cost(spans):
    exp = spans()
    with ao.agent("a"):
        with ao.model_call("hosted-small") as m:
            m.record(input_tokens=10, output_tokens=10)
    s = by_name(exp)
    assert C.COST not in s["chat hosted-small"].attributes
    assert C.TOTAL_COST not in s["invoke_agent a"].attributes
    assert C.TOTAL_UNPRICED_CALLS not in s["invoke_agent a"].attributes


def test_errors_mark_the_span_and_propagate(spans):
    exp = spans()
    with pytest.raises(TimeoutError):
        with ao.agent("a"):
            with ao.tool_call("slow"):
                raise TimeoutError("took too long")
    s = by_name(exp)
    for name in ["execute_tool slow", "invoke_agent a"]:
        assert s[name].status.status_code.name == "ERROR"
        assert s[name].attributes[C.ERROR_TYPE] == "TimeoutError"
    assert s["execute_tool slow"].events[0].name == "exception"


def test_content_is_off_by_default(spans):
    exp = spans()

    @ao.tool()
    def lookup(order_id, secret="x"):
        return {"status": "shipped"}

    assert lookup("A1") == {"status": "shipped"}
    attrs = by_name(exp)["execute_tool lookup"].attributes
    assert C.TOOL_CALL_ARGUMENTS not in attrs and C.TOOL_CALL_RESULT not in attrs


def test_content_when_asked_and_capped(spans):
    exp = spans(capture_content=True)

    @ao.tool("lookup_order")
    def lookup(order_id):
        return "x" * 10_000

    lookup("A1")
    attrs = by_name(exp)["execute_tool lookup_order"].attributes
    assert attrs[C.TOOL_CALL_ARGUMENTS] == '{"order_id": "A1"}'
    assert attrs[C.TOOL_CALL_RESULT].endswith("[10000 chars]") and len(attrs[C.TOOL_CALL_RESULT]) < 4100


def test_async_tools(spans):
    exp = spans()

    @ao.tool()
    async def fetch_page(url):
        await asyncio.sleep(0)
        return 200

    async def main():
        with ao.agent("a"):
            return await asyncio.gather(fetch_page("u1"), fetch_page("u2"))

    assert asyncio.run(main()) == [200, 200]
    s = [x for x in exp.get_finished_spans() if x.name == "execute_tool fetch_page"]
    run = by_name(exp)["invoke_agent a"]
    assert len(s) == 2 and all(x.parent.span_id == run.context.span_id for x in s)
    assert run.attributes[C.TOTAL_TOOL_CALLS] == 2


def test_threads_with_copied_context(spans):
    import contextvars
    from concurrent.futures import ThreadPoolExecutor

    exp = spans()

    @ao.tool()
    def work(i):
        return i

    with ao.agent("pool") as run_span:
        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(contextvars.copy_context().run, work, i) for i in range(3)]
            assert [f.result() for f in futures] == [0, 1, 2]
    s = by_name(exp)["invoke_agent pool"]
    tools = [x for x in exp.get_finished_spans() if x.name == "execute_tool work"]
    assert s.attributes[C.TOTAL_TOOL_CALLS] == 3
    assert all(t.parent.span_id == run_span.get_span_context().span_id for t in tools)


def test_joins_an_existing_global_setup(tmp_path):
    import subprocess
    import sys

    code = """
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import agent_otel as ao
exp = InMemorySpanExporter()
tp = TracerProvider(); tp.add_span_processor(SimpleSpanProcessor(exp)); trace.set_tracer_provider(tp)
with trace.get_tracer("app").start_as_current_span("POST /chat") as req:
    with ao.agent("bot"):
        with ao.tool_call("t"):
            pass
names = {s.name: s for s in exp.get_finished_spans()}
assert names["invoke_agent bot"].parent.span_id == req.get_span_context().span_id
assert names["execute_tool t"].parent.span_id == names["invoke_agent bot"].context.span_id
print("ok")
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.stdout.strip() == "ok", r.stderr


def test_cache_tokens_reported_apart_from_input_are_added_with_a_warning(spans):
    # Anthropic-style usage: input_tokens leaves out the cache reads. 20 + 50,000 read + 300 out.
    exp = spans(prices={"models": {"m": {"input": 3, "output": 15, "cache_read": 0.3}}})
    with ao.agent("a"):
        with ao.model_call("m") as m:
            with pytest.warns(UserWarning, match="exceed input_tokens"):
                m.record(input_tokens=20, output_tokens=300, cache_read_input_tokens=50_000)
    chat = by_name(exp)["chat m"].attributes
    assert chat[C.USAGE_INPUT_TOKENS] == 50_020 and chat[C.USAGE_CACHE_READ_INPUT_TOKENS] == 50_000
    assert m.cost == pytest.approx((20 * 3 + 50_000 * 0.3 + 300 * 15) / 1e6)


def test_cache_writes_are_counted_and_priced(spans, recwarn):
    exp = spans(prices={"models": {"m": {"input": 1, "output": 0, "cache_read": 0.1, "cache_write": 1.25}}})
    with ao.model_call("m") as m:
        m.record(input_tokens=1_000_000, output_tokens=0, cache_read_input_tokens=300_000,
                 cache_write_input_tokens=200_000)
    # 500k fresh at 1.0 + 300k read at 0.1 + 200k written at 1.25, per million
    assert m.cost == pytest.approx(0.5 + 0.03 + 0.25)
    assert by_name(exp)["chat m"].attributes[C.USAGE_CACHE_WRITE_INPUT_TOKENS] == 200_000
    assert not recwarn.list  # counts inside input_tokens are the normal case


class Unprintable:
    def __str__(self):
        raise RuntimeError("no str")

    def __repr__(self):
        raise RuntimeError("no repr")


def test_tracing_never_raises_into_the_app(spans):
    exp = spans(capture_content=True)

    @ao.tool()
    def make_graph():
        a = []
        a.append(a)  # json.dumps: "Circular reference detected"
        return a

    @ao.tool()
    def odd(x):
        return {"thing": Unprintable()}

    graph = make_graph()
    assert graph[0] is graph
    assert odd(Unprintable())["thing"]
    s = by_name(exp)
    assert s["execute_tool make_graph"].attributes[C.TOOL_CALL_RESULT].startswith("[[")  # a capped repr
    assert "Unprintable" in s["execute_tool odd"].attributes[C.TOOL_CALL_RESULT]
    assert "Unprintable" in s["execute_tool odd"].attributes[C.TOOL_CALL_ARGUMENTS]
    assert s["execute_tool odd"].status.status_code.name != "ERROR"
