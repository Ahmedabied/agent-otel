"""OpenTelemetry metrics for the same calls the spans describe, the GenAI way.

Off until configure(meter_provider=...) is given a MeterProvider. Each value is recorded once, when the
call's block ends, from the same numbers and the same clock readings as its span, so a dashboard and a
trace never disagree. Only low-cardinality attributes go on metrics: no conversation, response, call or
agent ids, and never content.
"""

from __future__ import annotations

from typing import Any

from . import __version__
from . import conventions as C


class Instruments:
    def __init__(self, meter_provider: Any, currency: str | None):
        m = meter_provider.get_meter("agent-otel", __version__)

        def hist(name: str, unit: str, desc: str, buckets: tuple) -> Any:
            return m.create_histogram(name, unit=unit, description=desc, explicit_bucket_boundaries_advisory=buckets)

        self.inference_duration = hist(C.M_INFERENCE_DURATION, "s", "GenAI client inference operation duration.",
                                       C.SECONDS_BUCKETS)
        self.inference_input = hist(C.M_INFERENCE_INPUT_TOKENS, "{token}",
                                    "The number of input (prompt) tokens used per inference operation.",
                                    C.TOKEN_BUCKETS)
        self.inference_output = hist(C.M_INFERENCE_OUTPUT_TOKENS, "{token}",
                                     "The number of output (completion) tokens used per inference operation.",
                                     C.TOKEN_BUCKETS)
        self.usage_input = m.create_counter(C.M_USAGE_INPUT_TOKENS, unit="{token}",
                                            description="The number of input (prompt) tokens used, including cached tokens.")
        self.usage_output = m.create_counter(C.M_USAGE_OUTPUT_TOKENS, unit="{token}",
                                             description="The number of output (completion) tokens used, including reasoning tokens.")
        self.usage_cache_read = m.create_counter(C.M_USAGE_CACHE_READ_INPUT_TOKENS, unit="{token}",
                                                 description="The number of input tokens served from a provider-managed cache.")
        self.usage_cache_write = m.create_counter(C.M_USAGE_CACHE_WRITE_INPUT_TOKENS, unit="{token}",
                                                  description="The number of input tokens written to a provider-managed cache.")
        self.usage_reasoning = m.create_counter(C.M_USAGE_REASONING_OUTPUT_TOKENS, unit="{token}",
                                                description="The number of output tokens used for reasoning.")
        self.embeddings_duration = hist(C.M_EMBEDDINGS_DURATION, "s", "GenAI client embeddings operation duration.",
                                        C.SECONDS_BUCKETS)
        self.embeddings_input = hist(C.M_EMBEDDINGS_INPUT_TOKENS, "{token}",
                                     "The number of input tokens used per embeddings operation.", C.TOKEN_BUCKETS)
        self.tool_duration = hist(C.M_EXECUTE_TOOL_DURATION, "s", "The duration of a single tool execution.",
                                  C.SECONDS_BUCKETS)
        self.agent_duration = hist(C.M_INVOKE_AGENT_DURATION, "s", "The end-to-end duration of one agent run.",
                                   C.AGENT_SECONDS_BUCKETS)
        self.agent_inference_calls = hist(C.M_INVOKE_AGENT_INFERENCE_CALLS, "{inference_call}",
                                          "The number of model calls one agent run made itself.", C.CALL_COUNT_BUCKETS)
        self.agent_tool_calls = hist(C.M_INVOKE_AGENT_TOOL_CALLS, "{tool_call}",
                                     "The number of tool calls one agent run made itself.", C.CALL_COUNT_BUCKETS)
        # Not in the conventions. Cost needs a price table; the unpriced counter says what the cost leaves out.
        self.currency = currency
        self.cost = m.create_counter(C.COST, unit="1", description="Model call cost from your price table.")
        self.unpriced = m.create_counter(C.M_UNPRICED_CALLS, unit="{call}",
                                         description="Model calls with usage but no price, left out of agent_otel.cost.")

    def model(self, operation: str, provider: str, model: str, response_model: str | None, error: str | None,
              seconds: float, usage: Any, cost: float | None, priced_table: bool) -> None:
        """usage is the ModelCall (its token fields), or None when record() was never called."""
        attrs: dict[str, Any] = {C.OPERATION_NAME: operation, C.PROVIDER_NAME: provider, C.REQUEST_MODEL: model}
        if response_model:
            attrs[C.RESPONSE_MODEL] = response_model
        with_error = dict(attrs, **{C.ERROR_TYPE: error}) if error else attrs
        tokens_in = usage.input_tokens if usage is not None else 0
        if operation == C.EMBEDDINGS:
            self.embeddings_duration.record(seconds, with_error)
            self.embeddings_input.record(tokens_in, with_error)
            return self._cost(attrs, usage, cost, priced_table)
        tokens_out = usage.output_tokens if usage is not None else 0
        self.inference_duration.record(seconds, with_error)
        self.inference_input.record(tokens_in, with_error)
        self.inference_output.record(tokens_out, with_error)
        if usage is not None:
            # Counters take positive amounts only; zeros would not change a total.
            by_modality = dict(attrs, **{C.TOKEN_MODALITY: C.MODALITY_UNKNOWN})
            for amount, counter in ((tokens_in, self.usage_input), (tokens_out, self.usage_output),
                                    (usage.cache_read, self.usage_cache_read),
                                    (usage.cache_write, self.usage_cache_write)):
                if amount > 0:
                    counter.add(amount, by_modality)
            if usage.reasoning > 0:
                self.usage_reasoning.add(usage.reasoning, attrs)  # this one has no modality in the conventions
        self._cost(attrs, usage, cost, priced_table)

    def _cost(self, attrs: dict[str, Any], usage: Any, cost: float | None, priced_table: bool) -> None:
        if usage is not None and priced_table:
            if cost is None:
                self.unpriced.add(1, attrs)
            elif cost > 0:
                self.cost.add(cost, dict(attrs, **{C.COST_CURRENCY: self.currency}))

    def tool(self, name: str, tool_type: str, agent_name: str | None, error: str | None, seconds: float) -> None:
        attrs: dict[str, Any] = {C.TOOL_NAME: name, C.TOOL_TYPE: tool_type}
        if agent_name:
            attrs[C.AGENT_NAME] = agent_name
        if error:
            attrs[C.ERROR_TYPE] = error
        self.tool_duration.record(seconds, attrs)

    def agent(self, name: str, error: str | None, seconds: float, model_calls: int, tool_calls: int,
              delegated: bool) -> None:
        attrs: dict[str, Any] = {C.OPERATION_NAME: C.INVOKE_AGENT, C.AGENT_NAME: name}
        self.agent_duration.record(seconds, dict(attrs, **{C.ERROR_TYPE: error}) if error else attrs)
        # Calls a run made itself; a nested run counts its own. A run that only handed work to nested runs
        # is an entrypoint, which the conventions say not to report; any other run reports 0 too.
        if model_calls or not delegated:
            self.agent_inference_calls.record(model_calls, attrs)
        if tool_calls or not delegated:
            self.agent_tool_calls.record(tool_calls, attrs)
