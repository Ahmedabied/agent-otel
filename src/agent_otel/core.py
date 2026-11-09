"""Spans for an agent run, its model calls and its tool calls.

    with agent("support-bot"):
        with model_call("llama3.1:8b", provider="ollama") as m:
            reply = ...
            m.record(input_tokens=812, output_tokens=64, finish_reasons=["tool_calls"])
        with tool_call("search_orders", arguments={"email": "..."}) as t:
            t.result(search_orders(...))

Each block is one OpenTelemetry span named and tagged the GenAI way. The agent span also gets
totals (tokens, cost, calls) for everything inside it, so one span answers "what did this run cost".
"""

from __future__ import annotations

import contextvars
import functools
import inspect
import json
import reprlib
import threading
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from opentelemetry import trace
from opentelemetry.trace import Span, SpanKind, Status, StatusCode

from . import __version__
from . import conventions as C
from .prices import Prices


@dataclass
class _Config:
    tracer: trace.Tracer | None = None
    prices: Prices | None = None
    capture_content: bool = False
    max_content_chars: int = 4000


_config = _Config()


@dataclass
class _Totals:
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    priced: bool = False
    unpriced_calls: int = 0
    model_calls: int = 0
    tool_calls: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


_runs: contextvars.ContextVar[tuple[_Totals, ...]] = contextvars.ContextVar("agent_otel_runs", default=())


def configure(tracer_provider: Any = None, prices: Any = None, capture_content: bool = False,
              max_content_chars: int = 4000) -> None:
    """Choose where spans go, the price table, and whether tool arguments and results are recorded.

    Without a tracer_provider, the global OpenTelemetry provider is used, so agent-otel spans join
    whatever tracing the application already has."""
    _config.tracer = tracer_provider.get_tracer("agent-otel", __version__) if tracer_provider else None
    _config.prices = Prices.load(prices) if prices is not None else None
    _config.capture_content = capture_content
    _config.max_content_chars = max_content_chars


def _tracer() -> trace.Tracer:
    return _config.tracer or trace.get_tracer("agent-otel", __version__)


def _content(value: Any) -> str:
    """Tool arguments or results as text for a span. Never raises: tracing must not break the app."""
    limit = _config.max_content_chars
    try:
        text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    except Exception:  # circular references, a __str__ that raises, ...
        try:
            r = reprlib.Repr()  # attributes, not constructor arguments: those need Python 3.12
            r.maxlevel, r.maxstring, r.maxother = 4, limit, limit
            text = r.repr(value)
        except Exception:
            text = f"<{type(value).__qualname__} that cannot be shown>"
    return text if len(text) <= limit else text[:limit] + f"... [{len(text)} chars]"


def _fail(span: Span, exc: BaseException) -> None:
    span.record_exception(exc)
    span.set_status(Status(StatusCode.ERROR, str(exc)[:200]))
    span.set_attribute(C.ERROR_TYPE, type(exc).__qualname__)


class ModelCall:
    def __init__(self, span: Span, model: str):
        self.span = span
        self.model = model
        self.cost: float | None = None

    def record(self, input_tokens: int | None = None, output_tokens: int | None = None, *,
               response_model: str | None = None, finish_reasons: list[str] | None = None,
               response_id: str | None = None, cache_read_input_tokens: int | None = None,
               cache_write_input_tokens: int | None = None,
               reasoning_output_tokens: int | None = None) -> None:
        """Call once the model has answered, with the usage numbers from its response.

        input_tokens is ALL input tokens, cached ones included, as the OpenTelemetry GenAI conventions
        define it; cache_read_input_tokens and cache_write_input_tokens are the parts of it that were
        read from or written to the provider's prompt cache. Some APIs report input without the cached
        parts (Anthropic's usage.input_tokens, for example): pass their sum as input_tokens. When the
        cache counts are bigger than input_tokens they cannot be part of it, so they are added to it,
        with a warning.
        """
        s = self.span
        read, write = int(cache_read_input_tokens or 0), int(cache_write_input_tokens or 0)
        if input_tokens is not None and read + write > int(input_tokens):
            warnings.warn(f"agent-otel: cache tokens ({read} read + {write} written) exceed input_tokens "
                          f"({int(input_tokens)}), so input_tokens was taken to exclude them and they were added; "
                          "pass input_tokens with the cached tokens included", stacklevel=2)
            input_tokens = int(input_tokens) + read + write
        if input_tokens is not None:
            s.set_attribute(C.USAGE_INPUT_TOKENS, int(input_tokens))
        if output_tokens is not None:
            s.set_attribute(C.USAGE_OUTPUT_TOKENS, int(output_tokens))
        if cache_read_input_tokens is not None:
            s.set_attribute(C.USAGE_CACHE_READ_INPUT_TOKENS, read)
        if cache_write_input_tokens is not None:
            s.set_attribute(C.USAGE_CACHE_WRITE_INPUT_TOKENS, write)
        if reasoning_output_tokens is not None:
            s.set_attribute(C.USAGE_REASONING_OUTPUT_TOKENS, int(reasoning_output_tokens))
        if response_model:
            s.set_attribute(C.RESPONSE_MODEL, response_model)
        if finish_reasons:
            s.set_attribute(C.RESPONSE_FINISH_REASONS, [str(r) for r in finish_reasons])
        if response_id:
            s.set_attribute(C.RESPONSE_ID, response_id)
        prices = _config.prices
        if prices is not None:
            self.cost = prices.cost(response_model or self.model, int(input_tokens or 0), int(output_tokens or 0),
                                    read, write)
            if self.cost is not None:
                s.set_attribute(C.COST, self.cost)
                s.set_attribute(C.COST_CURRENCY, prices.currency)
        for run in _runs.get():
            with run.lock:
                run.input_tokens += int(input_tokens or 0)
                run.output_tokens += int(output_tokens or 0)
                if self.cost is not None:
                    run.cost += self.cost
                    run.priced = True
                elif prices is not None:
                    run.unpriced_calls += 1


class ToolCall:
    def __init__(self, span: Span):
        self.span = span

    def result(self, value: Any) -> Any:
        """Record the tool's result (only kept when capture_content is on). Returns it unchanged."""
        if _config.capture_content:
            self.span.set_attribute(C.TOOL_CALL_RESULT, _content(value))
        return value


def _count(kind: str) -> None:
    for run in _runs.get():
        with run.lock:
            if kind == "model":
                run.model_calls += 1
            else:
                run.tool_calls += 1


@contextmanager
def model_call(model: str, provider: str = "unknown", operation: str = C.CHAT, *,
               max_tokens: int | None = None, temperature: float | None = None) -> Iterator[ModelCall]:
    attrs: dict[str, Any] = {C.OPERATION_NAME: operation, C.PROVIDER_NAME: provider, C.REQUEST_MODEL: model}
    if max_tokens is not None:
        attrs[C.REQUEST_MAX_TOKENS] = max_tokens
    if temperature is not None:
        attrs[C.REQUEST_TEMPERATURE] = temperature
    _count("model")
    with _tracer().start_as_current_span(f"{operation} {model}", kind=SpanKind.CLIENT, attributes=attrs,
                                         record_exception=False, set_status_on_exception=False) as span:
        try:
            yield ModelCall(span, model)
        except BaseException as e:
            _fail(span, e)
            raise


@contextmanager
def tool_call(name: str, *, call_id: str | None = None, arguments: Any = None,
              tool_type: str = "function") -> Iterator[ToolCall]:
    attrs: dict[str, Any] = {C.OPERATION_NAME: C.EXECUTE_TOOL, C.TOOL_NAME: name, C.TOOL_TYPE: tool_type}
    if call_id:
        attrs[C.TOOL_CALL_ID] = call_id
    if arguments is not None and _config.capture_content:
        attrs[C.TOOL_CALL_ARGUMENTS] = _content(arguments)
    _count("tool")
    with _tracer().start_as_current_span(f"{C.EXECUTE_TOOL} {name}", kind=SpanKind.INTERNAL, attributes=attrs,
                                         record_exception=False, set_status_on_exception=False) as span:
        try:
            yield ToolCall(span)
        except BaseException as e:
            _fail(span, e)
            raise


@contextmanager
def agent(name: str, *, conversation_id: str | None = None, agent_id: str | None = None) -> Iterator[Span]:
    """One agent run. Its span carries totals for every model and tool call inside it."""
    attrs: dict[str, Any] = {C.OPERATION_NAME: C.INVOKE_AGENT, C.AGENT_NAME: name}
    if conversation_id:
        attrs[C.CONVERSATION_ID] = conversation_id
    if agent_id:
        attrs[C.AGENT_ID] = agent_id
    totals = _Totals()
    token = _runs.set(_runs.get() + (totals,))
    with _tracer().start_as_current_span(f"{C.INVOKE_AGENT} {name}", kind=SpanKind.INTERNAL, attributes=attrs,
                                         record_exception=False, set_status_on_exception=False) as span:
        try:
            yield span
        except BaseException as e:
            _fail(span, e)
            raise
        finally:
            _runs.reset(token)
            span.set_attribute(C.TOTAL_INPUT_TOKENS, totals.input_tokens)
            span.set_attribute(C.TOTAL_OUTPUT_TOKENS, totals.output_tokens)
            span.set_attribute(C.TOTAL_MODEL_CALLS, totals.model_calls)
            span.set_attribute(C.TOTAL_TOOL_CALLS, totals.tool_calls)
            if _config.prices is not None:
                # The total covers priced calls only; the count says how many it left out.
                span.set_attribute(C.TOTAL_UNPRICED_CALLS, totals.unpriced_calls)
                if totals.priced:
                    span.set_attribute(C.TOTAL_COST, round(totals.cost, 10))
                    span.set_attribute(C.COST_CURRENCY, _config.prices.currency)


def tool(name: str | None = None, *, tool_type: str = "function") -> Callable:
    """Decorator: every call of the function is a tool span. Works on plain and async functions."""

    def wrap(fn: Callable) -> Callable:
        tool_name = name or fn.__name__

        def _args(a: tuple, kw: dict) -> Any:
            if not _config.capture_content:
                return None
            try:
                bound = inspect.signature(fn).bind_partial(*a, **kw)
                return dict(bound.arguments)
            except TypeError:
                return {"args": list(a), "kwargs": kw}

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def run_async(*a, **kw):
                with tool_call(tool_name, arguments=_args(a, kw), tool_type=tool_type) as t:
                    return t.result(await fn(*a, **kw))
            return run_async

        @functools.wraps(fn)
        def run(*a, **kw):
            with tool_call(tool_name, arguments=_args(a, kw), tool_type=tool_type) as t:
                return t.result(fn(*a, **kw))
        return run

    return wrap
