"""agent-otel: traces, token cost and latency for AI agent tool and model calls."""

__version__ = "0.1.0"

from .core import ModelCall, ToolCall, agent, configure, model_call, tool, tool_call  # noqa: E402
from .export import JsonlExporter, read_spans  # noqa: E402
from .prices import Prices  # noqa: E402


def setup(jsonl: str | None = None, *, service: str = "agent", prices=None, capture_content: bool = False):
    """One call for local use: spans go to a JSON lines file you can open with `agent-otel view`.

    Returns the TracerProvider; call .shutdown() on it (or let the program exit) to flush."""
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": service}))
    if jsonl:
        provider.add_span_processor(BatchSpanProcessor(JsonlExporter(jsonl)))
    configure(provider, prices=prices, capture_content=capture_content)
    return provider


__all__ = ["agent", "model_call", "tool_call", "tool", "configure", "setup", "ModelCall", "ToolCall",
           "JsonlExporter", "read_spans", "Prices", "__version__"]
