import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import agent_otel


@pytest.fixture
def spans():
    """Spans finish into memory; returns a function that gives them by name."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    agent_otel.configure(provider)

    def get(prices=None, capture_content=False):
        agent_otel.configure(provider, prices=prices, capture_content=capture_content)
        return exporter

    yield get
    agent_otel.configure(None)
    provider.shutdown()
