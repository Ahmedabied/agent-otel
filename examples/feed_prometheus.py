"""Keep the demo agents running for a while, exporting metrics every few seconds, so a dashboard has history.

    ./prometheus --web.enable-otlp-receiver
    pip install -e '.[otlp]'
    python examples/feed_prometheus.py --seconds 900    # then open the Grafana dashboard in examples/grafana

One process with one service.instance.id, as a real service would be: rate() and increase() need a series
with many samples, which one short export cannot give. The traffic is pretend but varied, from a fixed seed:
the support agent from demo_agent.py (its carrier call sometimes times out and is retried), and a triage agent
that sometimes fails, sometimes calls a model missing from the price table.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
import uuid
from pathlib import Path

from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource

import agent_otel as ao

sys.path.insert(0, str(Path(__file__).parent))
import demo_agent  # noqa: E402

PROMETHEUS = os.environ.get("PROMETHEUS_URL", "http://localhost:9090")
rng = random.Random(7)


class KnowledgeBaseDown(Exception):
    pass


@ao.tool()
def search_kb(query: str) -> list[str]:
    time.sleep(rng.uniform(0.01, 0.06))
    if rng.random() < 0.08:
        raise KnowledgeBaseDown("knowledge base did not answer")
    return ["returns-policy", "delivery-times"]


def triage(question: str) -> str:
    with ao.agent("triage-agent"):
        model = "example-unlisted" if rng.random() < 0.05 else "example-hosted-small"
        with ao.model_call(model, provider="example") as m:
            time.sleep(rng.uniform(0.03, 0.12))
            m.record(input_tokens=420 + rng.randint(0, 200), output_tokens=rng.randint(8, 40))
        return search_kb(question)[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=300)
    ap.add_argument("--export-every", type=float, default=5)
    a = ap.parse_args()
    instance = f"feeder-{uuid.uuid4().hex[:8]}"
    resource = Resource.create({"service.name": "demo-support", "service.instance.id": instance})
    exporter = OTLPMetricExporter(endpoint=f"{PROMETHEUS}/api/v1/otlp/v1/metrics")
    meters = MeterProvider(resource=resource, metric_readers=[
        PeriodicExportingMetricReader(exporter, export_interval_millis=int(a.export_every * 1000))])
    ao.configure(prices=str(demo_agent.HERE / "prices.example.json"), meter_provider=meters)  # spans: none
    deadline = time.monotonic() + a.seconds
    runs = failed = 0
    while time.monotonic() < deadline:
        try:
            if rng.random() < 0.6:
                demo_agent.flaky_calls["n"] = 0 if rng.random() < 0.3 else 1
                demo_agent.handle("where is my order?", "conv")
            else:
                triage("can I return a jacket I bought last week")
        except Exception:  # noqa: BLE001 - a failed run is part of the traffic
            failed += 1
        runs += 1
        time.sleep(rng.uniform(0.1, 0.6))
    meters.shutdown()
    print(f"fed {runs} agent runs ({failed} failed) to {PROMETHEUS} as instance {instance} in {a.seconds:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
