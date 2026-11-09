"""The demo agent's spans sent over OTLP to a real tracing backend, then read back from it.

    ./jaeger                         # the binary from a Jaeger 2.x GitHub release, no arguments needed
    pip install -e '.[otlp]'
    python examples/otlp_jaeger.py   # exports, then checks Jaeger stored what was sent

agent-otel is plain OpenTelemetry, so this is the standard OTLP exporter and nothing else. Open
http://localhost:16686 and pick the "demo-support" service to see the two runs.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

import agent_otel as ao

sys.path.insert(0, str(Path(__file__).parent))
import demo_agent  # noqa: E402  (the same fake support agent as the JSONL demo)

OTLP = os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://localhost:4318/v1/traces")
JAEGER = os.environ.get("JAEGER_QUERY_URL", "http://localhost:16686")
SERVICE = "demo-support"


def export() -> str:
    """Run the demo twice and send its spans. Returns this run's service.instance.id."""
    run_id = uuid.uuid4().hex[:12]
    provider = TracerProvider(resource=Resource.create({"service.name": SERVICE, "service.instance.id": run_id}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=OTLP)))
    ao.configure(provider, prices=str(demo_agent.HERE / "prices.example.json"))
    demo_agent.flaky_calls["n"] = 0
    demo_agent.handle("where is my order? it was meant to come yesterday", "conv-1")
    demo_agent.flaky_calls["n"] = 1
    demo_agent.handle("has my parcel shipped yet", "conv-2")
    provider.shutdown()  # flushes the batch
    return run_id


def _value(v: dict):
    """An OTLP JSON attribute value as Python (64-bit ints travel as strings)."""
    if "intValue" in v:
        return int(v["intValue"])
    for k in ("stringValue", "doubleValue", "boolValue"):
        if k in v:
            return v[k]
    return v


def stored_traces(run_id: str, timeout_s: float = 15.0) -> dict[str, list[dict]]:
    """This run's spans as Jaeger stored them, by trace id, read back through its API v3 (OTLP JSON)."""
    now = datetime.now(timezone.utc)
    query = urllib.parse.urlencode({"query.service_name": SERVICE,
                                    "query.start_time_min": (now - timedelta(hours=1)).isoformat(),
                                    "query.start_time_max": (now + timedelta(minutes=5)).isoformat()})
    deadline = time.monotonic() + timeout_s
    while True:
        traces: dict[str, list[dict]] = {}
        try:
            with urllib.request.urlopen(f"{JAEGER}/api/v3/traces?{query}", timeout=5) as r:
                data = json.load(r)
            for rs in data.get("result", {}).get("resourceSpans", []):
                resource = {a["key"]: _value(a["value"]) for a in rs.get("resource", {}).get("attributes", [])}
                if resource.get("service.instance.id") != run_id:
                    continue  # an earlier run of this example
                for ss in rs.get("scopeSpans", []):
                    for s in ss.get("spans", []):
                        attrs = {a["key"]: _value(a["value"]) for a in s.get("attributes", [])}
                        traces.setdefault(s["traceId"], []).append({"name": s["name"], "attrs": attrs})
        except urllib.error.HTTPError as e:
            if e.code != 404:  # 404: nothing indexed for the service yet
                raise
        if len(traces) >= 2 or time.monotonic() > deadline:
            return traces
        time.sleep(0.5)


def check(traces: dict[str, list[dict]]) -> list[str]:
    """What a backend must have kept for agent-otel's spans to be useful there."""
    problems = []
    runs = [s for spans in traces.values() for s in spans if s["name"] == "invoke_agent support-agent"]
    if len(runs) != 2:
        problems.append(f"expected 2 agent runs, found {len(runs)}")
    for tid, spans in traces.items():
        names = {s["name"] for s in spans}
        for want in ("invoke_agent support-agent", "chat local-qwen-7b", "chat example-hosted-small",
                     "execute_tool find_order", "execute_tool track_parcel"):
            if want not in names:
                problems.append(f"trace {tid[:8]}: no {want!r} span")
        for s in spans:
            if s["name"].startswith("chat ") and not isinstance(s["attrs"].get("gen_ai.usage.input_tokens"), int):
                problems.append(f"{s['name']}: gen_ai.usage.input_tokens missing or not an integer")
            if s["name"].startswith("invoke_agent") and "agent_otel.total.cost" not in s["attrs"]:
                problems.append("agent span without agent_otel.total.cost")
    return problems


def main() -> int:
    run_id = export()
    traces = stored_traces(run_id)
    problems = check(traces)
    spans = sum(len(t) for t in traces.values())
    print(f"Jaeger at {JAEGER} stored {len(traces)} traces, {spans} spans from this run ({SERVICE}, instance {run_id})")
    for p in problems:
        print("  problem:", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
