"""The demo agent's metrics sent over OTLP to a real Prometheus, then read back with PromQL.

    ./prometheus --web.enable-otlp-receiver   # Prometheus 3.x from its GitHub release
    pip install -e '.[otlp]'
    python examples/otlp_prometheus.py        # exports, then checks Prometheus holds what the spans say

agent-otel is plain OpenTelemetry, so this is the standard OTLP metric exporter. The check compares
two separate paths: the numbers on the spans (kept in memory here) and the numbers Prometheus stored,
so a dashboard built on these metrics shows the same tokens, calls and cost as the traces.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import agent_otel as ao
from agent_otel import conventions as C

sys.path.insert(0, str(Path(__file__).parent))
import demo_agent  # noqa: E402  (the same fake support agent as the other demos)

PROMETHEUS = os.environ.get("PROMETHEUS_URL", "http://localhost:9090")
SERVICE = "demo-support"
ALERTS = Path(__file__).parent / "prometheus" / "alerts.yml"


def export() -> tuple[str, list]:
    """Run the demo twice and one unpriced call; metrics go to Prometheus, spans stay in memory."""
    run_id = uuid.uuid4().hex[:12]
    resource = Resource.create({"service.name": SERVICE, "service.instance.id": run_id})
    spans = InMemorySpanExporter()
    tracer = TracerProvider(resource=resource)
    tracer.add_span_processor(SimpleSpanProcessor(spans))
    exporter = OTLPMetricExporter(endpoint=f"{PROMETHEUS}/api/v1/otlp/v1/metrics")
    # One export, at shutdown: the long interval keeps a slow machine from sending a partial first batch.
    meters = MeterProvider(resource=resource,
                           metric_readers=[PeriodicExportingMetricReader(exporter, export_interval_millis=3_600_000)])
    ao.configure(tracer, prices=str(demo_agent.HERE / "prices.example.json"), meter_provider=meters)
    demo_agent.flaky_calls["n"] = 0
    demo_agent.handle("where is my order? it was meant to come yesterday", "conv-1")
    demo_agent.flaky_calls["n"] = 1
    demo_agent.handle("has my parcel shipped yet", "conv-2")
    with ao.agent("support-agent"):  # a model missing from the price table, so the unpriced counter is sent too
        with ao.model_call("example-unlisted", provider="example") as m:
            m.record(input_tokens=12, output_tokens=3)
    meters.shutdown()  # sends the metrics
    tracer.shutdown()
    ao.configure(None)
    return run_id, list(spans.get_finished_spans())


def _get(path: str, **params) -> dict:
    url = f"{PROMETHEUS}{path}?{urllib.parse.urlencode(params, doseq=True)}"
    with urllib.request.urlopen(url, timeout=5) as r:
        body = json.load(r)
    if body.get("status") != "success":
        raise RuntimeError(f"{path}: {body}")
    return body["data"]


def stored_names(run_id: str, timeout_s: float = 20.0) -> list[str]:
    """Metric names Prometheus holds for this run, as it translated them."""
    deadline = time.monotonic() + timeout_s
    while True:
        names = _get("/api/v1/label/__name__/values", **{"match[]": f'{{instance="{run_id}"}}'})
        if any(n.startswith("gen_ai_") for n in names) or time.monotonic() > deadline:
            return names
        time.sleep(0.5)


# A metric name as Prometheus stores ours: counters end in _total, histograms in _bucket, _count and _sum.
# Label names (gen_ai_request_model and the like) have none of these endings.
STORED_NAME = r"\b((?:gen_ai|agent_otel)_[a-z_]+?_(?:total|bucket|count|sum))\b"


def rule_exprs(text: str) -> dict[str, str]:
    """alert name -> expression, read from the rules file without a YAML library."""
    rules, lines = {}, text.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"\s*- alert: (\S+)", line)
        if not m:
            continue
        for j in range(i + 1, len(lines)):
            e = re.match(r"(\s*)expr: ?(.*)", lines[j])
            if e:
                if e[2].strip() != "|":
                    rules[m[1]] = e[2].strip()
                else:
                    block = []
                    for k in range(j + 1, len(lines)):
                        if lines[k].strip() and len(lines[k]) - len(lines[k].lstrip()) <= len(e[1]):
                            break
                        block.append(lines[k].strip())
                    rules[m[1]] = " ".join(b for b in block if b)
                break
    return rules


def scoped(expr: str, run_id: str) -> str:
    """The expression with every metric selector limited to this run's series."""
    return re.sub(STORED_NAME + r"(\{)?", lambda m: f'{m[1]}{{instance="{run_id}"' + ("," if m[2] else "}"), expr)


def check(run_id: str, spans: list, names: list[str]) -> tuple[list[str], list[tuple]]:
    sel = f'instance="{run_id}"'
    problems, rows = [], []

    def find(metric: str, histogram: bool) -> str | None:
        stem = metric.replace(".", "_")
        if histogram:
            hits = [n for n in names if n.startswith(stem) and n.endswith("_count")]
        else:
            hits = [n for n in names if n.startswith(stem) and not n.endswith(("_bucket", "_sum", "_count"))]
        if len(hits) != 1:
            problems.append(f"{metric}: expected one stored name, found {hits}")
            return None
        return hits[0]

    def total(promql: str) -> float:
        result = _get("/api/v1/query", query=promql)["result"]
        return sum(float(r["value"][1]) for r in result)

    def compare(label: str, metric: str, histogram: bool, from_spans: float, extra: str = "") -> None:
        name = find(metric, histogram)
        if name is None:
            return
        stored = total(f"sum({name}{{{sel}{extra}}})")
        rows.append((label, name, from_spans, stored))
        if abs(stored - from_spans) > 1e-9 * max(1.0, abs(from_spans)):
            problems.append(f"{label}: spans say {from_spans}, Prometheus has {stored}")

    chats = [s for s in spans if s.name.startswith("chat ")]
    tools = [s for s in spans if s.name.startswith("execute_tool ")]
    agents = [s for s in spans if s.name.startswith("invoke_agent ")]
    compare("model calls", C.M_INFERENCE_DURATION, True, len(chats))
    compare("input tokens", C.M_USAGE_INPUT_TOKENS, False, sum(s.attributes[C.USAGE_INPUT_TOKENS] for s in chats))
    compare("output tokens", C.M_USAGE_OUTPUT_TOKENS, False,
            sum(s.attributes[C.USAGE_OUTPUT_TOKENS] for s in chats))
    compare("tool calls", C.M_EXECUTE_TOOL_DURATION, True, len(tools))
    compare("failed tool calls", C.M_EXECUTE_TOOL_DURATION, True,
            sum(1 for s in tools if C.ERROR_TYPE in s.attributes), extra=',error_type!=""')
    compare("agent runs", C.M_INVOKE_AGENT_DURATION, True, len(agents))
    compare("cost", C.COST, False, sum(s.attributes.get(C.TOTAL_COST, 0.0) for s in agents))
    compare("unpriced model calls", C.M_UNPRICED_CALLS, False, sum(1 for s in chats if C.COST not in s.attributes))
    calls = find(C.M_INVOKE_AGENT_INFERENCE_CALLS, True)
    if calls:
        made = total(f"sum({calls.removesuffix('_count')}_sum{{{sel}}})")
        rows.append(("model calls, per run sums", calls.removesuffix("_count") + "_sum", len(chats), made))
        if made != len(chats):
            problems.append(f"agent runs made {made} model calls by the metric, {len(chats)} by the spans")
    duration = find(C.M_INFERENCE_DURATION, True)
    if duration:
        bucket = duration.removesuffix("_count") + "_bucket"
        les = {r["metric"]["le"] for r in _get("/api/v1/query", query=f"{bucket}{{{sel}}}")["result"]}
        want = {repr(b) for b in C.SECONDS_BUCKETS} | {"+Inf"}
        if not want <= les:
            problems.append(f"{bucket}: bucket bounds {sorted(les)} miss {sorted(want - les)}")
    # The alert rules are unit tested on series written by hand. These make sure they also work on real ones:
    # every name they use was stored, and the two rules this run can trigger give the spans' answer.
    text = ALERTS.read_text()
    in_rules = set(re.findall(STORED_NAME, text))
    if not in_rules:
        problems.append(f"{ALERTS.name}: no metric names found")
    for name in sorted(in_rules - set(names)):
        problems.append(f"{ALERTS.name} uses {name}, which Prometheus did not store")
    rows.append((f"names in {ALERTS.name} stored", "", len(in_rules), len(in_rules & set(names))))
    rules = rule_exprs(text)
    unpriced = {s.attributes[C.REQUEST_MODEL] for s in chats if C.COST not in s.attributes}
    if "ModelCallsUnpriced" not in rules or "SpendAboveBudget" not in rules:
        problems.append(f"{ALERTS.name}: ModelCallsUnpriced or SpendAboveBudget is missing")
        return problems, rows
    fired = {r["metric"].get("gen_ai_request_model") for r in
             _get("/api/v1/query", query=scoped(rules["ModelCallsUnpriced"], run_id))["result"]}
    rows.append(("ModelCallsUnpriced fires", "", len(unpriced), len(fired & unpriced)))
    if fired != unpriced:
        problems.append(f"ModelCallsUnpriced fired for {sorted(fired)}, the spans have unpriced {sorted(unpriced)}")
    spend_expr = re.sub(r"\)\s*>\s*[0-9.]+\s*$", ")", rules["SpendAboveBudget"])  # the hour's spend, no threshold
    spent = total(scoped(spend_expr, run_id))
    cost = sum(s.attributes.get(C.TOTAL_COST, 0.0) for s in agents)
    rows.append(("SpendAboveBudget's hour", "", cost, spent))
    if abs(spent - cost) > 1e-9 * max(1.0, cost):
        problems.append(f"SpendAboveBudget counts {spent} for this run, the spans say {cost}")
    return problems, rows


def main() -> int:
    run_id, spans = export()
    names = stored_names(run_id)
    problems, rows = check(run_id, spans, names)
    ours = sorted(n for n in names if n.startswith(("gen_ai_", "agent_otel_")))
    print(f"Prometheus at {PROMETHEUS} stored {len(ours)} agent-otel series names for this run "
          f"({SERVICE}, instance {run_id}):")
    for n in ours:
        print("  ", n)
    print(f"{'check':28} {'from the spans':>15} {'in Prometheus':>15}")
    for label, _name, want, got in rows:
        print(f"{label:28} {want:15.6g} {got:15.6g}")
    for p in problems:
        print("  problem:", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
