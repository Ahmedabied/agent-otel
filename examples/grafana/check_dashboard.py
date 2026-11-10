"""Run every query of the dashboard and fail if a panel would be empty.

    python examples/grafana/check_dashboard.py --grafana http://localhost:3000      # through Grafana itself
    python examples/grafana/check_dashboard.py --prometheus http://localhost:9090   # straight to Prometheus

Through Grafana, it first reads the provisioned dashboard back (same panels and titles as the file), then
sends each panel's queries to Grafana's query API, so Grafana's own handling ($__rate_interval, the data
source) is what gets tested. A panel passes when every query returns at least one finite number: a
histogram_quantile over empty buckets gives NaN, which would show as a blank panel.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

DASHBOARD = Path(__file__).parent / "agent-otel.json"


def _call(url: str, body: dict | None = None, auth: str | None = None) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json", "Accept": "application/json"})
    if auth:
        req.add_header("Authorization", "Basic " + base64.b64encode(auth.encode()).decode())
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def finite(values) -> int:
    return sum(1 for v in values if isinstance(v, (int, float)) and math.isfinite(v))


def via_grafana(base: str, auth: str | None, panels: list[dict], minutes: int) -> dict[tuple[int, str], tuple[int, int]]:
    out = {}
    for p in panels:
        queries = [dict(q, intervalMs=15000, maxDataPoints=200) for q in p["targets"]]
        res = _call(f"{base}/api/ds/query", {"from": f"now-{minutes}m", "to": "now", "queries": queries}, auth)
        for q in p["targets"]:
            r = res["results"][q["refId"]]
            if r.get("error"):
                raise RuntimeError(f"panel {p['title']!r} query {q['refId']}: {r['error']}")
            frames = r.get("frames", [])
            points = sum(finite(f["data"]["values"][-1]) for f in frames if f.get("data", {}).get("values"))
            out[(p["id"], q["refId"])] = (len(frames), points)
    return out


def via_prometheus(base: str, panels: list[dict], minutes: int) -> dict[tuple[int, str], tuple[int, int]]:
    out, now = {}, time.time()
    for p in panels:
        for q in p["targets"]:
            expr = q["expr"].replace("$__rate_interval", "1m")  # what Grafana does for a 15 s scrape interval
            params = urllib.parse.urlencode({"query": expr, "start": now - minutes * 60, "end": now, "step": 15})
            res = _call(f"{base}/api/v1/query_range?{params}")["data"]["result"]
            points = sum(finite(float(v) for _, v in s["values"]) for s in res)
            out[(p["id"], q["refId"])] = (len(res), points)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--grafana")
    g.add_argument("--prometheus")
    ap.add_argument("--minutes", type=int, default=10)
    a = ap.parse_args()
    dash = json.loads(DASHBOARD.read_text())
    panels = dash["panels"]
    problems = []
    if a.grafana:
        auth = os.environ.get("GRAFANA_AUTH")  # user:password, when anonymous access is off
        stored = _call(f"{a.grafana}/api/dashboards/uid/{dash['uid']}", auth=auth)["dashboard"]
        want = [(p["id"], p["title"]) for p in panels]
        got = [(p["id"], p["title"]) for p in stored.get("panels", [])]
        if got != want:
            problems.append(f"Grafana has panels {got}, the file has {want}")
        results = via_grafana(a.grafana, auth, panels, a.minutes)
    else:
        results = via_prometheus(a.prometheus, panels, a.minutes)
    print(f"{'panel':40} {'query':>5} {'series':>6} {'finite points':>13}")
    for p in panels:
        for q in p["targets"]:
            series, points = results[(p["id"], q["refId"])]
            print(f"{p['title'][:40]:40} {q['refId']:>5} {series:6d} {points:13d}")
            if points == 0:
                problems.append(f"panel {p['title']!r} query {q['refId']} has no finite value")
    for pr in problems:
        print("  problem:", pr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
