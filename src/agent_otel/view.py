"""Read spans back from a JSON lines file: a summary per run, and a self-contained HTML waterfall."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from . import conventions as C


def _kind(s: dict) -> str:
    op = s["attributes"].get(C.OPERATION_NAME)
    if op == C.INVOKE_AGENT:
        return "agent"
    if op == C.EXECUTE_TOOL:
        return "tool"
    if op in (C.CHAT, C.TEXT_COMPLETION, C.EMBEDDINGS, "generate_content"):
        return "model"
    return "other"


def traces(spans: list[dict]) -> list[dict]:
    """Group spans into runs, in start order, each span with its depth and offsets in ms."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for s in spans:
        groups[s["trace_id"]].append(s)
    out = []
    for tid, items in groups.items():
        ids = {s["span_id"] for s in items}
        children: dict[str | None, list[dict]] = defaultdict(list)
        for s in items:
            parent = s["parent_id"] if s["parent_id"] in ids else None
            children[parent].append(s)
        for v in children.values():
            v.sort(key=lambda s: (s["start_ns"], s["span_id"]))
        start = min(s["start_ns"] for s in items)
        end = max(s["end_ns"] or s["start_ns"] for s in items)
        rows: list[dict] = []

        def walk(parent: str | None, depth: int) -> None:
            for s in children.get(parent, []):
                a = s["attributes"]
                rows.append({
                    "name": s["name"], "kind": _kind(s), "depth": depth,
                    "offset_ms": (s["start_ns"] - start) / 1e6,
                    "duration_ms": ((s["end_ns"] or s["start_ns"]) - s["start_ns"]) / 1e6,
                    "input_tokens": a.get(C.USAGE_INPUT_TOKENS), "output_tokens": a.get(C.USAGE_OUTPUT_TOKENS),
                    "cost": a.get(C.COST), "error": s["status"] == "ERROR",
                    "attributes": a,
                })
                walk(s["span_id"], depth + 1)

        walk(None, 0)
        models = [r for r in rows if r["kind"] == "model"]
        costs = [r["cost"] for r in models if r["cost"] is not None]
        # Calls that reported usage but matched no price: the cost total leaves them out.
        unpriced = sum(1 for r in models if r["cost"] is None and (r["input_tokens"] is not None or r["output_tokens"] is not None))
        steps = [r for r in rows if r["kind"] in ("model", "tool")]
        slowest = max(steps, key=lambda r: r["duration_ms"], default=None)
        currency = next((s["attributes"].get(C.COST_CURRENCY) for s in items if s["attributes"].get(C.COST_CURRENCY)), "")
        out.append({
            "trace_id": tid,
            "name": rows[0]["name"] if rows else tid,
            "start_ns": start,
            "duration_ms": (end - start) / 1e6,
            "model_calls": len(models),
            "tool_calls": sum(1 for r in rows if r["kind"] == "tool"),
            "input_tokens": sum(r["input_tokens"] or 0 for r in models),
            "output_tokens": sum(r["output_tokens"] or 0 for r in models),
            "cost": round(sum(costs), 10) if costs else None,
            "unpriced_calls": unpriced if costs else 0,
            "currency": currency,
            "errors": sum(1 for r in rows if r["error"]),
            "slowest": {"name": slowest["name"], "duration_ms": slowest["duration_ms"]} if slowest else None,
            "rows": rows,
        })
    out.sort(key=lambda t: t["start_ns"])
    return out


def _count(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def summary_text(runs: list[dict]) -> str:
    lines = []
    for t in runs:
        cost = f"{t['cost']:.4f} {t['currency']}".strip() if t["cost"] is not None else "no prices"
        if t["unpriced_calls"]:
            cost += f" ({t['unpriced_calls']} of {t['model_calls']} model calls unpriced, not in it)"
        slow = f"{t['slowest']['name']} {t['slowest']['duration_ms']:.0f} ms" if t["slowest"] else "-"
        lines.append(f"{t['name']}  ({t['trace_id'][:8]})")
        lines.append(f"  {t['duration_ms']:.0f} ms, {_count(t['model_calls'], 'model call')}, "
                     f"{_count(t['tool_calls'], 'tool call')}, {t['input_tokens']} in / {t['output_tokens']} out "
                     f"tokens, cost {cost}, {_count(t['errors'], 'error')}")
        lines.append(f"  slowest step: {slow}")
    return "\n".join(lines)


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'">
<title>__TITLE__</title>
<style>
:root { --bg:#f6f5f2; --fg:#1b1b1a; --muted:#6a6862; --card:#fff; --line:#e4e1da; --model:#2f6fb0; --tool:#2c7a51;
  --agent:#8a8f98; --other:#a08a5a; --err:#c2410c; }
@media (prefers-color-scheme: dark) { :root { --bg:#111214; --fg:#ebebea; --muted:#9b9a96; --card:#1a1c1f; --line:#2b2e33;
  --model:#7fb0e6; --tool:#5cc28d; --agent:#8d99a8; --other:#c8b07a; --err:#f2773b; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1100px; margin: 0 auto; padding: 40px 24px; }
h1 { font-size: 28px; letter-spacing: -.02em; margin: 0 0 4px; }
.sub { color: var(--muted); margin: 0 0 28px; font: 13px ui-monospace, Menlo, monospace; }
.run { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 18px 20px; margin: 0 0 22px; }
.run h2 { font-size: 18px; margin: 0 0 10px; }
.stats { display: flex; flex-wrap: wrap; gap: 8px 22px; margin: 0 0 16px; color: var(--muted); font: 13px ui-monospace, Menlo, monospace; }
.stats b { color: var(--fg); font-weight: 600; }
.row { display: grid; grid-template-columns: minmax(220px, 34%) 1fr 76px 112px 86px; gap: 10px; align-items: center;
  padding: 3px 0; border-top: 1px solid var(--line); font-size: 13px; }
.row.head { color: var(--muted); border-top: 0; font-size: 12px; text-transform: uppercase; letter-spacing: .05em; }
.name { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; cursor: pointer; }
.name::before { content: ""; display: inline-block; width: 8px; height: 8px; border-radius: 2px; margin-right: 8px; background: var(--c); }
.track { position: relative; height: 12px; background: color-mix(in srgb, var(--line) 45%, transparent); border-radius: 3px; }
.bar { position: absolute; top: 0; bottom: 0; min-width: 2px; border-radius: 3px; background: var(--c); }
.num { text-align: right; font: 12px ui-monospace, Menlo, monospace; color: var(--muted); }
.k-model { --c: var(--model); } .k-tool { --c: var(--tool); } .k-agent { --c: var(--agent); } .k-other { --c: var(--other); }
.err { --c: var(--err); } .err .name { color: var(--err); }
pre { grid-column: 1 / -1; margin: 4px 0 8px; padding: 10px 12px; background: var(--bg); border-radius: 8px;
  font: 12px/1.5 ui-monospace, Menlo, monospace; white-space: pre-wrap; overflow-wrap: anywhere; }
.legend { color: var(--muted); font-size: 13px; margin-top: 8px; }
.legend span { margin-right: 14px; } .legend span::before { content: ""; display: inline-block; width: 8px; height: 8px;
  border-radius: 2px; margin-right: 6px; background: var(--c); }
</style>
</head>
<body>
<main>
  <h1>__TITLE__</h1>
  <p class="sub" id="sub"></p>
  <div id="runs"></div>
  <p class="legend"><span class="k-agent">agent</span><span class="k-model">model call</span><span class="k-tool">tool call</span><span class="err">error</span> Click a step to see its attributes.</p>
</main>
<script type="application/json" id="data">__DATA__</script>
<script>
"use strict";
const runs = JSON.parse(document.getElementById("data").textContent);
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const ms = (v) => v >= 1000 ? (v / 1000).toFixed(2) + " s" : v.toFixed(v < 10 ? 2 : 0) + " ms";
document.getElementById("sub").textContent = runs.length + (runs.length === 1 ? " run" : " runs") + " \\u00b7 times are wall clock, from the spans";
const box = document.getElementById("runs");
for (const t of runs) {
  const card = el("section", "run");
  card.append(el("h2", null, t.name));
  const stats = el("div", "stats");
  const stat = (label, value) => { const s = el("span", null, label + " "); s.append(el("b", null, value)); stats.append(s); };
  stat("duration", ms(t.duration_ms));
  stat("model calls", String(t.model_calls));
  stat("tool calls", String(t.tool_calls));
  stat("tokens", t.input_tokens + " in / " + t.output_tokens + " out");
  stat("cost", t.cost == null ? "no prices" : t.cost.toFixed(4) + " " + t.currency +
    (t.unpriced_calls ? " (" + t.unpriced_calls + " of " + t.model_calls + " calls unpriced)" : ""));
  if (t.errors) stat("errors", String(t.errors));
  if (t.slowest) stat("slowest", t.slowest.name + " " + ms(t.slowest.duration_ms));
  card.append(stats);
  const head = el("div", "row head");
  head.append(el("div", null, "step"), el("div", null, "timeline"), el("div", "num", "time"), el("div", "num", "tokens"), el("div", "num", "cost"));
  card.append(head);
  const total = Math.max(t.duration_ms, 0.001);
  for (const r of t.rows) {
    const row = el("div", "row k-" + r.kind + (r.error ? " err" : ""));
    const name = el("div", "name", r.name);
    name.style.paddingLeft = (r.depth * 16) + "px";
    const track = el("div", "track");
    const bar = el("div", "bar");
    bar.style.left = (100 * r.offset_ms / total) + "%";
    bar.style.width = (100 * r.duration_ms / total) + "%";
    track.append(bar);
    const tokens = r.input_tokens == null && r.output_tokens == null ? "" : (r.input_tokens || 0) + " / " + (r.output_tokens || 0);
    row.append(name, track, el("div", "num", ms(r.duration_ms)), el("div", "num", tokens), el("div", "num", r.cost == null ? "" : r.cost.toFixed(5)));
    let open = null;
    name.addEventListener("click", () => {
      if (open) { open.remove(); open = null; return; }
      open = el("pre", null, JSON.stringify(r.attributes, null, 2));
      row.after(open);
    });
    card.append(row);
  }
  box.append(card);
}
</script>
</body>
</html>
"""


def _html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def render_html(runs: list[dict], title: str = "agent runs") -> str:
    # The data sits inside a <script> tag: escape "<" so no value can close it early.
    data = json.dumps(runs, ensure_ascii=False, default=str).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return _PAGE.replace("__TITLE__", _html_escape(title)).replace("__DATA__", data)
