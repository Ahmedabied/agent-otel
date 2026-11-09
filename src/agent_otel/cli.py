"""Command line: agent-otel summary | view"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path

from . import __version__
from .export import read_spans
from .view import render_html, summary_text, traces


def cmd_summary(a) -> int:
    runs = traces(read_spans(a.spans))
    if a.json:
        print(json.dumps([{k: v for k, v in t.items() if k != "rows"} for t in runs], indent=2))
    else:
        print(summary_text(runs) if runs else "no spans")
    return 0


def cmd_view(a) -> int:
    runs = traces(read_spans(a.spans))
    out = Path(a.output or Path(a.spans).with_suffix(".html"))
    out.write_text(render_html(runs, a.title or Path(a.spans).stem), encoding="utf-8")
    print(out)
    if a.open:
        webbrowser.open(out.resolve().as_uri())
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="agent-otel", description="Look at agent runs recorded as OpenTelemetry spans.")
    p.add_argument("--version", action="version", version=f"agent-otel {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summary", help="duration, calls, tokens, cost and the slowest step per run")
    s.add_argument("spans", help="JSON lines file written by agent_otel.setup(jsonl=...)")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_summary)
    v = sub.add_parser("view", help="write a self-contained HTML waterfall of each run")
    v.add_argument("spans")
    v.add_argument("-o", "--output", help="HTML file (default: next to the spans file)")
    v.add_argument("--title")
    v.add_argument("--open", action="store_true", help="open it in the browser")
    v.set_defaults(fn=cmd_view)
    a = p.parse_args(argv)
    try:
        return a.fn(a)
    except (OSError, ValueError) as e:
        print(f"agent-otel: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
