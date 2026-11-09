"""Write finished spans to a JSON lines file, one span per line, and read them back."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult


def span_to_dict(s: ReadableSpan) -> dict[str, Any]:
    ctx = s.get_span_context()
    return {
        "trace_id": format(ctx.trace_id, "032x"),
        "span_id": format(ctx.span_id, "016x"),
        "parent_id": format(s.parent.span_id, "016x") if s.parent else None,
        "name": s.name,
        "kind": s.kind.name,
        "start_ns": s.start_time,
        "end_ns": s.end_time,
        "status": s.status.status_code.name,
        "error": s.status.description,
        "attributes": {k: list(v) if isinstance(v, tuple) else v for k, v in (s.attributes or {}).items()},
        "events": [{"name": e.name, "time_ns": e.timestamp,
                    "attributes": {k: list(v) if isinstance(v, tuple) else v for k, v in (e.attributes or {}).items()}}
                   for e in s.events],
        "service": (s.resource.attributes or {}).get("service.name") if s.resource else None,
    }


class JsonlExporter(SpanExporter):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", encoding="utf-8")
        self._lock = threading.Lock()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        lines = "".join(json.dumps(span_to_dict(s), ensure_ascii=False, default=str) + "\n" for s in spans)
        with self._lock:
            if self._f.closed:
                return SpanExportResult.FAILURE
            self._f.write(lines)
            self._f.flush()
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        with self._lock:
            if not self._f.closed:
                self._f.close()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def read_spans(path: str | Path) -> list[dict[str, Any]]:
    out = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    out.append(json.loads(line))
                except ValueError as e:
                    raise ValueError(f"{path}:{n}: not a span line: {e}") from None
    return out
