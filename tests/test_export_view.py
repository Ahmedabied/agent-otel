import json
import subprocess
import sys

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

import agent_otel as ao
from agent_otel import JsonlExporter, read_spans
from agent_otel.view import render_html, traces


def record(path, prices=None):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(JsonlExporter(path)))
    ao.configure(provider, prices=prices)
    try:
        with ao.agent("bot"):
            with ao.model_call("local-1") as m:
                m.record(input_tokens=10, output_tokens=5)
            try:
                with ao.tool_call("</script><b>x"):
                    raise ValueError("boom")
            except ValueError:
                pass
        with ao.agent("bot-2"):
            pass
    finally:
        ao.configure(None)
        provider.shutdown()


def test_jsonl_roundtrip_and_traces(tmp_path):
    f = tmp_path / "run.jsonl"
    record(f, prices={"models": {"local-*": {"input": 1, "output": 1}}})
    spans = read_spans(f)
    assert len(spans) == 4 and all(len(s["trace_id"]) == 32 for s in spans)
    runs = traces(spans)
    assert [r["name"] for r in runs] == ["invoke_agent bot", "invoke_agent bot-2"]
    bot = runs[0]
    assert bot["model_calls"] == 1 and bot["tool_calls"] == 1 and bot["errors"] == 1
    assert bot["input_tokens"] == 10 and bot["cost"] == 15 / 1e6
    assert [r["depth"] for r in bot["rows"]] == [0, 1, 1]
    assert runs[1]["cost"] is None and runs[1]["model_calls"] == 0


def test_html_cannot_be_broken_by_span_data(tmp_path):
    f = tmp_path / "run.jsonl"
    record(f)
    html = render_html(traces(read_spans(f)), title="<i>t</i>")
    data = html.split('<script type="application/json" id="data">', 1)[1].split("</script>", 1)[0]
    assert "</script>" not in data and "<b>" not in data  # the tool name is escaped inside the data block
    assert json.loads(data)[0]["rows"][2]["name"] == "execute_tool </script><b>x"
    assert "<title>&lt;i&gt;t&lt;/i&gt;</title>" in html


def test_cli(tmp_path):
    f = tmp_path / "run.jsonl"
    record(f)
    r = subprocess.run([sys.executable, "-m", "agent_otel", "summary", str(f)], capture_output=True, text=True)
    assert r.returncode == 0 and "1 model call, 1 tool call, " in r.stdout and ", 1 error\n" in r.stdout
    assert "no prices" in r.stdout
    r = subprocess.run([sys.executable, "-m", "agent_otel", "summary", str(f), "--json"], capture_output=True, text=True)
    assert json.loads(r.stdout)[0]["errors"] == 1
    r = subprocess.run([sys.executable, "-m", "agent_otel", "view", str(f), "-o", str(tmp_path / "v.html")],
                       capture_output=True, text=True)
    assert r.returncode == 0 and (tmp_path / "v.html").read_text().startswith("<!doctype html>")
    r = subprocess.run([sys.executable, "-m", "agent_otel", "summary", str(tmp_path / "nope.jsonl")], capture_output=True, text=True)
    assert r.returncode == 2


def test_demo_runs():
    r = subprocess.run([sys.executable, "examples/demo_agent.py"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    runs = traces(read_spans("runs/demo.jsonl"))
    assert len(runs) == 2 and runs[0]["errors"] == 1 and runs[0]["cost"] > 0


def test_unpriced_calls_are_shown(tmp_path):
    f = tmp_path / "run.jsonl"
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(JsonlExporter(f)))
    ao.configure(provider, prices={"models": {"local-*": {"input": 1, "output": 1}}})
    try:
        with ao.agent("bot"):
            with ao.model_call("local-1") as m:
                m.record(input_tokens=10, output_tokens=5)
            with ao.model_call("big-unpriced") as m:
                m.record(input_tokens=200_000, output_tokens=5_000)
    finally:
        ao.configure(None)
        provider.shutdown()
    run = traces(read_spans(f))[0]
    assert run["cost"] == 15 / 1e6 and run["unpriced_calls"] == 1
    from agent_otel.view import summary_text
    text = summary_text([run])
    assert "1 of 2 model calls unpriced" in text and ", 2 model calls, " in text
    assert text.splitlines()[1].endswith(", 0 errors")
    assert "calls unpriced" in render_html([run])
