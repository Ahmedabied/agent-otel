"""What does tracing add to one tool or model call?

Times the same tiny function bare, wrapped with tracing off (no provider set up), and wrapped with
spans going through the SDK's BatchSpanProcessor to a JSON lines file.

    python bench/bench.py --calls 100000
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import agent_otel as ao  # noqa: E402


def per_call_us(fn, n):
    for _ in range(min(n, 2000)):
        fn()
    t0 = time.perf_counter_ns()
    for _ in range(n):
        fn()
    return (time.perf_counter_ns() - t0) / n / 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=100_000)
    n = ap.parse_args().calls

    def work():
        return 1

    traced = ao.tool("work")(work)

    def model():
        with ao.model_call("local-1", provider="bench") as m:
            m.record(input_tokens=100, output_tokens=10)

    bare = per_call_us(work, n)
    ao.configure(None)
    off = per_call_us(traced, n)
    with tempfile.TemporaryDirectory() as tmp:
        provider = ao.setup(str(Path(tmp) / "bench.jsonl"), prices={"models": {"local-*": {"input": 0, "output": 0}}})
        on = per_call_us(traced, n)
        with ao.agent("bench"):
            on_model = per_call_us(model, n)
        provider.shutdown()
        spans = sum(1 for _ in open(Path(tmp) / "bench.jsonl"))
    print(f"{n} calls each, python {sys.version.split()[0]}")
    print(f"bare function call              {bare:8.2f} us")
    print(f"tool span, tracing off          {off:8.2f} us")
    print(f"tool span, to a JSONL file      {on:8.2f} us")
    print(f"model span with usage and cost  {on_model:8.2f} us")
    print(f"spans written: {spans} (the batch processor drops spans only if its queue overflows)")


if __name__ == "__main__":
    main()
