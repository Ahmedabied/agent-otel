"""A pretend support agent, traced with agent-otel. No real model is called: the "model" is a
deterministic fake that sleeps a little per token, so the traces look like a real run.

    python examples/demo_agent.py            # writes runs/demo.jsonl
    agent-otel summary runs/demo.jsonl
    agent-otel view runs/demo.jsonl --open

The prices in prices.example.json are placeholders. Put in your own, with the date you read them.
"""

import time
from pathlib import Path

import agent_otel as ao

HERE = Path(__file__).parent


class FakeModel:
    """Answers from a script. Token counts are words times 1.3, like a rough tokenizer."""

    def __init__(self, name: str, ms_per_token: float):
        self.name = name
        self.ms_per_token = ms_per_token

    def chat(self, prompt: str, reply: str) -> tuple[str, int, int]:
        tokens_in = int(len(prompt.split()) * 1.3) + 400  # plus a system prompt
        tokens_out = int(len(reply.split()) * 1.3)
        time.sleep(0.02 + tokens_out * self.ms_per_token / 1000)
        return reply, tokens_in, tokens_out


small = FakeModel("example-hosted-small", ms_per_token=1.5)
local = FakeModel("local-qwen-7b", ms_per_token=4.0)
flaky_calls = {"n": 0}


@ao.tool()
def find_order(email: str) -> dict:
    time.sleep(0.035)
    return {"order": "A-1042", "status": "in transit", "carrier": "parcel co"}


@ao.tool()
def track_parcel(order: str) -> dict:
    time.sleep(0.09)
    flaky_calls["n"] += 1
    if flaky_calls["n"] == 1:
        raise TimeoutError("carrier API did not answer in 80 ms")
    return {"eta": "Thursday"}


def ask(model: FakeModel, prompt: str, reply: str) -> str:
    with ao.model_call(model.name, provider="example", max_tokens=512) as m:
        text, tin, tout = model.chat(prompt, reply)
        m.record(input_tokens=tin, output_tokens=tout, finish_reasons=["stop"])
        return text


def handle(question: str, conversation: str) -> str:
    with ao.agent("support-agent", conversation_id=conversation):
        ask(local, question, "intent: order status. need email lookup then tracking")
        order = find_order("someone@example.org")
        for attempt in range(2):
            try:
                eta = track_parcel(order["order"])
                break
            except TimeoutError:
                if attempt == 1:
                    raise
        return ask(small, f"{question} {order} {eta}",
                   "Your order A-1042 is on its way with Parcel Co and should arrive on Thursday. "
                   "I will send you the tracking link by email as well.")


if __name__ == "__main__":
    out = Path("runs/demo.jsonl")
    out.unlink(missing_ok=True)
    provider = ao.setup(str(out), service="demo-support", prices=str(HERE / "prices.example.json"))
    print(handle("where is my order? it was meant to come yesterday", "conv-1"))
    flaky_calls["n"] = 1  # the second run has no carrier timeout
    print(handle("has my parcel shipped yet", "conv-2"))
    provider.shutdown()
    print(f"spans written to {out}")
