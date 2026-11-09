# agent-otel

An agent run is a chain of model calls and tool calls, and when it is slow or expensive you want to know which
step. agent-otel records each step as an OpenTelemetry span with the GenAI semantic conventions, adds token and
cost totals per run, and shows a run as a waterfall you can open in a browser.

![agent-otel viewer: two runs of a support agent, one with a failed tool call and a retry](docs/viewer.png)

## Try it in under a minute

```sh
git clone https://github.com/Ahmedabied/agent-otel && cd agent-otel
python3 -m venv .venv && . .venv/bin/activate && pip install -e .
python examples/demo_agent.py          # a pretend support agent; no real model is called
agent-otel summary runs/demo.jsonl
agent-otel view runs/demo.jsonl --open
```

```
invoke_agent support-agent  (464a8c2f)
  346 ms, 2 model calls, 3 tool calls, 839 in / 43 out tokens, cost 0.0003 USD, 1 errors
  slowest step: execute_tool track_parcel 91 ms
```

## Use it

```python
import agent_otel as ao

ao.setup("runs/today.jsonl", service="support-bot", prices="prices.json")

with ao.agent("support-bot", conversation_id=conv_id):
    with ao.model_call("qwen2.5:7b", provider="ollama", max_tokens=512) as m:
        reply = client.chat(...)
        m.record(input_tokens=reply.usage.prompt_tokens, output_tokens=reply.usage.completion_tokens,
                 finish_reasons=[reply.finish_reason])

@ao.tool()                      # every call of this function is a tool span, sync or async
def find_order(email: str) -> dict:
    ...
```

`setup` is the quick local path: spans go to a JSON lines file for `agent-otel summary` and `view`. If your
application already sets up OpenTelemetry, skip `setup`: agent-otel uses the global tracer provider, so its spans
join your existing traces and go wherever your exporters send them. `ao.configure(provider)` picks a provider
explicitly.

## What each span carries

| span | name | attributes |
|---|---|---|
| agent run | `invoke_agent <name>` | `gen_ai.operation.name`, `gen_ai.agent.name`, `gen_ai.conversation.id`, and totals for everything inside it: `agent_otel.total.input_tokens`, `.output_tokens`, `.model_calls`, `.tool_calls`, `.cost` |
| model call | `chat <model>` (kind CLIENT) | `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.request.max_tokens`, `gen_ai.request.temperature`, `gen_ai.response.model`, `gen_ai.response.finish_reasons`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.usage.cache_read.input_tokens`, `agent_otel.cost` |
| tool call | `execute_tool <name>` | `gen_ai.tool.name`, `gen_ai.tool.call.id`, `gen_ai.tool.type`; arguments and result only with `capture_content=True` |

An exception marks the span as an error with `error.type` and an exception event, and still propagates.

The GenAI names were read from `opentelemetry-semantic-conventions` 0.66b2, where they are marked as moved to the
separate [semantic-conventions-genai](https://github.com/open-telemetry/semantic-conventions-genai) repository.
They are defined in [conventions.py](src/agent_otel/conventions.py) rather than imported, so the next move does not
break anything. The deprecated `gen_ai.system` and `gen_ai.usage.prompt_tokens` are not used.

## Prices are yours

agent-otel ships no price list. Prices change, and a number from someone else's table is a guess with a dollar sign
on it. You give a table, per million tokens, with model name globs, and cost is computed only for models in it:

```json
{"currency": "USD",
 "models": {"local-*": {"input": 0, "output": 0},
            "example-hosted-small": {"input": 0.5, "output": 1.5, "cache_read": 0.05}}}
```

The values in [examples/prices.example.json](examples/prices.example.json) are placeholders. A model that is not
in your table gets token counts but no cost, never a made-up one.

## How much it costs to trace

`python bench/bench.py --calls 50000`, Python 3.12 on an Intel i5-14400F:

| | per call |
|---|---|
| bare function call | 0.04 us |
| tool span, tracing not set up | 5.5 us |
| tool span, exported to a JSON lines file | 21 us |
| model span with usage and cost, exported | 30 us |

104,001 spans were written in that run and none were dropped. Next to a model call that takes hundreds of
milliseconds, 30 microseconds does not show.

## Limits

- **You wrap the calls.** There is no automatic patching of model SDKs; two lines per call site, or `@ao.tool()`.
- **Threads need their context.** Totals and the current span follow asyncio tasks, but not new threads. In a
  thread pool, submit with `contextvars.copy_context().run`.
- **The GenAI conventions are still in development.** Names may change again; agent-otel will follow.
- **Content is off by default.** With `capture_content=True`, tool arguments and results are stored (capped at
  4,000 characters) and may contain personal data.
- **The viewer is a static file** for looking at local runs, not a tracing backend.

## Development

```sh
pip install -e '.[test]'
pytest -q               # 22 tests, about 1 s
```

Tests use the SDK's in-memory exporter and a fake model. CI also runs the demo, the viewer and a short benchmark.

Built with AI pair programming.

## Licence

MIT
