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

**Token counts mean what the OpenTelemetry GenAI conventions say.** `input_tokens` is all input tokens, cached
ones included; `cache_read_input_tokens` and `cache_write_input_tokens` say how many of them were read from or
written to the provider's prompt cache. Some APIs report input without the cached parts (Anthropic's
`usage.input_tokens` is one), so pass the sum there:

```python
u = reply.usage
read, write = u.cache_read_input_tokens or 0, u.cache_creation_input_tokens or 0
m.record(input_tokens=u.input_tokens + read + write, output_tokens=u.output_tokens,
         cache_read_input_tokens=read, cache_write_input_tokens=write)
```

If the cache counts are bigger than `input_tokens`, they cannot be part of it: agent-otel adds them and warns, so
the cost is not silently too low.

Tracing never raises into your code. A tool result that cannot be turned into JSON (a circular list, an object
whose `__str__` fails) is recorded as a size-capped repr.

`setup` is the quick local path: spans go to a JSON lines file for `agent-otel summary` and `view`. If your
application already sets up OpenTelemetry, skip `setup`: agent-otel uses the global tracer provider, so its spans
join your existing traces and go wherever your exporters send them. `ao.configure(provider)` picks a provider
explicitly.

## Send it to Jaeger, or any OTLP backend

agent-otel only makes standard OpenTelemetry spans, so the stock OTLP exporter sends them anywhere. The demo agent,
exported to Jaeger 2.23 and opened in its trace view:

![The demo agent's trace in Jaeger](docs/jaeger.png)

Jaeger's GenAI view recognises the spans as they are: it counts the model and tool calls ("GenAI calls: 6") and
marks the agent, model and tool spans. The failed carrier call and its retry show as two tool spans.

```sh
./jaeger                         # the binary from a Jaeger 2.x release; OTLP on :4318, the UI on :16686
pip install -e '.[otlp]'
python examples/otlp_jaeger.py   # sends two runs, reads them back from Jaeger's API, checks every span
```

CI runs the same check against a real Jaeger on every push: both runs stored, all five kinds of span present,
token counts kept as integers, the run cost on the agent span.

## What each span carries

| span | name | attributes |
|---|---|---|
| agent run | `invoke_agent <name>` | `gen_ai.operation.name`, `gen_ai.agent.name`, `gen_ai.conversation.id`, and totals for everything inside it: `agent_otel.total.input_tokens`, `.output_tokens`, `.model_calls`, `.tool_calls`, `.cost`, and `.unpriced_calls` (model calls the cost leaves out because their model is not in your price table) |
| model call | `chat <model>` (kind CLIENT) | `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.request.max_tokens`, `gen_ai.request.temperature`, `gen_ai.response.model`, `gen_ai.response.finish_reasons`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.usage.cache_read.input_tokens`, `gen_ai.usage.cache_write.input_tokens`, `agent_otel.cost` |
| tool call | `execute_tool <name>` | `gen_ai.tool.name`, `gen_ai.tool.call.id`, `gen_ai.tool.type`; arguments and result only with `capture_content=True` |

An exception marks the span as an error with `error.type` and an exception event, and still propagates.

The GenAI names were read from `opentelemetry-semantic-conventions` 0.66b2, where they are marked as moved to the
separate [semantic-conventions-genai](https://github.com/open-telemetry/semantic-conventions-genai) repository.
They are defined in [conventions.py](src/agent_otel/conventions.py) rather than imported, so the next move does not
break anything. The deprecated `gen_ai.system` and `gen_ai.usage.prompt_tokens` are not used. Cache writes use
`gen_ai.usage.cache_write.input_tokens`, the name in the GenAI repository's registry as of 2026-11-05 (0.66b2 still
had `cache_creation`).

## Prices are yours

agent-otel ships no price list. Prices change, and a number from someone else's table is a guess with a dollar sign
on it. You give a table, per million tokens, with model name globs, and cost is computed only for models in it:

```json
{"currency": "USD",
 "models": {"local-*": {"input": 0, "output": 0},
            "example-hosted-small": {"input": 0.5, "output": 1.5, "cache_read": 0.05, "cache_write": 0.625}}}
```

`cache_read` and `cache_write` are optional; without them, cached tokens cost the same as other input. The values
in [examples/prices.example.json](examples/prices.example.json) are placeholders. A model that is not in your table
gets token counts but no cost, never a made-up one, and the run's total says how many calls it left out:
`agent-otel summary` prints `cost 0.0045 USD (1 of 2 model calls unpriced, not in it)`.

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

## TypeScript

`js/` is the same library for Node 22 and later: the same span names, attributes and run totals, written to the
same JSON lines, so `agent-otel summary` and `agent-otel view` read runs from both (the TypeScript tests feed their
output to the Python CLI in CI). About 2.7 us added per traced tool call on a desktop (Node 22), 3.5 to 4.2 us on CI
runners (Node 24 and 22), measured with `node js/bench/overhead.ts`.

```ts
import { agent, configure, modelCall, tool } from "agent-otel";

configure({ prices: { currency: "USD", models: { "my-model-*": { input: 0.5, output: 1.5 } } } });
const lookupOrder = tool("lookup_order", async (id: string) => orders.get(id));

await agent("support", async () => {
  await modelCall("my-model-small", async (call) => {
    const answer = await myClient.chat(messages);
    call.record({ inputTokens: answer.usage.input, outputTokens: answer.usage.output });
  });
  await lookupOrder("A1");
});
```

Run totals are kept in an `AsyncLocalStorage`, so two agents running at once never mix their counts (tested). With
no OpenTelemetry context manager registered, spans still nest under their agent; with one (the Node SDK registers
it), they join the application's own traces. It is not on npm (that needs an account): build it from the clone with
`cd js && npm ci && npm run compile` and install that folder.

Over OTLP it needs nothing of its own either: `node js/examples/otlp-jaeger.ts` sends the same pretend support agent
to Jaeger with the stock exporter, reads both runs back from Jaeger's API and makes the same checks as the Python
example. CI runs it against the real Jaeger after the Python one: 2 traces, 11 spans, token counts stored as
integers, the failed carrier call marked with its error type.

## Limits

- **You wrap the calls.** There is no automatic patching of model SDKs; two lines per call site, or `@ao.tool()`.
- **Threads need their context.** Totals and the current span follow asyncio tasks, but not new threads. In a
  thread pool, submit with `contextvars.copy_context().run`.
- **The GenAI conventions are still in development.** Names may change again; agent-otel will follow.
- **Content is off by default.** With `capture_content=True`, tool arguments and results are stored (capped at
  4,000 characters) and may contain personal data. A big result is cut while it is encoded, not after: a tool
  returning 10,000 documents costs 32 us to record instead of 4.8 ms, a million numbers 126 us instead of 29 ms
  (Python 3.13). The same text on every Python from 3.10.
- **The viewer is a static file** for looking at local runs, not a tracing backend.

## Development

```sh
pip install -e '.[test]'
pytest -q               # 29 tests, about 1 s
cd js && npm ci
npm run typecheck       # src, tests, bench and examples
npm test                # 7 tests; with AGENT_OTEL_PYTHON=../.venv/bin/python the Python CLI reads their output too
```

Tests use the SDK's in-memory exporter and a fake model. CI also runs the demo, the viewer and a short benchmark.

Built with AI pair programming.

## Licence

MIT
