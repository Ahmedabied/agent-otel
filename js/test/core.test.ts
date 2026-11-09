import { test, afterEach } from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { context } from "@opentelemetry/api";
import { AsyncLocalStorageContextManager } from "@opentelemetry/context-async-hooks";
import { InMemorySpanExporter, SimpleSpanProcessor, TracerProvider } from "@opentelemetry/sdk-trace";
import type { ReadableSpan } from "@opentelemetry/sdk-trace";
import { agent, C, configure, content, JsonlExporter, modelCall, PriceError, Prices, tool, toolCall } from "../src/index.ts";

function setup(opts: Parameters<typeof configure>[0] = {}) {
  const exporter = new InMemorySpanExporter();
  const provider = new TracerProvider({ spanProcessors: [new SimpleSpanProcessor({ exporter })] });
  configure({ ...opts, tracerProvider: provider });
  const byName = () => Object.fromEntries(exporter.getFinishedSpans().map((s) => [s.name, s])) as Record<string, ReadableSpan>;
  return { exporter, byName, provider };
}

afterEach(() => configure());

const parentOf = (s: ReadableSpan) => s.parentSpanContext?.spanId;

async function supportRun() {
  const lookup = tool("lookup_order", async (id: string) => ({ id, status: "shipped" }));
  return agent("support", async () => {
    await modelCall("local-small", async (m) => m.record({ inputTokens: 1200, outputTokens: 40, cacheReadInputTokens: 1000 }),
      { provider: "ollama" });
    await lookup("A1");
    await modelCall("unknown-model", async (m) => m.record({ inputTokens: 100, outputTokens: 10 }));
  }, { conversationId: "c-1" });
}

test("an agent run: span names, kinds, parents and totals, the same as the Python side", async () => {
  const { byName } = setup({ prices: { currency: "USD", models: { "local-*": { input: 1, output: 2, cache_read: 0.1 } } } });
  await supportRun();
  const s = byName();
  const run = s["invoke_agent support"]!;
  const chat = s["chat local-small"]!;
  const toolSpan = s["execute_tool lookup_order"]!;
  assert.equal(chat.kind, 2); // CLIENT
  assert.equal(parentOf(chat), run.spanContext().spanId);
  assert.equal(parentOf(toolSpan), run.spanContext().spanId);
  assert.equal(chat.attributes[C.PROVIDER_NAME], "ollama");
  assert.equal(chat.attributes[C.USAGE_INPUT_TOKENS], 1200);
  assert.equal(chat.attributes[C.USAGE_CACHE_READ_INPUT_TOKENS], 1000);
  // 200 fresh at 1, 1000 cached at 0.1, 40 out at 2, per million
  assert.equal(chat.attributes[C.COST], (200 * 1 + 1000 * 0.1 + 40 * 2) / 1e6);
  assert.equal(run.attributes[C.TOTAL_INPUT_TOKENS], 1300);
  assert.equal(run.attributes[C.TOTAL_OUTPUT_TOKENS], 50);
  assert.equal(run.attributes[C.TOTAL_MODEL_CALLS], 2);
  assert.equal(run.attributes[C.TOTAL_TOOL_CALLS], 1);
  assert.equal(run.attributes[C.TOTAL_UNPRICED_CALLS], 1); // unknown-model is not in the table
  assert.equal(run.attributes[C.TOTAL_COST], 0.00038);
  assert.equal(run.attributes[C.CONVERSATION_ID], "c-1");
  assert.equal(toolSpan.attributes[C.TOOL_CALL_ARGUMENTS], undefined); // content is off by default
});

test("a failing tool marks its span and the error still reaches the caller", async () => {
  const { byName } = setup();
  const broken = tool("charge_card", async () => {
    throw new RangeError("card declined");
  });
  await assert.rejects(agent("pay", () => broken()), /card declined/);
  const s = byName()["execute_tool charge_card"]!;
  assert.equal(s.status.code, 2);
  assert.equal(s.status.message, "card declined");
  assert.equal(s.attributes[C.ERROR_TYPE], "RangeError");
  assert.equal(byName()["invoke_agent pay"]!.status.code, 2);
});

test("two runs at the same time keep their own totals", async () => {
  const { byName } = setup();
  const slow = (ms: number) => new Promise((r) => setTimeout(r, ms));
  await Promise.all([
    agent("a", async () => {
      await modelCall("m", async (c) => { await slow(5); c.record({ inputTokens: 10, outputTokens: 1 }); });
      await modelCall("m", async (c) => c.record({ inputTokens: 10, outputTokens: 1 }));
    }),
    agent("b", async () => {
      await modelCall("m", async (c) => c.record({ inputTokens: 7, outputTokens: 3 }));
    }),
  ]);
  assert.equal(byName()["invoke_agent a"]!.attributes[C.TOTAL_INPUT_TOKENS], 20);
  assert.equal(byName()["invoke_agent a"]!.attributes[C.TOTAL_MODEL_CALLS], 2);
  assert.equal(byName()["invoke_agent b"]!.attributes[C.TOTAL_INPUT_TOKENS], 7);
});

test("joins the application's own tracing when it has a context manager", async () => {
  const manager = new AsyncLocalStorageContextManager().enable();
  context.setGlobalContextManager(manager);
  try {
    const { exporter, provider } = setup();
    const t = provider.getTracer("app"); // the application's own tracer, same provider
    await t.startActiveSpan("http GET /help", async (outer) => {
      await agent("support", () => toolCall("search", async () => {
        t.startSpan("db query").end(); // the app's own span inside our tool span
      }));
      outer.end();
    });
    const spans = Object.fromEntries(exporter.getFinishedSpans().map((s) => [s.name, s]));
    assert.equal(parentOf(spans["invoke_agent support"]!), spans["http GET /help"]?.spanContext().spanId);
    assert.equal(parentOf(spans["db query"]!), spans["execute_tool search"]!.spanContext().spanId);
  } finally {
    context.disable();
  }
});

test("content: off by default, capped when on, and never thrown into the app", async () => {
  const { byName } = setup({ captureContent: true, maxContentChars: 50 });
  const echo = tool("echo", async (x: unknown) => x);
  const loop: unknown[] = [];
  loop.push(loop);
  await echo(loop);
  assert.match(String(byName()["execute_tool echo"]!.attributes[C.TOOL_CALL_RESULT]), /Circular/);
  configure({ captureContent: true, maxContentChars: 50 });
  assert.equal(content({ path: "/tmp/x", n: 3 }), '{"path":"/tmp/x","n":3}');
  assert.equal(content("x".repeat(100)), "x".repeat(50) + "... [100 chars]");
  const big = Array.from({ length: 1_000_000 }, (_, i) => i);
  const t0 = performance.now();
  const head = content(big);
  const ms = performance.now() - t0;
  assert.equal(head, JSON.stringify(big).slice(0, 50) + "... [more than 50 chars]");
  assert.ok(ms < 50, `a million numbers took ${ms.toFixed(1)} ms to cut`);
  let deep: unknown = 0;
  for (let i = 0; i < 100_000; i++) deep = [deep];
  assert.ok(content(deep).length <= 50 + 30);
});

test("prices: globs, first match wins, bad tables refused", () => {
  const p = new Prices({ models: { "gpt-[45]*": { input: 2, output: 8 }, "*": { input: 1, output: 1 } } });
  assert.equal(p.lookup("gpt-4o")?.input, 2);
  assert.equal(p.lookup("gpt-3.5")?.input, 1);
  assert.equal(p.cost("anything", 1_000_000, 0), 1);
  assert.equal(new Prices({ models: { "a.b": { input: 1, output: 1 } } }).lookup("axb"), null); // dot is literal
  assert.throws(() => new Prices({ models: { x: { input: -1, output: 1 } } }), PriceError);
  assert.throws(() => new Prices({} as never), PriceError);
});

test("JSON lines in the Python format, read by the Python summary when it is installed", async () => {
  const dir = mkdtempSync(join(tmpdir(), "agent-otel-"));
  const file = join(dir, "run.jsonl");
  const provider = new TracerProvider({ spanProcessors: [new SimpleSpanProcessor({ exporter: new JsonlExporter(file) as never })] });
  configure({ tracerProvider: provider, prices: { models: { "local-*": { input: 1, output: 2, cache_read: 0.1 } } } });
  await supportRun();
  const lines = readFileSync(file, "utf8").trim().split("\n").map((l) => JSON.parse(l));
  assert.equal(lines.length, 4);
  const run = lines.find((l) => l.name === "invoke_agent support");
  assert.equal(run.kind, "INTERNAL");
  assert.equal(run.parent_id, null);
  assert.equal(run.status, "UNSET");
  assert.match(readFileSync(file, "utf8"), /"start_ns":\d{19}/); // exact nanoseconds, as Python writes them
  const python = process.env.AGENT_OTEL_PYTHON;
  if (!python) return; // CI sets it, next to an installed Python agent-otel
  const out = execFileSync(python, ["-m", "agent_otel", "summary", file], { encoding: "utf8" });
  assert.match(out, /invoke_agent support/);
  assert.match(out, /2 model calls, 1 tool calls?/);
  assert.match(out, /1300 in \/ 50 out tokens/);
});
