import { test, afterEach } from "node:test";
import assert from "node:assert/strict";
import type { Attributes, MeterProvider as ApiMeterProvider } from "@opentelemetry/api";
import { MeterProvider, MetricReader } from "@opentelemetry/sdk-metrics";
import type { DataPoint, Histogram } from "@opentelemetry/sdk-metrics";
import { InMemorySpanExporter, SimpleSpanProcessor, TracerProvider } from "@opentelemetry/sdk-trace";
import type { ReadableSpan } from "@opentelemetry/sdk-trace";
import { agent, C, configure, modelCall, tool, toolCall } from "../src/index.ts";
import type { PriceTable } from "../src/index.ts";

class Reader extends MetricReader {
  protected override async onForceFlush(): Promise<void> {}
  protected override async onShutdown(): Promise<void> {}
}

const PRICES: PriceTable = { currency: "USD", models: { "gpt-x*": { input: 1, output: 2, cache_read: 0.1 } } };

// Every attribute a metric may carry. Ids and content must never be here: each value would be its own series.
const ALLOWED = new Set([C.OPERATION_NAME, C.PROVIDER_NAME, C.REQUEST_MODEL, C.RESPONSE_MODEL, C.ERROR_TYPE,
  C.TOKEN_MODALITY, C.TOOL_NAME, C.TOOL_TYPE, C.AGENT_NAME, C.COST_CURRENCY]);

type Point = DataPoint<number | Histogram>;

function setup(opts: { prices?: PriceTable; captureContent?: boolean } = {}) {
  const reader = new Reader();
  const meterProvider = new MeterProvider({ readers: [reader] });
  const exporter = new InMemorySpanExporter();
  const tracerProvider = new TracerProvider({ spanProcessors: [new SimpleSpanProcessor({ exporter })] });
  configure({ ...opts, tracerProvider, meterProvider });
  const collect = async () => {
    const { resourceMetrics } = await reader.collect();
    const out: Record<string, Point[]> = {};
    for (const sm of resourceMetrics.scopeMetrics) for (const m of sm.metrics) out[m.descriptor.name] = m.dataPoints as Point[];
    return out;
  };
  const span = (name: string) => {
    const found = exporter.getFinishedSpans().filter((s) => s.name === name);
    assert.equal(found.length, 1, name);
    return found[0]!;
  };
  return { collect, span, exporter };
}

afterEach(() => configure());

const one = (points: Point[] | undefined, attrs: Attributes = {}) => {
  const found = (points ?? []).filter((p) => Object.entries(attrs).every(([k, v]) => p.attributes[k] === v));
  assert.equal(found.length, 1, JSON.stringify((points ?? []).map((p) => p.attributes)));
  return found[0]!;
};
const hist = (p: Point) => p.value as Histogram;
const spanSeconds = (s: ReadableSpan) => (s.duration[0] * 1e9 + s.duration[1]) / 1e9;

test("a model call gives the inference metrics, with the span's own numbers and duration", async () => {
  const { collect, span } = setup({ prices: PRICES });
  await agent("support", () => modelCall("gpt-x", async (m) => m.record({ inputTokens: 1000, outputTokens: 50,
    cacheReadInputTokens: 400, reasoningOutputTokens: 20, responseModel: "gpt-x-2026", responseId: "r-1" }),
  { provider: "openai" }), { conversationId: "c-1" });
  const m = await collect();
  const base = { [C.OPERATION_NAME]: "chat", [C.PROVIDER_NAME]: "openai", [C.REQUEST_MODEL]: "gpt-x",
    [C.RESPONSE_MODEL]: "gpt-x-2026" };
  const d = one(m[C.M_INFERENCE_DURATION]);
  assert.deepEqual(d.attributes, base);
  assert.equal(hist(d).count, 1);
  assert.equal(hist(d).sum, spanSeconds(span("chat gpt-x")));
  assert.deepEqual(hist(d).buckets.boundaries, C.SECONDS_BUCKETS);
  assert.equal(hist(one(m[C.M_INFERENCE_INPUT_TOKENS])).sum, 1000);
  assert.deepEqual(hist(one(m[C.M_INFERENCE_INPUT_TOKENS])).buckets.boundaries, C.TOKEN_BUCKETS);
  assert.equal(hist(one(m[C.M_INFERENCE_OUTPUT_TOKENS])).sum, 50);
  const input = one(m[C.M_USAGE_INPUT_TOKENS]);
  assert.deepEqual(input.attributes, { ...base, [C.TOKEN_MODALITY]: "unknown" });
  assert.equal(input.value, 1000);
  assert.equal(one(m[C.M_USAGE_CACHE_READ_INPUT_TOKENS]).value, 400);
  assert.equal(m[C.M_USAGE_CACHE_WRITE_INPUT_TOKENS]?.length ?? 0, 0);
  const reasoning = one(m[C.M_USAGE_REASONING_OUTPUT_TOKENS]);
  assert.equal(reasoning.value, 20);
  assert.deepEqual(reasoning.attributes, base);
  const cost = one(m[C.COST]);
  assert.ok(Math.abs((cost.value as number) - (span("chat gpt-x").attributes[C.COST] as number)) < 1e-15);
  assert.equal(cost.attributes[C.COST_CURRENCY], "USD");
});

test("metrics never carry ids or content", async () => {
  const { collect } = setup({ prices: PRICES, captureContent: true });
  await agent("support", async () => {
    await modelCall("gpt-x", async (m) => m.record({ inputTokens: 10, outputTokens: 5, responseId: "resp-secret" }));
    await toolCall("lookup", async (t) => t.result({ order: "secret-order" }), { callId: "call-secret", arguments: { email: "secret-email" } });
  }, { conversationId: "conv-secret", agentId: "agent-secret" });
  const seen = new Set<string>();
  for (const points of Object.values(await collect())) {
    for (const p of points) {
      for (const [k, v] of Object.entries(p.attributes)) {
        seen.add(k);
        assert.ok(!String(v).includes("secret"), `${k}=${String(v)}`);
      }
    }
  }
  assert.deepEqual([...seen].filter((k) => !ALLOWED.has(k)), []);
});

test("a failed call records zero tokens with its error and adds to no counter", async () => {
  const { collect } = setup({ prices: PRICES });
  class TimeoutError extends Error {}
  await assert.rejects(modelCall("gpt-x", async () => { throw new TimeoutError("slow"); }, { provider: "openai" }));
  const m = await collect();
  assert.equal(one(m[C.M_INFERENCE_DURATION]).attributes[C.ERROR_TYPE], "TimeoutError");
  assert.equal(hist(one(m[C.M_INFERENCE_INPUT_TOKENS], { [C.ERROR_TYPE]: "TimeoutError" })).sum, 0);
  for (const name of [C.M_USAGE_INPUT_TOKENS, C.M_USAGE_OUTPUT_TOKENS, C.COST, C.M_UNPRICED_CALLS]) {
    assert.equal(m[name]?.length ?? 0, 0, name);
  }
});

test("record() called twice counts the final numbers once, on the span, the run and the metrics", async () => {
  const { collect, span } = setup({ prices: PRICES });
  await agent("streamer", () => modelCall("gpt-x", async (m) => {
    m.record({ inputTokens: 100 });
    m.record({ outputTokens: 40 });
  }, { provider: "openai" }));
  const chat = span("chat gpt-x");
  const run = span("invoke_agent streamer");
  assert.equal(chat.attributes[C.USAGE_INPUT_TOKENS], 100);
  assert.equal(chat.attributes[C.USAGE_OUTPUT_TOKENS], 40);
  assert.equal(run.attributes[C.TOTAL_INPUT_TOKENS], 100);
  assert.equal(run.attributes[C.TOTAL_OUTPUT_TOKENS], 40);
  assert.ok(Math.abs((chat.attributes[C.COST] as number) - (100 * 1 + 40 * 2) / 1e6) < 1e-15);
  assert.equal(run.attributes[C.TOTAL_COST], chat.attributes[C.COST]);
  const m = await collect();
  assert.equal(hist(one(m[C.M_INFERENCE_INPUT_TOKENS])).count, 1);
  assert.equal(one(m[C.M_USAGE_INPUT_TOKENS]).value, 100);
  assert.equal(one(m[C.M_USAGE_OUTPUT_TOKENS]).value, 40);
});

test("embeddings get their own metrics, not the inference ones", async () => {
  const { collect, span } = setup();
  await modelCall("embed-x", async (m) => m.record({ inputTokens: 64 }), { provider: "openai", operation: "embeddings" });
  const m = await collect();
  assert.equal(hist(one(m[C.M_EMBEDDINGS_INPUT_TOKENS])).sum, 64);
  assert.equal(hist(one(m[C.M_EMBEDDINGS_DURATION])).sum, spanSeconds(span("embeddings embed-x")));
  for (const name of [C.M_INFERENCE_DURATION, C.M_INFERENCE_INPUT_TOKENS, C.M_USAGE_INPUT_TOKENS]) {
    assert.equal(m[name]?.length ?? 0, 0, name);
  }
});

test("unpriced calls are counted beside the cost", async () => {
  const { collect } = setup({ prices: PRICES });
  await modelCall("gpt-x", async (m) => m.record({ inputTokens: 10, outputTokens: 1 }), { provider: "openai" });
  await modelCall("mystery-model", async (m) => m.record({ inputTokens: 10, outputTokens: 1 }), { provider: "other" });
  const m = await collect();
  assert.equal(one(m[C.M_UNPRICED_CALLS]).value, 1);
  assert.equal(one(m[C.M_UNPRICED_CALLS]).attributes[C.REQUEST_MODEL], "mystery-model");
  assert.equal(one(m[C.COST]).attributes[C.REQUEST_MODEL], "gpt-x");
});

test("tool durations name the agent and the error", async () => {
  const { collect, exporter } = setup();
  const lookup = tool("lookup", async (id: string) => ({ id }));
  await agent("support", async () => {
    await lookup("A1");
    await assert.rejects(toolCall("refund", async () => { throw new RangeError("no such order"); }));
  });
  await lookup("A2"); // outside any agent
  const m = await collect();
  const refund = one(m[C.M_EXECUTE_TOOL_DURATION], { [C.TOOL_NAME]: "refund" });
  assert.deepEqual(refund.attributes, { [C.TOOL_NAME]: "refund", [C.TOOL_TYPE]: "function", [C.AGENT_NAME]: "support",
    [C.ERROR_TYPE]: "RangeError" });
  const refundSpan = exporter.getFinishedSpans().find((s) => s.name === "execute_tool refund")!;
  assert.equal(hist(refund).sum, spanSeconds(refundSpan));
  assert.equal(hist(one(m[C.M_EXECUTE_TOOL_DURATION], { [C.TOOL_NAME]: "lookup", [C.AGENT_NAME]: "support" })).count, 1);
  const outside = m[C.M_EXECUTE_TOOL_DURATION]!.filter((p) => p.attributes[C.TOOL_NAME] === "lookup" && !(C.AGENT_NAME in p.attributes));
  assert.equal(outside.length, 1);
});

test("each call is counted once across nested agents; an entrypoint that only delegates reports none", async () => {
  const { collect, span } = setup();
  const call = () => modelCall("gpt-x", async (m) => m.record({ inputTokens: 1, outputTokens: 1 }), { provider: "openai" });
  await agent("router", async () => {
    await agent("planner", async () => {
      for (let i = 0; i < 3; i++) await call();
      await toolCall("search", async () => null);
    });
    await agent("writer", call);
  });
  await agent("idle", async () => null);
  const m = await collect();
  const sums = (name: string) => Object.fromEntries(m[name]!.map((p) => [p.attributes[C.AGENT_NAME], hist(p).sum]));
  assert.deepEqual(sums(C.M_INVOKE_AGENT_INFERENCE_CALLS), { planner: 3, writer: 1, idle: 0 });
  assert.deepEqual(sums(C.M_INVOKE_AGENT_TOOL_CALLS), { planner: 1, writer: 0, idle: 0 });
  assert.equal(span("invoke_agent router").attributes[C.TOTAL_MODEL_CALLS], 4);
  const router = one(m[C.M_INVOKE_AGENT_DURATION], { [C.AGENT_NAME]: "router" });
  assert.equal(hist(router).sum, spanSeconds(span("invoke_agent router")));
  assert.deepEqual(hist(router).buckets.boundaries, C.AGENT_SECONDS_BUCKETS);
});

test("metrics are off without a meter provider, and a broken meter never reaches the app", async () => {
  configure();
  await modelCall("gpt-x", async (m) => m.record({ inputTokens: 1, outputTokens: 1 })); // no throw, nothing set up
  const exporter = new InMemorySpanExporter();
  const tracerProvider = new TracerProvider({ spanProcessors: [new SimpleSpanProcessor({ exporter })] });
  const meterProvider = { getMeter: () => ({
    createHistogram: () => ({ record: () => { throw new Error("broken"); } }),
    createCounter: () => ({ add: () => { throw new Error("broken"); } }),
  }) } as unknown as ApiMeterProvider;
  configure({ tracerProvider, meterProvider });
  const answer = await agent("support", async () => {
    await modelCall("gpt-x", async (m) => m.record({ inputTokens: 5, outputTokens: 1 }));
    return toolCall("lookup", async (t) => t.result(41) + 1);
  });
  assert.equal(answer, 42);
  assert.equal(exporter.getFinishedSpans().length, 3); // every span still ended
});
