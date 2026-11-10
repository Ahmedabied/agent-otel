// Cost of one traced tool call: a wrapped no-op tool, every span processed and handed to an exporter that keeps
// nothing, against the same no-op called plainly; then a model call, and both again with metrics recorded too.
// Run: node bench/overhead.ts
import { MeterProvider, MetricReader } from "@opentelemetry/sdk-metrics";
import { SimpleSpanProcessor, TracerProvider } from "@opentelemetry/sdk-trace";
import { agent, configure, modelCall, tool } from "../src/index.ts";

const exporter = { export: (_s: unknown[], done: (r: { code: number }) => void) => done({ code: 0 }), shutdown: async () => {} };
const tracerProvider = new TracerProvider({ spanProcessors: [new SimpleSpanProcessor({ exporter: exporter as never })] });
const prices = { currency: "USD", models: { "local-*": { input: 0, output: 0 } } };
configure({ tracerProvider, prices });
const plain = async (x: number) => x + 1;
const traced = tool("add_one", plain);
const model = (x: number) => modelCall("local-1", async (m) => {
  m.record({ inputTokens: 100, outputTokens: 10 });
  return x;
}, { provider: "bench" });
const N = 100_000;

async function time(fn: (x: number) => Promise<number>): Promise<number> {
  for (let i = 0; i < 5_000; i++) await fn(i); // warm up
  const t0 = performance.now();
  for (let i = 0; i < N; i++) await fn(i);
  return ((performance.now() - t0) * 1000) / N;
}

// Metrics kept in memory and never exported: what recording costs.
class Keep extends MetricReader {
  protected override async onForceFlush(): Promise<void> {}
  protected override async onShutdown(): Promise<void> {}
}

await agent("bench", async () => {
  const base = await time(plain);
  const withSpans = await time(traced);
  const modelSpan = await time(model);
  console.log(`plain ${base.toFixed(2)} us, traced ${withSpans.toFixed(2)} us, added ${(withSpans - base).toFixed(2)} us per tool call (Node ${process.version})`);
  console.log(`model call span with usage and cost ${modelSpan.toFixed(2)} us`);
});
configure({ tracerProvider, prices, meterProvider: new MeterProvider({ readers: [new Keep()] }) });
await agent("bench", async () => {
  const withMetrics = await time(traced);
  const modelMetrics = await time(model);
  console.log(`with metrics: tool call ${withMetrics.toFixed(2)} us, model call ${modelMetrics.toFixed(2)} us`);
});
