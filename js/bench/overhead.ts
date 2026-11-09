// Cost of one traced tool call: a wrapped no-op tool, every span processed and handed to an exporter that keeps
// nothing, against the same no-op called plainly. Run: node bench/overhead.ts
import { SimpleSpanProcessor, TracerProvider } from "@opentelemetry/sdk-trace";
import { agent, configure, tool } from "../src/index.ts";

const exporter = { export: (_s: unknown[], done: (r: { code: number }) => void) => done({ code: 0 }), shutdown: async () => {} };
configure({ tracerProvider: new TracerProvider({ spanProcessors: [new SimpleSpanProcessor({ exporter: exporter as never })] }) });
const plain = async (x: number) => x + 1;
const traced = tool("add_one", plain);
const N = 100_000;

async function time(fn: (x: number) => Promise<number>): Promise<number> {
  for (let i = 0; i < 5_000; i++) await fn(i); // warm up
  const t0 = performance.now();
  for (let i = 0; i < N; i++) await fn(i);
  return ((performance.now() - t0) * 1000) / N;
}

await agent("bench", async () => {
  const base = await time(plain);
  const withSpans = await time(traced);
  console.log(`plain ${base.toFixed(2)} us, traced ${withSpans.toFixed(2)} us, added ${(withSpans - base).toFixed(2)} us per tool call (Node ${process.version})`);
});
