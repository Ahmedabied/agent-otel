// The TypeScript side's spans sent over OTLP to a real tracing backend, then read back from it. The same pretend
// support agent and the same checks as examples/otlp_jaeger.py on the Python side.
//
//   ./jaeger                               # the binary from a Jaeger 2.x GitHub release, no arguments needed
//   cd js && npm ci
//   node examples/otlp-jaeger.ts           # exports, then checks Jaeger stored what was sent
//
// agent-otel is plain OpenTelemetry, so this is the stock OTLP exporter and nothing else. Open
// http://localhost:16686 and pick the "demo-support-ts" service to see the two runs.
import { randomUUID } from "node:crypto";
import { readFileSync } from "node:fs";
import { setTimeout as sleep } from "node:timers/promises";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-http";
import { resourceFromAttributes } from "@opentelemetry/resources";
import { BatchSpanProcessor, TracerProvider } from "@opentelemetry/sdk-trace";
import { agent, configure, modelCall, tool } from "../src/index.ts";

const OTLP = process.env.OTEL_EXPORTER_OTLP_TRACES_ENDPOINT ?? "http://localhost:4318/v1/traces";
const JAEGER = process.env.JAEGER_QUERY_URL ?? "http://localhost:16686";
const SERVICE = "demo-support-ts";

// ---- the pretend agent: a scripted "model" that sleeps a little per token ----------------------------------------

class TimeoutError extends Error {
  override name = "TimeoutError";
}

const words = (s: string) => s.split(/\s+/).filter(Boolean).length;

async function ask(model: string, msPerToken: number, prompt: string, reply: string): Promise<string> {
  return modelCall(model, async (m) => {
    const inputTokens = Math.floor(words(prompt) * 1.3) + 400; // plus a system prompt
    const outputTokens = Math.floor(words(reply) * 1.3);
    await sleep(20 + outputTokens * msPerToken);
    m.record({ inputTokens, outputTokens, finishReasons: ["stop"] });
    return reply;
  }, { provider: "example", maxTokens: 512 });
}

let trackCalls = 0;
const findOrder = tool("find_order", async (_email: string) => {
  await sleep(35);
  return { order: "A-1042", status: "in transit", carrier: "parcel co" };
});
const trackParcel = tool("track_parcel", async (_order: string) => {
  await sleep(90);
  trackCalls += 1;
  if (trackCalls === 1) throw new TimeoutError("carrier API did not answer in 80 ms");
  return { eta: "Thursday" };
});

function handle(question: string, conversation: string): Promise<string> {
  return agent("support-agent", async () => {
    await ask("local-qwen-7b", 4, question, "intent: order status. need email lookup then tracking");
    const order = await findOrder("someone@example.org");
    let eta: { eta: string } | undefined;
    for (let attempt = 0; attempt < 2 && !eta; attempt++) {
      try {
        eta = await trackParcel(order.order);
      } catch (e) {
        if (!(e instanceof TimeoutError) || attempt === 1) throw e;
      }
    }
    return ask("example-hosted-small", 1.5, `${question} ${order.order} ${eta?.eta}`,
      "Your order A-1042 is on its way with Parcel Co and should arrive on Thursday. " +
      "I will send you the tracking link by email as well.");
  }, { conversationId: conversation });
}

// ---- export, read back, check -------------------------------------------------------------------------------------

async function exportRuns(): Promise<string> {
  const runId = randomUUID().replaceAll("-", "").slice(0, 12);
  const provider = new TracerProvider({
    resource: resourceFromAttributes({ "service.name": SERVICE, "service.instance.id": runId }),
    spanProcessors: [new BatchSpanProcessor({ exporter: new OTLPTraceExporter({ url: OTLP }) })],
  });
  const prices = JSON.parse(readFileSync(new URL("../../examples/prices.example.json", import.meta.url), "utf8"));
  configure({ tracerProvider: provider, prices });
  trackCalls = 0;
  await handle("where is my order? it was meant to come yesterday", "conv-1");
  trackCalls = 1; // the second run has no carrier timeout
  await handle("has my parcel shipped yet", "conv-2");
  await provider.shutdown(); // flushes the batch
  return runId;
}

type Stored = { name: string; attrs: Record<string, unknown> };
type AnyValue = { intValue?: string; stringValue?: string; doubleValue?: number; boolValue?: boolean };
type KeyValue = { key: string; value: AnyValue };

/** An OTLP JSON attribute value. 64-bit ints travel as strings: they come back as BigInt, so "is it an integer"
 *  is a type check, the same as on the Python side. */
function value(v: AnyValue): unknown {
  if (v.intValue !== undefined) return BigInt(v.intValue);
  return v.stringValue ?? v.doubleValue ?? v.boolValue ?? v;
}

/** This run's spans as Jaeger stored them, by trace id, read back through its API v3 (OTLP JSON). */
async function storedTraces(runId: string, timeoutMs = 15_000): Promise<Map<string, Stored[]>> {
  const now = Date.now();
  const query = new URLSearchParams({
    "query.service_name": SERVICE,
    "query.start_time_min": new Date(now - 3_600_000).toISOString(),
    "query.start_time_max": new Date(now + 300_000).toISOString(),
  });
  const deadline = performance.now() + timeoutMs;
  for (;;) {
    const traces = new Map<string, Stored[]>();
    const r = await fetch(`${JAEGER}/api/v3/traces?${query}`, { signal: AbortSignal.timeout(5_000) });
    if (r.ok) {
      const data = await r.json() as { result?: { resourceSpans?: Array<{
        resource?: { attributes?: KeyValue[] };
        scopeSpans?: Array<{ spans?: Array<{ traceId: string; name: string; attributes?: KeyValue[] }> }>;
      }> } };
      for (const rs of data.result?.resourceSpans ?? []) {
        const resource = Object.fromEntries((rs.resource?.attributes ?? []).map((a) => [a.key, value(a.value)]));
        if (resource["service.instance.id"] !== runId) continue; // an earlier run of this example
        for (const ss of rs.scopeSpans ?? []) {
          for (const s of ss.spans ?? []) {
            const attrs = Object.fromEntries((s.attributes ?? []).map((a) => [a.key, value(a.value)]));
            if (!traces.has(s.traceId)) traces.set(s.traceId, []);
            traces.get(s.traceId)!.push({ name: s.name, attrs });
          }
        }
      }
    } else if (r.status !== 404) { // 404: nothing indexed for the service yet
      throw new Error(`Jaeger answered ${r.status}: ${await r.text()}`);
    }
    if (traces.size >= 2 || performance.now() > deadline) return traces;
    await sleep(500);
  }
}

/** What a backend must have kept for agent-otel's spans to be useful there. */
function check(traces: Map<string, Stored[]>): string[] {
  const problems: string[] = [];
  const all = [...traces.values()].flat();
  const runs = all.filter((s) => s.name === "invoke_agent support-agent");
  if (runs.length !== 2) problems.push(`expected 2 agent runs, found ${runs.length}`);
  for (const [tid, spans] of traces) {
    const names = new Set(spans.map((s) => s.name));
    for (const want of ["invoke_agent support-agent", "chat local-qwen-7b", "chat example-hosted-small",
      "execute_tool find_order", "execute_tool track_parcel"]) {
      if (!names.has(want)) problems.push(`trace ${tid.slice(0, 8)}: no '${want}' span`);
    }
  }
  for (const s of all) {
    if (s.name.startsWith("chat ") && typeof s.attrs["gen_ai.usage.input_tokens"] !== "bigint") {
      problems.push(`${s.name}: gen_ai.usage.input_tokens missing or not an integer`);
    }
    if (s.name.startsWith("invoke_agent") && !("agent_otel.total.cost" in s.attrs)) {
      problems.push("agent span without agent_otel.total.cost");
    }
  }
  const failed = all.filter((s) => s.name === "execute_tool track_parcel" && s.attrs["error.type"] === "TimeoutError");
  if (failed.length !== 1) problems.push(`expected 1 failed track_parcel span, found ${failed.length}`);
  return problems;
}

let runId: string;
try {
  runId = await exportRuns();
} catch (e) {
  console.log(`  problem: the export to ${OTLP} failed: ${(e as Error).message}`);
  process.exit(1);
}
const traces = await storedTraces(runId);
const problems = check(traces);
const spans = [...traces.values()].reduce((n, t) => n + t.length, 0);
console.log(`Jaeger at ${JAEGER} stored ${traces.size} traces, ${spans} spans from this run (${SERVICE}, instance ${runId})`);
for (const p of problems) console.log("  problem:", p);
process.exitCode = problems.length ? 1 : 0;
