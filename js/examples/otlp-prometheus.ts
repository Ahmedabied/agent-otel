// The TypeScript side's metrics sent over OTLP to a real Prometheus, then read back with PromQL. The same pretend
// support agent and the same checks as examples/otlp_prometheus.py on the Python side.
//
//   ./prometheus --web.enable-otlp-receiver    # Prometheus 3.x from its GitHub release
//   cd js && npm ci
//   node examples/otlp-prometheus.ts           # exports, then checks Prometheus holds what the spans say
//
// The check compares two separate paths: the numbers on the spans (kept in memory here) and the numbers
// Prometheus stored, so a dashboard built on these metrics shows the same tokens, calls and cost as the traces.
import { randomUUID } from "node:crypto";
import { readFileSync } from "node:fs";
import { setTimeout as sleep } from "node:timers/promises";
import { OTLPMetricExporter } from "@opentelemetry/exporter-metrics-otlp-http";
import { resourceFromAttributes } from "@opentelemetry/resources";
import { MeterProvider, PeriodicExportingMetricReader } from "@opentelemetry/sdk-metrics";
import { InMemorySpanExporter, SimpleSpanProcessor, TracerProvider } from "@opentelemetry/sdk-trace";
import type { ReadableSpan } from "@opentelemetry/sdk-trace";
import { C, configure } from "../src/index.ts";
import { handle, setTrackCalls } from "./demo-agent.ts";

const PROMETHEUS = process.env.PROMETHEUS_URL ?? "http://localhost:9090";
const SERVICE = "demo-support-ts";

async function exportRuns(): Promise<[string, ReadableSpan[]]> {
  const runId = randomUUID().replaceAll("-", "").slice(0, 12);
  const resource = resourceFromAttributes({ "service.name": SERVICE, "service.instance.id": runId });
  const spans = new InMemorySpanExporter();
  const tracerProvider = new TracerProvider({ resource, spanProcessors: [new SimpleSpanProcessor({ exporter: spans })] });
  const exporter = new OTLPMetricExporter({ url: `${PROMETHEUS}/api/v1/otlp/v1/metrics` });
  // One export, at shutdown: the long interval keeps a slow machine from sending a partial first batch.
  const meterProvider = new MeterProvider({ resource,
    readers: [new PeriodicExportingMetricReader({ exporter, exportIntervalMillis: 3_600_000 })] });
  const prices = JSON.parse(readFileSync(new URL("../../examples/prices.example.json", import.meta.url), "utf8"));
  configure({ tracerProvider, prices, meterProvider });
  setTrackCalls(0);
  await handle("where is my order? it was meant to come yesterday", "conv-1");
  setTrackCalls(1); // the second run has no carrier timeout
  await handle("has my parcel shipped yet", "conv-2");
  await meterProvider.shutdown(); // sends the metrics
  const finished = spans.getFinishedSpans(); // before shutdown, which empties the in-memory exporter
  await tracerProvider.shutdown();
  configure();
  return [runId, finished];
}

async function get(path: string, params: Record<string, string>): Promise<unknown> {
  const r = await fetch(`${PROMETHEUS}${path}?${new URLSearchParams(params)}`, { signal: AbortSignal.timeout(5_000) });
  const body = await r.json() as { status: string; data: unknown };
  if (body.status !== "success") throw new Error(`${path}: ${JSON.stringify(body)}`);
  return body.data;
}

type Sample = { metric: Record<string, string>; value: [number, string] };
const query = async (promql: string) => (await get("/api/v1/query", { query: promql }) as { result: Sample[] }).result;

/** Metric names Prometheus holds for this run, as it translated them. */
async function storedNames(runId: string, timeoutMs = 20_000): Promise<string[]> {
  const deadline = performance.now() + timeoutMs;
  for (;;) {
    const names = await get("/api/v1/label/__name__/values", { "match[]": `{instance="${runId}"}` }) as string[];
    if (names.some((n) => n.startsWith("gen_ai_")) || performance.now() > deadline) return names;
    await sleep(500);
  }
}

async function check(runId: string, spans: ReadableSpan[], names: string[]): Promise<[string[], [string, number, number][]]> {
  const sel = `instance="${runId}"`;
  const problems: string[] = [];
  const rows: [string, number, number][] = [];
  const find = (metric: string, histogram: boolean): string | undefined => {
    const stem = metric.replaceAll(".", "_");
    const hits = names.filter((n) => n.startsWith(stem)
      && (histogram ? n.endsWith("_count") : !/_(bucket|sum|count)$/.test(n)));
    if (hits.length !== 1) problems.push(`${metric}: expected one stored name, found ${JSON.stringify(hits)}`);
    return hits.length === 1 ? hits[0] : undefined;
  };
  const total = async (promql: string) => (await query(promql)).reduce((n, s) => n + Number(s.value[1]), 0);
  const compare = async (label: string, metric: string, histogram: boolean, fromSpans: number, extra = "") => {
    const name = find(metric, histogram);
    if (!name) return;
    const stored = await total(`sum(${name}{${sel}${extra}})`);
    rows.push([label, fromSpans, stored]);
    if (Math.abs(stored - fromSpans) > 1e-9 * Math.max(1, Math.abs(fromSpans))) {
      problems.push(`${label}: spans say ${fromSpans}, Prometheus has ${stored}`);
    }
  };
  const chats = spans.filter((s) => s.name.startsWith("chat "));
  const tools = spans.filter((s) => s.name.startsWith("execute_tool "));
  const agents = spans.filter((s) => s.name.startsWith("invoke_agent "));
  const sum = (list: ReadableSpan[], key: string) => list.reduce((n, s) => n + Number(s.attributes[key] ?? 0), 0);
  await compare("model calls", C.M_INFERENCE_DURATION, true, chats.length);
  await compare("input tokens", C.M_USAGE_INPUT_TOKENS, false, sum(chats, C.USAGE_INPUT_TOKENS));
  await compare("output tokens", C.M_USAGE_OUTPUT_TOKENS, false, sum(chats, C.USAGE_OUTPUT_TOKENS));
  await compare("tool calls", C.M_EXECUTE_TOOL_DURATION, true, tools.length);
  await compare("failed tool calls", C.M_EXECUTE_TOOL_DURATION, true,
    tools.filter((s) => C.ERROR_TYPE in s.attributes).length, ',error_type!=""');
  await compare("agent runs", C.M_INVOKE_AGENT_DURATION, true, agents.length);
  await compare("cost", C.COST, false, sum(agents, C.TOTAL_COST));
  const calls = find(C.M_INVOKE_AGENT_INFERENCE_CALLS, true);
  if (calls) {
    const made = await total(`sum(${calls.replace(/_count$/, "_sum")}{${sel}})`);
    rows.push(["model calls, per run sums", chats.length, made]);
    if (made !== chats.length) problems.push(`agent runs made ${made} model calls by the metric, ${chats.length} by the spans`);
  }
  const duration = find(C.M_INFERENCE_DURATION, true);
  if (duration) {
    const bucket = duration.replace(/_count$/, "_bucket");
    const les = new Set((await query(`${bucket}{${sel}}`)).map((s) => s.metric.le));
    const missing = [...C.SECONDS_BUCKETS.map(String), "+Inf"].filter((b) => !les.has(b));
    if (missing.length) problems.push(`${bucket}: bucket bounds miss ${missing.join(", ")}`);
  }
  return [problems, rows];
}

let runId: string;
let spans: ReadableSpan[];
try {
  [runId, spans] = await exportRuns();
} catch (e) {
  console.log(`  problem: the export to ${PROMETHEUS} failed: ${(e as Error).message}`);
  process.exit(1);
}
const names = await storedNames(runId);
const [problems, rows] = await check(runId, spans, names);
const ours = names.filter((n) => n.startsWith("gen_ai_") || n.startsWith("agent_otel_")).sort();
console.log(`Prometheus at ${PROMETHEUS} stored ${ours.length} agent-otel series names for this run (${SERVICE}, instance ${runId})`);
console.log(`${"check".padEnd(28)} ${"from the spans".padStart(15)} ${"in Prometheus".padStart(15)}`);
for (const [label, want, got] of rows) {
  console.log(`${label.padEnd(28)} ${String(Number(want.toPrecision(6))).padStart(15)} ${String(Number(got.toPrecision(6))).padStart(15)}`);
}
for (const p of problems) console.log("  problem:", p);
process.exitCode = problems.length ? 1 : 0;
