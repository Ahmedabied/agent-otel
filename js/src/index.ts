// agent-otel for TypeScript: the same spans, attributes and totals as the Python package, so one viewer reads
// runs from both. Spans follow the OpenTelemetry GenAI conventions; cost comes only from a price table you supply.
import { AsyncLocalStorage } from "node:async_hooks";
import { appendFileSync } from "node:fs";
import { inspect } from "node:util";
import { context, SpanKind, SpanStatusCode, trace } from "@opentelemetry/api";
import type { Attributes, Context, Span, TracerProvider } from "@opentelemetry/api";
import * as C from "./conventions.ts";
import { Prices } from "./prices.ts";
import type { PriceTable } from "./prices.ts";

export { C, Prices };
export { PriceError } from "./prices.ts";
export type { Price, PriceTable } from "./prices.ts";

export const VERSION = "0.1.0";

interface Config {
  provider: TracerProvider | null;
  prices: Prices | null;
  captureContent: boolean;
  maxContentChars: number;
}

const config: Config = { provider: null, prices: null, captureContent: false, maxContentChars: 4000 };

/**
 * Choose where spans go, the price table, and whether tool arguments and results are recorded (off by default:
 * they may hold personal data). Without a tracerProvider, the global one is used, so these spans join whatever
 * tracing the application already has.
 */
export function configure(opts: {
  tracerProvider?: TracerProvider | null;
  prices?: Prices | PriceTable | null;
  captureContent?: boolean;
  maxContentChars?: number;
} = {}): void {
  config.provider = opts.tracerProvider ?? null;
  config.prices = opts.prices == null ? null : opts.prices instanceof Prices ? opts.prices : new Prices(opts.prices);
  config.captureContent = opts.captureContent ?? false;
  config.maxContentChars = opts.maxContentChars ?? 4000;
}

const tracer = () => (config.provider ?? trace.getTracerProvider()).getTracer("agent-otel", VERSION);

class Totals {
  inputTokens = 0;
  outputTokens = 0;
  cost = 0;
  priced = false;
  unpricedCalls = 0;
  modelCalls = 0;
  toolCalls = 0;
}

// Our own context: the runs a call counts toward, and our innermost span. The span is the parent of the next one
// even when the application registered no OpenTelemetry context manager; when it did, its active span wins.
interface Scope {
  runs: Totals[];
  span?: Span;
}
const scope = new AsyncLocalStorage<Scope>();

function parentContext(): Context {
  const active = context.active();
  if (trace.getSpan(active)) return active;
  const ours = scope.getStore()?.span;
  return ours ? trace.setSpan(active, ours) : active;
}

async function inSpan<T>(name: string, kind: SpanKind, attributes: Attributes, runs: Totals[],
                         fn: (span: Span) => T | Promise<T>, after?: (span: Span) => void): Promise<T> {
  const parent = parentContext();
  const span = tracer().startSpan(name, { kind, attributes }, parent);
  try {
    return await context.with(trace.setSpan(parent, span), () => scope.run({ runs, span }, () => fn(span)));
  } catch (e) {
    fail(span, e);
    throw e;
  } finally {
    after?.(span);
    span.end();
  }
}

function fail(span: Span, e: unknown): void {
  const err = e instanceof Error ? e : new Error(String(e));
  span.recordException(err);
  span.setStatus({ code: SpanStatusCode.ERROR, message: err.message.slice(0, 200) });
  span.setAttribute(C.ERROR_TYPE, (e as { constructor?: { name?: string } })?.constructor?.name ?? typeof e);
}

const runs = () => scope.getStore()?.runs ?? [];

// ---- content: tool arguments and results as capped text, never throwing into the app ----------------------------

const CUT = Symbol("cut");

/** JSON text of value, stopped just past limit characters, so a huge result costs little to record. */
function jsonHead(value: unknown, limit: number): [string, boolean] {
  const parts: string[] = [];
  let size = 0;
  const onPath = new Set<object>();
  const push = (s: string) => {
    parts.push(s);
    size += s.length;
    if (size > limit) throw CUT;
  };
  const walk = (v: unknown, depth: number, inArray: boolean): void => {
    if (depth > 1000) throw new RangeError("nested too deep");
    if (v !== null && typeof v === "object" && typeof (v as { toJSON?: unknown }).toJSON === "function") {
      v = (v as { toJSON: () => unknown }).toJSON();
    }
    if (v === null) return push("null");
    switch (typeof v) {
      case "string":
        return push(JSON.stringify(v));
      case "number":
        return push(Number.isFinite(v) ? String(v) : "null");
      case "boolean":
        return push(String(v));
      case "bigint":
        return push(JSON.stringify(v.toString()));
      case "undefined":
      case "function":
      case "symbol":
        return push(inArray || depth === 0 ? "null" : "");
    }
    const o = v as object;
    if (onPath.has(o)) throw new TypeError("circular structure");
    onPath.add(o);
    if (Array.isArray(o)) {
      push("[");
      o.forEach((item, i) => {
        if (i) push(",");
        walk(item, depth + 1, true);
      });
      push("]");
    } else {
      push("{");
      let first = true;
      for (const [k, item] of Object.entries(o)) {
        if (item === undefined || typeof item === "function" || typeof item === "symbol") continue;
        push((first ? "" : ",") + JSON.stringify(k) + ":");
        first = false;
        walk(item, depth + 1, false);
      }
      push("}");
    }
    onPath.delete(o);
  };
  try {
    walk(value, 0, false);
    return [parts.join(""), false];
  } catch (e) {
    if (e === CUT) return [parts.join("").slice(0, limit), true];
    throw e;
  }
}

/** Exported for tests. */
export function content(value: unknown): string {
  const limit = config.maxContentChars;
  let text: string;
  try {
    if (typeof value === "string") text = value;
    else {
      const [head, cut] = jsonHead(value, limit);
      if (cut) return `${head}... [more than ${limit} chars]`;
      text = head;
    }
  } catch {
    // circular references, nesting too deep, a toJSON that throws ...
    try {
      text = inspect(value, { depth: 4, breakLength: Infinity, maxArrayLength: 50, maxStringLength: limit });
    } catch {
      text = "<a value that cannot be shown>";
    }
  }
  return text.length <= limit ? text : `${text.slice(0, limit)}... [${text.length} chars]`;
}

// ---- model calls ---------------------------------------------------------------------------------------------------

export interface Usage {
  /** ALL input tokens, cached ones included (the GenAI meaning). Some APIs report input without the cached parts:
   *  pass their sum here. Cache counts bigger than inputTokens are added to it, with a warning. */
  inputTokens?: number;
  outputTokens?: number;
  cacheReadInputTokens?: number;
  cacheWriteInputTokens?: number;
  reasoningOutputTokens?: number;
  responseModel?: string;
  responseId?: string;
  finishReasons?: string[];
}

export class ModelCall {
  cost: number | null = null;
  readonly span: Span;
  readonly model: string;
  private readonly counted: Totals[];

  constructor(span: Span, model: string, counted: Totals[]) {
    this.span = span;
    this.model = model;
    this.counted = counted;
  }

  /** Call once the model has answered, with the usage numbers from its response. */
  record(u: Usage): void {
    const s = this.span;
    const read = Math.trunc(u.cacheReadInputTokens ?? 0);
    const write = Math.trunc(u.cacheWriteInputTokens ?? 0);
    let input = u.inputTokens === undefined ? undefined : Math.trunc(u.inputTokens);
    if (input !== undefined && read + write > input) {
      process.emitWarning(`agent-otel: cache tokens (${read} read + ${write} written) exceed inputTokens (${input}), `
        + "so inputTokens was taken to exclude them and they were added; pass inputTokens with the cached tokens included");
      input += read + write;
    }
    if (input !== undefined) s.setAttribute(C.USAGE_INPUT_TOKENS, input);
    if (u.outputTokens !== undefined) s.setAttribute(C.USAGE_OUTPUT_TOKENS, Math.trunc(u.outputTokens));
    if (u.cacheReadInputTokens !== undefined) s.setAttribute(C.USAGE_CACHE_READ_INPUT_TOKENS, read);
    if (u.cacheWriteInputTokens !== undefined) s.setAttribute(C.USAGE_CACHE_WRITE_INPUT_TOKENS, write);
    if (u.reasoningOutputTokens !== undefined) {
      s.setAttribute(C.USAGE_REASONING_OUTPUT_TOKENS, Math.trunc(u.reasoningOutputTokens));
    }
    if (u.responseModel) s.setAttribute(C.RESPONSE_MODEL, u.responseModel);
    if (u.finishReasons?.length) s.setAttribute(C.RESPONSE_FINISH_REASONS, u.finishReasons.map(String));
    if (u.responseId) s.setAttribute(C.RESPONSE_ID, u.responseId);
    const prices = config.prices;
    const output = Math.trunc(u.outputTokens ?? 0);
    if (prices) {
      this.cost = prices.cost(u.responseModel || this.model, input ?? 0, output, read, write);
      if (this.cost !== null) {
        s.setAttribute(C.COST, this.cost);
        s.setAttribute(C.COST_CURRENCY, prices.currency);
      }
    }
    for (const run of this.counted) {
      run.inputTokens += input ?? 0;
      run.outputTokens += output;
      if (this.cost !== null) {
        run.cost += this.cost;
        run.priced = true;
      } else if (prices) run.unpricedCalls += 1;
    }
  }
}

/** One model call. Record its usage with call.record({...}) when the answer is in. */
export function modelCall<T>(model: string, fn: (call: ModelCall) => T | Promise<T>,
                             opts: { provider?: string; operation?: string; maxTokens?: number; temperature?: number } = {}): Promise<T> {
  const operation = opts.operation ?? C.CHAT;
  const attrs: Attributes = {
    [C.OPERATION_NAME]: operation, [C.PROVIDER_NAME]: opts.provider ?? "unknown", [C.REQUEST_MODEL]: model,
  };
  if (opts.maxTokens !== undefined) attrs[C.REQUEST_MAX_TOKENS] = opts.maxTokens;
  if (opts.temperature !== undefined) attrs[C.REQUEST_TEMPERATURE] = opts.temperature;
  const counted = runs();
  for (const run of counted) run.modelCalls += 1;
  return inSpan(`${operation} ${model}`, SpanKind.CLIENT, attrs, counted, (span) => fn(new ModelCall(span, model, counted)));
}

// ---- tool calls ----------------------------------------------------------------------------------------------------

export class ToolCall {
  readonly span: Span;

  constructor(span: Span) {
    this.span = span;
  }

  /** Record the tool's result (kept only when captureContent is on). Returns it unchanged. */
  result<R>(value: R): R {
    if (config.captureContent) this.span.setAttribute(C.TOOL_CALL_RESULT, content(value));
    return value;
  }
}

/** One tool call. Pass the arguments to have them recorded when captureContent is on. */
export function toolCall<T>(name: string, fn: (call: ToolCall) => T | Promise<T>,
                            opts: { callId?: string; arguments?: unknown; toolType?: string } = {}): Promise<T> {
  const attrs: Attributes = {
    [C.OPERATION_NAME]: C.EXECUTE_TOOL, [C.TOOL_NAME]: name, [C.TOOL_TYPE]: opts.toolType ?? "function",
  };
  if (opts.callId) attrs[C.TOOL_CALL_ID] = opts.callId;
  if (opts.arguments !== undefined && config.captureContent) attrs[C.TOOL_CALL_ARGUMENTS] = content(opts.arguments);
  const counted = runs();
  for (const run of counted) run.toolCalls += 1;
  return inSpan(`${C.EXECUTE_TOOL} ${name}`, SpanKind.INTERNAL, attrs, counted, (span) => fn(new ToolCall(span)));
}

/** Wrap a function so every call is a tool span. The wrapper is always async. */
export function tool<A extends unknown[], R>(name: string, fn: (...args: A) => R | Promise<R>,
                                             opts: { toolType?: string } = {}): (...args: A) => Promise<Awaited<R>> {
  return (...args: A) => toolCall(name, async (t): Promise<Awaited<R>> => t.result(await fn(...args)),
    { arguments: config.captureContent ? (args.length === 1 ? args[0] : args) : undefined, toolType: opts.toolType });
}

// ---- agent runs ----------------------------------------------------------------------------------------------------

/** One agent run. Its span carries totals for every model and tool call inside it, however deeply nested. */
export function agent<T>(name: string, fn: (span: Span) => T | Promise<T>,
                         opts: { conversationId?: string; agentId?: string } = {}): Promise<T> {
  const attrs: Attributes = { [C.OPERATION_NAME]: C.INVOKE_AGENT, [C.AGENT_NAME]: name };
  if (opts.conversationId) attrs[C.CONVERSATION_ID] = opts.conversationId;
  if (opts.agentId) attrs[C.AGENT_ID] = opts.agentId;
  const totals = new Totals();
  const prices = config.prices;
  return inSpan(`${C.INVOKE_AGENT} ${name}`, SpanKind.INTERNAL, attrs, [...runs(), totals], fn, (span) => {
    span.setAttribute(C.TOTAL_INPUT_TOKENS, totals.inputTokens);
    span.setAttribute(C.TOTAL_OUTPUT_TOKENS, totals.outputTokens);
    span.setAttribute(C.TOTAL_MODEL_CALLS, totals.modelCalls);
    span.setAttribute(C.TOTAL_TOOL_CALLS, totals.toolCalls);
    if (prices) {
      // The total covers priced calls only; the count says how many it left out.
      span.setAttribute(C.TOTAL_UNPRICED_CALLS, totals.unpricedCalls);
      if (totals.priced) {
        span.setAttribute(C.TOTAL_COST, Math.round(totals.cost * 1e10) / 1e10);
        span.setAttribute(C.COST_CURRENCY, prices.currency);
      }
    }
  });
}

// ---- JSON lines export, the Python side's format ------------------------------------------------------------------

const KINDS = ["INTERNAL", "SERVER", "CLIENT", "PRODUCER", "CONSUMER"];
const STATUS = ["UNSET", "OK", "ERROR"];

/** The parts of an SDK ReadableSpan this exporter reads (sdk-trace-base 1.x and 2.x). */
export interface FinishedSpan {
  name: string;
  kind: number;
  spanContext(): { traceId: string; spanId: string };
  parentSpanContext?: { spanId: string };
  parentSpanId?: string;
  startTime: [number, number];
  endTime: [number, number];
  status: { code: number; message?: string };
  attributes: Attributes;
  events: Array<{ name: string; time: [number, number]; attributes?: Attributes }>;
  resource?: { attributes?: Attributes };
}

const ns = (t: [number, number]) => BigInt(t[0]) * 1_000_000_000n + BigInt(t[1]);
const BIG = `__agent_otel_${Math.random().toString(36).slice(2)}__`;

/** One span as one JSON line, field for field what the Python JsonlExporter writes (times as exact integers). */
export function spanToJson(s: FinishedSpan): string {
  const ctx = s.spanContext();
  const doc = {
    trace_id: ctx.traceId,
    span_id: ctx.spanId,
    parent_id: s.parentSpanContext?.spanId ?? s.parentSpanId ?? null,
    name: s.name,
    kind: KINDS[s.kind] ?? "INTERNAL",
    start_ns: ns(s.startTime),
    end_ns: ns(s.endTime),
    status: STATUS[s.status.code] ?? "UNSET",
    error: s.status.message ?? null,
    attributes: s.attributes,
    events: s.events.map((e) => ({ name: e.name, time_ns: ns(e.time), attributes: e.attributes ?? {} })),
    service: (s.resource?.attributes?.["service.name"] as string | undefined) ?? null,
  };
  return JSON.stringify(doc, (_k, v) => (typeof v === "bigint" ? `${BIG}${v}` : v))
    .replace(new RegExp(`"${BIG}(\\d+)"`, "g"), "$1");
}

/** A span exporter that appends finished spans to a JSON lines file, readable by `agent-otel summary` and `view`. */
export class JsonlExporter {
  private readonly path: string;

  constructor(path: string) {
    this.path = path;
  }

  export(spans: FinishedSpan[], done: (result: { code: number; error?: Error }) => void): void {
    try {
      appendFileSync(this.path, spans.map((s) => spanToJson(s) + "\n").join(""), "utf8");
      done({ code: 0 });
    } catch (e) {
      done({ code: 1, error: e instanceof Error ? e : new Error(String(e)) });
    }
  }

  shutdown(): Promise<void> {
    return Promise.resolve();
  }

  forceFlush(): Promise<void> {
    return Promise.resolve();
  }
}
