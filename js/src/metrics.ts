// OpenTelemetry metrics for the same calls the spans describe, the GenAI way; the same as the Python side
// (src/agent_otel/metrics.py). Off until configure({ meterProvider }). Each value is recorded once, when the call's
// block ends, from the same numbers and clock readings as its span. Only low-cardinality attributes: no conversation,
// response, call or agent ids, and never content.
import { ValueType } from "@opentelemetry/api";
import type { Attributes, Counter, Histogram, MeterProvider } from "@opentelemetry/api";
import * as C from "./conventions.ts";

/** What a model call reported, as metrics need it; undefined when record() was never called. */
export interface CallUsage {
  inputTokens: number;
  outputTokens: number;
  cacheRead: number;
  cacheWrite: number;
  reasoning: number;
}

export class Instruments {
  private readonly inferenceDuration: Histogram;
  private readonly inferenceInput: Histogram;
  private readonly inferenceOutput: Histogram;
  private readonly usage: [keyof CallUsage, Counter][];
  private readonly usageReasoning: Counter;
  private readonly embeddingsDuration: Histogram;
  private readonly embeddingsInput: Histogram;
  private readonly toolDuration: Histogram;
  private readonly agentDuration: Histogram;
  private readonly agentInferenceCalls: Histogram;
  private readonly agentToolCalls: Histogram;
  private readonly cost: Counter;
  private readonly unpriced: Counter;
  private readonly currency: string;

  constructor(provider: MeterProvider, version: string, currency: string | null) {
    this.currency = currency ?? "";
    const m = provider.getMeter("agent-otel", version);
    const hist = (name: string, unit: string, description: string, buckets: number[], int = false) =>
      m.createHistogram(name, { unit, description, advice: { explicitBucketBoundaries: buckets },
        ...(int ? { valueType: ValueType.INT } : {}) });
    const tokens = (name: string, description: string) =>
      m.createCounter(name, { unit: "{token}", description, valueType: ValueType.INT });
    this.inferenceDuration = hist(C.M_INFERENCE_DURATION, "s", "GenAI client inference operation duration.", C.SECONDS_BUCKETS);
    this.inferenceInput = hist(C.M_INFERENCE_INPUT_TOKENS, "{token}",
      "The number of input (prompt) tokens used per inference operation.", C.TOKEN_BUCKETS, true);
    this.inferenceOutput = hist(C.M_INFERENCE_OUTPUT_TOKENS, "{token}",
      "The number of output (completion) tokens used per inference operation.", C.TOKEN_BUCKETS, true);
    this.usage = [
      ["inputTokens", tokens(C.M_USAGE_INPUT_TOKENS, "The number of input (prompt) tokens used, including cached tokens.")],
      ["outputTokens", tokens(C.M_USAGE_OUTPUT_TOKENS, "The number of output (completion) tokens used, including reasoning tokens.")],
      ["cacheRead", tokens(C.M_USAGE_CACHE_READ_INPUT_TOKENS, "The number of input tokens served from a provider-managed cache.")],
      ["cacheWrite", tokens(C.M_USAGE_CACHE_WRITE_INPUT_TOKENS, "The number of input tokens written to a provider-managed cache.")],
    ];
    this.usageReasoning = tokens(C.M_USAGE_REASONING_OUTPUT_TOKENS, "The number of output tokens used for reasoning.");
    this.embeddingsDuration = hist(C.M_EMBEDDINGS_DURATION, "s", "GenAI client embeddings operation duration.", C.SECONDS_BUCKETS);
    this.embeddingsInput = hist(C.M_EMBEDDINGS_INPUT_TOKENS, "{token}",
      "The number of input tokens used per embeddings operation.", C.TOKEN_BUCKETS, true);
    this.toolDuration = hist(C.M_EXECUTE_TOOL_DURATION, "s", "The duration of a single tool execution.", C.SECONDS_BUCKETS);
    this.agentDuration = hist(C.M_INVOKE_AGENT_DURATION, "s", "The end-to-end duration of one agent run.", C.AGENT_SECONDS_BUCKETS);
    this.agentInferenceCalls = hist(C.M_INVOKE_AGENT_INFERENCE_CALLS, "{inference_call}",
      "The number of model calls one agent run made itself.", C.CALL_COUNT_BUCKETS, true);
    this.agentToolCalls = hist(C.M_INVOKE_AGENT_TOOL_CALLS, "{tool_call}",
      "The number of tool calls one agent run made itself.", C.CALL_COUNT_BUCKETS, true);
    // Not in the conventions. Cost needs a price table; the unpriced counter says what the cost leaves out.
    this.cost = m.createCounter(C.COST, { unit: "1", description: "Model call cost from your price table." });
    this.unpriced = m.createCounter(C.M_UNPRICED_CALLS, { unit: "{call}",
      description: "Model calls with usage but no price, left out of agent_otel.cost.", valueType: ValueType.INT });
  }

  model(operation: string, provider: string, model: string, responseModel: string | undefined, error: string | undefined,
        seconds: number, usage: CallUsage | undefined, cost: number | null, pricedTable: boolean): void {
    const attrs: Attributes = { [C.OPERATION_NAME]: operation, [C.PROVIDER_NAME]: provider, [C.REQUEST_MODEL]: model };
    if (responseModel) attrs[C.RESPONSE_MODEL] = responseModel;
    const withError = error ? { ...attrs, [C.ERROR_TYPE]: error } : attrs;
    if (operation === C.EMBEDDINGS) {
      this.embeddingsDuration.record(seconds, withError);
      this.embeddingsInput.record(usage?.inputTokens ?? 0, withError);
    } else {
      this.inferenceDuration.record(seconds, withError);
      this.inferenceInput.record(usage?.inputTokens ?? 0, withError);
      this.inferenceOutput.record(usage?.outputTokens ?? 0, withError);
      if (usage) {
        // Counters take positive amounts only; zeros would not change a total.
        const byModality = { ...attrs, [C.TOKEN_MODALITY]: C.MODALITY_UNKNOWN };
        for (const [key, counter] of this.usage) if (usage[key] > 0) counter.add(usage[key], byModality);
        if (usage.reasoning > 0) this.usageReasoning.add(usage.reasoning, attrs); // no modality on this one
      }
    }
    if (usage && pricedTable) {
      if (cost === null) this.unpriced.add(1, attrs);
      else if (cost > 0) this.cost.add(cost, { ...attrs, [C.COST_CURRENCY]: this.currency });
    }
  }

  tool(name: string, toolType: string, agentName: string | undefined, error: string | undefined, seconds: number): void {
    const attrs: Attributes = { [C.TOOL_NAME]: name, [C.TOOL_TYPE]: toolType };
    if (agentName) attrs[C.AGENT_NAME] = agentName;
    if (error) attrs[C.ERROR_TYPE] = error;
    this.toolDuration.record(seconds, attrs);
  }

  agent(name: string, error: string | undefined, seconds: number, modelCalls: number, toolCalls: number,
        delegated: boolean): void {
    const attrs: Attributes = { [C.OPERATION_NAME]: C.INVOKE_AGENT, [C.AGENT_NAME]: name };
    this.agentDuration.record(seconds, error ? { ...attrs, [C.ERROR_TYPE]: error } : attrs);
    // Calls a run made itself; a nested run counts its own. A run that only handed work to nested runs is an
    // entrypoint, which the conventions say not to report; any other run reports 0 too.
    if (modelCalls || !delegated) this.agentInferenceCalls.record(modelCalls, attrs);
    if (toolCalls || !delegated) this.agentToolCalls.record(toolCalls, attrs);
  }
}
