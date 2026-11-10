// OpenTelemetry GenAI attribute names, the same as the Python side (src/agent_otel/conventions.py), which reads
// them from the semantic conventions. Defined here, not imported, so a move of the conventions cannot break this.
export const OPERATION_NAME = "gen_ai.operation.name";
export const PROVIDER_NAME = "gen_ai.provider.name";
export const REQUEST_MODEL = "gen_ai.request.model";
export const REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens";
export const REQUEST_TEMPERATURE = "gen_ai.request.temperature";
export const RESPONSE_MODEL = "gen_ai.response.model";
export const RESPONSE_ID = "gen_ai.response.id";
export const RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons";
export const USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens";
export const USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens";
export const USAGE_CACHE_READ_INPUT_TOKENS = "gen_ai.usage.cache_read.input_tokens";
export const USAGE_CACHE_WRITE_INPUT_TOKENS = "gen_ai.usage.cache_write.input_tokens";
export const USAGE_REASONING_OUTPUT_TOKENS = "gen_ai.usage.reasoning.output_tokens";
export const TOOL_NAME = "gen_ai.tool.name";
export const TOOL_CALL_ID = "gen_ai.tool.call.id";
export const TOOL_TYPE = "gen_ai.tool.type";
export const TOOL_CALL_ARGUMENTS = "gen_ai.tool.call.arguments";
export const TOOL_CALL_RESULT = "gen_ai.tool.call.result";
export const AGENT_NAME = "gen_ai.agent.name";
export const AGENT_ID = "gen_ai.agent.id";
export const CONVERSATION_ID = "gen_ai.conversation.id";
export const ERROR_TYPE = "error.type";

export const TOKEN_MODALITY = "gen_ai.token.modality"; // metrics only; never told, so always "unknown"
export const MODALITY_UNKNOWN = "unknown";

export const CHAT = "chat";
export const EMBEDDINGS = "embeddings";
export const EXECUTE_TOOL = "execute_tool";
export const INVOKE_AGENT = "invoke_agent";

// Metric names and buckets: the same as the Python side, from semantic-conventions-genai at commit e6658f1
// (2026-11-06), where they are unreleased. They replace gen_ai.client.token.usage.
export const M_INFERENCE_DURATION = "gen_ai.client.inference.duration";
export const M_INFERENCE_INPUT_TOKENS = "gen_ai.client.inference.operation.input_tokens";
export const M_INFERENCE_OUTPUT_TOKENS = "gen_ai.client.inference.operation.output_tokens";
export const M_USAGE_INPUT_TOKENS = "gen_ai.client.inference.usage.input_tokens";
export const M_USAGE_OUTPUT_TOKENS = "gen_ai.client.inference.usage.output_tokens";
export const M_USAGE_CACHE_READ_INPUT_TOKENS = "gen_ai.client.inference.usage.cache_read.input_tokens";
export const M_USAGE_CACHE_WRITE_INPUT_TOKENS = "gen_ai.client.inference.usage.cache_write.input_tokens";
export const M_USAGE_REASONING_OUTPUT_TOKENS = "gen_ai.client.inference.usage.reasoning.output_tokens";
export const M_EMBEDDINGS_DURATION = "gen_ai.client.embeddings.duration";
export const M_EMBEDDINGS_INPUT_TOKENS = "gen_ai.client.embeddings.operation.input_tokens";
export const M_EXECUTE_TOOL_DURATION = "gen_ai.execute_tool.duration";
export const M_INVOKE_AGENT_DURATION = "gen_ai.invoke_agent.duration";
export const M_INVOKE_AGENT_INFERENCE_CALLS = "gen_ai.invoke_agent.inference_calls";
export const M_INVOKE_AGENT_TOOL_CALLS = "gen_ai.invoke_agent.tool_calls";
export const M_UNPRICED_CALLS = "agent_otel.unpriced_calls";

export const SECONDS_BUCKETS = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92];
export const AGENT_SECONDS_BUCKETS = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2, 102.4, 204.8, 409.6, 900, 1800,
  3600, 7200, 14400, 43200, 86400];
export const TOKEN_BUCKETS = [1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864];
export const CALL_COUNT_BUCKETS = [1, 2, 4, 8, 16, 32, 64, 128];

export const COST = "agent_otel.cost";
export const COST_CURRENCY = "agent_otel.cost.currency";
export const TOTAL_INPUT_TOKENS = "agent_otel.total.input_tokens";
export const TOTAL_OUTPUT_TOKENS = "agent_otel.total.output_tokens";
export const TOTAL_COST = "agent_otel.total.cost";
export const TOTAL_MODEL_CALLS = "agent_otel.total.model_calls";
export const TOTAL_TOOL_CALLS = "agent_otel.total.tool_calls";
export const TOTAL_UNPRICED_CALLS = "agent_otel.total.unpriced_calls";
