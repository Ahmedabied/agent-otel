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

export const CHAT = "chat";
export const EXECUTE_TOOL = "execute_tool";
export const INVOKE_AGENT = "invoke_agent";

export const COST = "agent_otel.cost";
export const COST_CURRENCY = "agent_otel.cost.currency";
export const TOTAL_INPUT_TOKENS = "agent_otel.total.input_tokens";
export const TOTAL_OUTPUT_TOKENS = "agent_otel.total.output_tokens";
export const TOTAL_COST = "agent_otel.total.cost";
export const TOTAL_MODEL_CALLS = "agent_otel.total.model_calls";
export const TOTAL_TOOL_CALLS = "agent_otel.total.tool_calls";
export const TOTAL_UNPRICED_CALLS = "agent_otel.total.unpriced_calls";
