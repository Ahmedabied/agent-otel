"""OpenTelemetry GenAI attribute names used by agent-otel.

Read from opentelemetry-semantic-conventions 0.66b2, where the GenAI names are marked as moved to
the separate open-telemetry/semantic-conventions-genai repository (the names themselves did not
change). They are defined here, not imported, so a further move cannot break this library.
"""

OPERATION_NAME = "gen_ai.operation.name"
PROVIDER_NAME = "gen_ai.provider.name"  # replaces the deprecated gen_ai.system
REQUEST_MODEL = "gen_ai.request.model"
REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
REQUEST_TEMPERATURE = "gen_ai.request.temperature"
RESPONSE_MODEL = "gen_ai.response.model"
RESPONSE_ID = "gen_ai.response.id"
RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"  # replaces gen_ai.usage.prompt_tokens
USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"  # replaces gen_ai.usage.completion_tokens
USAGE_CACHE_READ_INPUT_TOKENS = "gen_ai.usage.cache_read.input_tokens"  # part of input_tokens
USAGE_CACHE_WRITE_INPUT_TOKENS = "gen_ai.usage.cache_write.input_tokens"  # part of input_tokens; was cache_creation
USAGE_REASONING_OUTPUT_TOKENS = "gen_ai.usage.reasoning.output_tokens"
TOOL_NAME = "gen_ai.tool.name"
TOOL_CALL_ID = "gen_ai.tool.call.id"
TOOL_TYPE = "gen_ai.tool.type"
TOOL_CALL_ARGUMENTS = "gen_ai.tool.call.arguments"  # content: only when capture_content=True
TOOL_CALL_RESULT = "gen_ai.tool.call.result"  # content: only when capture_content=True
AGENT_NAME = "gen_ai.agent.name"
AGENT_ID = "gen_ai.agent.id"
CONVERSATION_ID = "gen_ai.conversation.id"
ERROR_TYPE = "error.type"

# gen_ai.operation.name values
CHAT = "chat"
TEXT_COMPLETION = "text_completion"
EMBEDDINGS = "embeddings"
EXECUTE_TOOL = "execute_tool"
INVOKE_AGENT = "invoke_agent"

# Not part of the conventions: agent-otel's own attributes, in its own namespace.
COST = "agent_otel.cost"
COST_CURRENCY = "agent_otel.cost.currency"
TOTAL_INPUT_TOKENS = "agent_otel.total.input_tokens"
TOTAL_OUTPUT_TOKENS = "agent_otel.total.output_tokens"
TOTAL_COST = "agent_otel.total.cost"
TOTAL_MODEL_CALLS = "agent_otel.total.model_calls"
TOTAL_TOOL_CALLS = "agent_otel.total.tool_calls"
TOTAL_UNPRICED_CALLS = "agent_otel.total.unpriced_calls"  # model calls left out of total.cost
