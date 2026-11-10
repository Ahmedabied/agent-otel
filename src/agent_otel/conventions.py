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

TOKEN_MODALITY = "gen_ai.token.modality"  # metrics only; agent-otel is never told it, so always "unknown"
MODALITY_UNKNOWN = "unknown"

# Metric names, read from open-telemetry/semantic-conventions-genai at commit e6658f1 (2026-11-06), where
# they are unreleased and in development. They replace gen_ai.client.token.usage and, for these
# operations, gen_ai.client.operation.duration (changelog.d/374, 521 and the execute-tool entry there).
M_INFERENCE_DURATION = "gen_ai.client.inference.duration"
M_INFERENCE_INPUT_TOKENS = "gen_ai.client.inference.operation.input_tokens"  # one value per call
M_INFERENCE_OUTPUT_TOKENS = "gen_ai.client.inference.operation.output_tokens"
M_USAGE_INPUT_TOKENS = "gen_ai.client.inference.usage.input_tokens"  # counters: running totals
M_USAGE_OUTPUT_TOKENS = "gen_ai.client.inference.usage.output_tokens"
M_USAGE_CACHE_READ_INPUT_TOKENS = "gen_ai.client.inference.usage.cache_read.input_tokens"
M_USAGE_CACHE_WRITE_INPUT_TOKENS = "gen_ai.client.inference.usage.cache_write.input_tokens"
M_USAGE_REASONING_OUTPUT_TOKENS = "gen_ai.client.inference.usage.reasoning.output_tokens"
M_EMBEDDINGS_DURATION = "gen_ai.client.embeddings.duration"
M_EMBEDDINGS_INPUT_TOKENS = "gen_ai.client.embeddings.operation.input_tokens"
M_EXECUTE_TOOL_DURATION = "gen_ai.execute_tool.duration"
M_INVOKE_AGENT_DURATION = "gen_ai.invoke_agent.duration"  # an in-process agent, as agent() is
M_INVOKE_AGENT_INFERENCE_CALLS = "gen_ai.invoke_agent.inference_calls"
M_INVOKE_AGENT_TOOL_CALLS = "gen_ai.invoke_agent.tool_calls"

# The bucket boundaries the conventions ask for.
SECONDS_BUCKETS = (0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92)
AGENT_SECONDS_BUCKETS = (0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2, 102.4, 204.8, 409.6, 900, 1800, 3600,
                         7200, 14400, 43200, 86400)
TOKEN_BUCKETS = (1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864)
CALL_COUNT_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128)

# Not part of the conventions: agent-otel's own attributes, in its own namespace.
COST = "agent_otel.cost"
COST_CURRENCY = "agent_otel.cost.currency"
TOTAL_INPUT_TOKENS = "agent_otel.total.input_tokens"
TOTAL_OUTPUT_TOKENS = "agent_otel.total.output_tokens"
TOTAL_COST = "agent_otel.total.cost"
TOTAL_MODEL_CALLS = "agent_otel.total.model_calls"
TOTAL_TOOL_CALLS = "agent_otel.total.tool_calls"
TOTAL_UNPRICED_CALLS = "agent_otel.total.unpriced_calls"  # model calls left out of total.cost
M_UNPRICED_CALLS = "agent_otel.unpriced_calls"  # metric: model calls with usage but no price
