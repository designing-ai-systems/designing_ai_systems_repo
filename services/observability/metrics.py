"""
Platform metric definitions and naming conventions.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 7.4: Platform metric definitions and naming conventions

Naming convention: ``ai.platform.{service}.{metric_name}``. Counters end
in ``_total``, histograms end in ``_ms`` or ``_score``.
"""


class PlatformMetrics:
    """Standard metrics emitted by platform services."""

    # Model Service
    MODEL_REQUEST_TOTAL = "ai.platform.models.requests_total"
    MODEL_REQUEST_DURATION = "ai.platform.models.request_duration_ms"
    MODEL_TOKENS_PROMPT = "ai.platform.models.tokens.prompt"
    MODEL_TOKENS_COMPLETION = "ai.platform.models.tokens.completion"
    MODEL_COST_USD = "ai.platform.models.cost_usd"
    MODEL_CACHE_HITS = "ai.platform.models.cache_hits_total"
    MODEL_FALLBACKS = "ai.platform.models.fallbacks_total"

    # Data Service
    DATA_SEARCH_DURATION = "ai.platform.data.search_duration_ms"
    DATA_RELEVANCE_SCORE = "ai.platform.data.relevance_score"

    # Session Service
    SESSION_CONTEXT_TOKENS = "ai.platform.sessions.context_tokens"
    SESSION_TRUNCATIONS = "ai.platform.sessions.truncations_total"

    # Guardrails Service
    GUARDRAIL_VIOLATIONS = "ai.platform.guardrails.violations_total"
    GUARDRAIL_LATENCY = "ai.platform.guardrails.evaluation_duration_ms"

    # Tool Service
    TOOL_EXECUTIONS = "ai.platform.tools.executions_total"
    TOOL_CIRCUIT_BREAKS = "ai.platform.tools.circuit_breaks_total"
