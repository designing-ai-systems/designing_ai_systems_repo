"""
Model Service metrics publisher.

Translates per-request RequestMetrics into the platform's standard
metric format. The publisher is a thin adapter: it doesn't know
anything about providers or models; it just maps fields to labels
and counters/histograms.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 7.4: PlatformMetrics constants
  - Listing 7.8: ModelServiceMetricsPublisher
"""

from __future__ import annotations

from dataclasses import dataclass

from services.observability.metrics import PlatformMetrics
from services.shared.observability_client import ObservabilityClient


@dataclass
class ModelRequestMetrics:
    """Per-request metric data the publisher consumes."""

    provider: str
    model: str
    workflow_id: str = ""
    cache_hit: bool = False
    fallback_used: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    duration_ms: float = 0.0


class ModelServiceMetricsPublisher:
    """Publishes per-request Model Service metrics (Listing 7.8)."""

    def __init__(self, observability: ObservabilityClient) -> None:
        self.observability = observability

    def publish(self, request_metrics: ModelRequestMetrics) -> None:
        labels = {
            "provider": request_metrics.provider or "unknown",
            "model": request_metrics.model or "unknown",
            "workflow_id": request_metrics.workflow_id or "unknown",
            "cache_hit": str(request_metrics.cache_hit),
        }
        self.observability.record_counter(PlatformMetrics.MODEL_REQUEST_TOTAL, 1.0, labels)
        self.observability.record_histogram(
            PlatformMetrics.MODEL_REQUEST_DURATION,
            request_metrics.duration_ms,
            labels,
        )
        self.observability.record_counter(
            PlatformMetrics.MODEL_COST_USD,
            request_metrics.cost_usd,
            labels,
        )
        if request_metrics.prompt_tokens:
            self.observability.record_counter(
                PlatformMetrics.MODEL_TOKENS_PROMPT,
                request_metrics.prompt_tokens,
                labels,
            )
        if request_metrics.completion_tokens:
            self.observability.record_counter(
                PlatformMetrics.MODEL_TOKENS_COMPLETION,
                request_metrics.completion_tokens,
                labels,
            )
        if request_metrics.cache_hit:
            self.observability.record_counter(PlatformMetrics.MODEL_CACHE_HITS, 1.0, labels)
        if request_metrics.fallback_used:
            self.observability.record_counter(PlatformMetrics.MODEL_FALLBACKS, 1.0, labels)
