"""Unit tests for ModelServiceMetricsPublisher (Listing 7.8)."""

from typing import List, Tuple

from services.models.metrics_publisher import (
    ModelRequestMetrics,
    ModelServiceMetricsPublisher,
)
from services.shared.observability_client import ObservabilityClient


class _RecordingClient(ObservabilityClient):
    def __init__(self):
        super().__init__(stub=None, autostart=False)
        self.counters: List[Tuple[str, float, dict]] = []
        self.histograms: List[Tuple[str, float, dict]] = []

    def record_counter(self, name, value, labels=None):
        self.counters.append((name, value, dict(labels or {})))

    def record_histogram(self, name, value, labels=None):
        self.histograms.append((name, value, dict(labels or {})))


class TestPublish:
    def test_emits_three_listing_7_8_calls(self):
        rec = _RecordingClient()
        ModelServiceMetricsPublisher(rec).publish(
            ModelRequestMetrics(
                provider="openai",
                model="gpt-4o",
                workflow_id="wf-1",
                cache_hit=False,
                prompt_tokens=100,
                completion_tokens=20,
                cost_usd=0.001,
                duration_ms=350.0,
            )
        )
        names = [c[0] for c in rec.counters] + [h[0] for h in rec.histograms]
        assert "ai.platform.models.requests_total" in names
        assert "ai.platform.models.request_duration_ms" in names
        assert "ai.platform.models.cost_usd" in names

    def test_labels_include_provider_model_workflow(self):
        rec = _RecordingClient()
        ModelServiceMetricsPublisher(rec).publish(
            ModelRequestMetrics(provider="anthropic", model="claude-sonnet-4-5", workflow_id="wf-7")
        )
        for _, _, labels in rec.counters + rec.histograms:
            assert labels["provider"] == "anthropic"
            assert labels["model"] == "claude-sonnet-4-5"
            assert labels["workflow_id"] == "wf-7"

    def test_cache_hit_emits_extra_counter(self):
        rec = _RecordingClient()
        ModelServiceMetricsPublisher(rec).publish(
            ModelRequestMetrics(provider="openai", model="gpt-4o", cache_hit=True)
        )
        assert any(c[0] == "ai.platform.models.cache_hits_total" for c in rec.counters)

    def test_fallback_emits_extra_counter(self):
        rec = _RecordingClient()
        ModelServiceMetricsPublisher(rec).publish(
            ModelRequestMetrics(provider="openai", model="gpt-4o", fallback_used=True)
        )
        assert any(c[0] == "ai.platform.models.fallbacks_total" for c in rec.counters)
