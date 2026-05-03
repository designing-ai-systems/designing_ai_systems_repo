"""Integration test: Model Service Chat emits a traced Generation (Listing 7.7)."""

from typing import List

from proto import models_pb2
from services.models.metrics_publisher import ModelServiceMetricsPublisher
from services.models.models import (
    ChatResponse,
    ModelCapability,
    ModelInfo,
    TokenUsage,
)
from services.models.providers.base import ModelProvider
from services.models.service import ModelService
from services.shared.observability_client import ObservabilityClient


class FakeProvider(ModelProvider):
    name = "fake"

    def get_supported_models(self) -> List[ModelInfo]:
        return [
            ModelInfo(
                name="fake-model",
                provider="fake",
                capabilities=ModelCapability(context_window=4096),
            )
        ]

    def chat(self, *, model, messages, config, tools, response_format, system_prompt):
        return ChatResponse(
            content="hello",
            model=model,
            provider="fake",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            finish_reason="stop",
        )

    def chat_stream(self, *args, **kwargs):
        return iter([])

    def embed(self, *args, **kwargs):
        raise NotImplementedError


class _RecordingClient(ObservabilityClient):
    def __init__(self):
        super().__init__(stub=None, autostart=False)
        self.recorded_generations = []
        self.counter_calls = []
        self.histogram_calls = []

    def end_generation(self, span_id, *, status=None, error_message="", **attrs):
        # Snapshot the in-flight generation record before the parent removes it.
        gen = self._pending_generations.get(span_id)
        if gen is not None:
            self.recorded_generations.append(
                {
                    "span_id": span_id,
                    "model": gen.model,
                    "provider": gen.provider,
                    "prompt_tokens": gen.prompt_tokens,
                    "completion_tokens": gen.completion_tokens,
                    "cost_usd": gen.cost_usd,
                }
            )
        self._pending_generations.pop(span_id, None)

    def record_counter(self, name, value, labels=None):
        self.counter_calls.append((name, value, labels or {}))

    def record_histogram(self, name, value, labels=None):
        self.histogram_calls.append((name, value, labels or {}))


class FakeContext:
    def __init__(self, metadata=None):
        self._metadata = list((metadata or {}).items())
        self.code = None
        self.details_str = None

    def invocation_metadata(self):
        return self._metadata

    def set_code(self, code):
        self.code = code

    def set_details(self, details):
        self.details_str = details


class TestModelServiceTracing:
    def test_chat_emits_generation_with_token_usage(self):
        client = _RecordingClient()
        svc = ModelService(observability=client)
        svc._providers = {"fake": FakeProvider()}
        svc._metrics_publisher = ModelServiceMetricsPublisher(client)

        request = models_pb2.ChatRequest(model="fake-model")
        msg = request.messages.add()
        msg.role = "user"
        msg.content = "hi"

        ctx = FakeContext(metadata={"x-trace-id": "trace-7", "x-workflow-id": "wf-7"})
        response = svc.Chat(request, ctx)

        assert response.content == "hello"
        assert len(client.recorded_generations) == 1
        gen = client.recorded_generations[0]
        assert gen["prompt_tokens"] == 10
        assert gen["completion_tokens"] == 5
        # ModelServiceMetricsPublisher emitted the standard counters/histograms.
        counter_names = {c[0] for c in client.counter_calls}
        assert "ai.platform.models.requests_total" in counter_names
        assert "ai.platform.models.cost_usd" in counter_names

    def test_default_construction_uses_noop_client(self):
        # Without an observability client the existing tests must still pass.
        svc = ModelService()
        assert svc.observability._noop is True  # noqa: SLF001
