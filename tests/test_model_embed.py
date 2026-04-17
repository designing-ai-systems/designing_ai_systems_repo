"""Tests for embedding functionality: EmbeddingProvider ABC, providers, servicer, SDK client."""

from unittest.mock import MagicMock, patch

import pytest

from services.models.embedding_providers.base import EmbeddingProvider
from services.models.models import (
    EmbeddingResponse,
    ModelInfo,
    TokenUsage,
)

# ---------------------------------------------------------------------------
# Stub provider for ABC contract testing
# ---------------------------------------------------------------------------


class StubEmbeddingProvider(EmbeddingProvider):
    """Minimal provider for testing the ABC contract."""

    def embed(self, texts, model):
        vecs = [[float(i)] * 3 for i in range(len(texts))]
        return EmbeddingResponse(
            embeddings=vecs,
            model=model,
            provider="stub",
            usage=TokenUsage(
                prompt_tokens=len(texts),
                completion_tokens=0,
                total_tokens=len(texts),
            ),
        )

    def get_supported_embedding_models(self):
        return [ModelInfo(name="stub-embed", provider="stub")]


# ===========================================================================
# 1. EmbeddingResponse dataclass
# ===========================================================================


class TestEmbeddingResponse:
    def test_basic_response(self):
        resp = EmbeddingResponse(
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            model="text-embedding-3-small",
            provider="openai",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=0, total_tokens=10),
        )
        assert len(resp.embeddings) == 2
        assert resp.model == "text-embedding-3-small"
        assert resp.provider == "openai"
        assert resp.usage.total_tokens == 10

    def test_without_usage(self):
        resp = EmbeddingResponse(
            embeddings=[[1.0, 2.0]],
            model="all-MiniLM-L6-v2",
            provider="huggingface",
        )
        assert resp.usage is None


# ===========================================================================
# 2. EmbeddingProvider ABC contract
# ===========================================================================


class TestEmbeddingProviderABC:
    def test_embed_returns_domain_type(self):
        provider = StubEmbeddingProvider()
        resp = provider.embed(["hello", "world"], "stub-embed")
        assert isinstance(resp, EmbeddingResponse)
        assert len(resp.embeddings) == 2
        assert resp.provider == "stub"

    def test_get_supported_embedding_models(self):
        provider = StubEmbeddingProvider()
        models = provider.get_supported_embedding_models()
        assert len(models) == 1
        assert isinstance(models[0], ModelInfo)
        assert models[0].name == "stub-embed"

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            EmbeddingProvider()


# ===========================================================================
# 3. OpenAI embedding provider
# ===========================================================================


class TestOpenAIEmbeddingProvider:
    def test_embed_calls_openai_api(self):
        from services.models.embedding_providers.openai_provider import (
            OpenAIEmbeddingProvider,
        )

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3]),
            MagicMock(embedding=[0.4, 0.5, 0.6]),
        ]
        mock_response.model = "text-embedding-3-small"
        mock_response.usage.prompt_tokens = 8
        mock_response.usage.total_tokens = 8
        mock_client.embeddings.create.return_value = mock_response

        provider = OpenAIEmbeddingProvider(api_key="test-key")
        provider._client = mock_client

        resp = provider.embed(["hello", "world"], "text-embedding-3-small")
        assert isinstance(resp, EmbeddingResponse)
        assert len(resp.embeddings) == 2
        assert resp.embeddings[0] == [0.1, 0.2, 0.3]
        assert resp.model == "text-embedding-3-small"
        assert resp.provider == "openai"

        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input=["hello", "world"],
        )

    def test_get_supported_embedding_models(self):
        from services.models.embedding_providers.openai_provider import (
            OpenAIEmbeddingProvider,
        )

        provider = OpenAIEmbeddingProvider(api_key="test-key")
        models = provider.get_supported_embedding_models()
        names = [m.name for m in models]
        assert "text-embedding-3-small" in names
        assert "text-embedding-3-large" in names
        assert "text-embedding-ada-002" in names


# ===========================================================================
# 4. HuggingFace embedding provider
# ===========================================================================


class TestHuggingFaceEmbeddingProvider:
    def test_embed_uses_sentence_transformers(self):
        import numpy as np

        from services.models.embedding_providers.huggingface_provider import (
            HuggingFaceEmbeddingProvider,
        )

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])

        provider = HuggingFaceEmbeddingProvider(model_names=["all-MiniLM-L6-v2"])
        provider._models = {"all-MiniLM-L6-v2": mock_model}

        resp = provider.embed(["hello", "world"], "all-MiniLM-L6-v2")
        assert isinstance(resp, EmbeddingResponse)
        assert len(resp.embeddings) == 2
        assert resp.embeddings[0] == pytest.approx([0.1, 0.2])
        assert resp.provider == "huggingface"

    def test_get_supported_embedding_models(self):
        from services.models.embedding_providers.huggingface_provider import (
            HuggingFaceEmbeddingProvider,
        )

        provider = HuggingFaceEmbeddingProvider(
            model_names=["all-MiniLM-L6-v2", "all-mpnet-base-v2"]
        )
        models = provider.get_supported_embedding_models()
        names = [m.name for m in models]
        assert "all-MiniLM-L6-v2" in names
        assert "all-mpnet-base-v2" in names

    def test_unsupported_model_raises(self):
        from services.models.embedding_providers.huggingface_provider import (
            HuggingFaceEmbeddingProvider,
        )

        provider = HuggingFaceEmbeddingProvider(model_names=["all-MiniLM-L6-v2"])
        with pytest.raises(ValueError, match="not configured"):
            provider.embed(["hello"], "nonexistent-model")


# ===========================================================================
# 5. Model Service servicer — Embed RPC
# ===========================================================================


class TestServicerEmbed:
    def _make_service_with_embedding(self):
        """Create a ModelService with a mock embedding provider injected."""
        from services.models.service import ModelService

        svc = ModelService()
        mock_provider = StubEmbeddingProvider()
        svc._embedding_providers = {"stub": mock_provider}
        return svc

    def _make_context(self):
        ctx = MagicMock()
        ctx.invocation_metadata.return_value = [("x-target-service", "models")]
        return ctx

    def test_embed_returns_vectors(self):
        from proto import models_pb2

        svc = self._make_service_with_embedding()
        ctx = self._make_context()

        request = models_pb2.EmbedRequest(
            texts=["hello", "world"],
            model="stub-embed",
        )
        resp = svc.Embed(request, ctx)
        assert len(resp.embeddings) == 2
        assert resp.model == "stub-embed"
        assert resp.provider == "stub"

    def test_embed_unknown_model(self):
        from proto import models_pb2

        svc = self._make_service_with_embedding()
        ctx = self._make_context()

        request = models_pb2.EmbedRequest(
            texts=["hello"],
            model="nonexistent-model",
        )
        svc.Embed(request, ctx)
        ctx.set_code.assert_called()

    def test_list_embedding_models(self):
        from proto import models_pb2

        svc = self._make_service_with_embedding()
        ctx = self._make_context()

        resp = svc.ListEmbeddingModels(models_pb2.ListEmbeddingModelsRequest(), ctx)
        assert len(resp.models) >= 1
        names = [m.name for m in resp.models]
        assert "stub-embed" in names


# ===========================================================================
# 6. SDK client — embed() and list_embedding_models()
# ===========================================================================


class TestModelClientEmbed:
    def _make_client(self):
        """Create a ModelClient with a mock stub."""
        from genai_platform.clients.models import ModelClient

        mock_platform = MagicMock()
        mock_platform.gateway_url = "localhost:50051"

        with patch("genai_platform.clients.models.models_pb2_grpc.ModelServiceStub"):
            client = ModelClient(mock_platform)
        client._stub = MagicMock()
        return client

    def test_embed_returns_domain_type(self):
        from proto import models_pb2

        client = self._make_client()
        mock_response = models_pb2.EmbedResponse(
            embeddings=[
                models_pb2.Embedding(values=[0.1, 0.2, 0.3]),
                models_pb2.Embedding(values=[0.4, 0.5, 0.6]),
            ],
            model="text-embedding-3-small",
            provider="openai",
            usage=models_pb2.TokenUsage(prompt_tokens=8, completion_tokens=0, total_tokens=8),
        )
        client._stub.Embed.return_value = mock_response

        resp = client.embed(["hello", "world"], model="text-embedding-3-small")
        assert isinstance(resp, EmbeddingResponse)
        assert len(resp.embeddings) == 2
        assert resp.embeddings[0] == pytest.approx([0.1, 0.2, 0.3])
        assert resp.model == "text-embedding-3-small"

    def test_list_embedding_models(self):
        from proto import models_pb2

        client = self._make_client()
        mock_response = models_pb2.ListEmbeddingModelsResponse(
            models=[
                models_pb2.ModelInfo(
                    name="text-embedding-3-small",
                    provider="openai",
                ),
            ]
        )
        client._stub.ListEmbeddingModels.return_value = mock_response

        models = client.list_embedding_models()
        assert len(models) == 1
        assert isinstance(models[0], ModelInfo)
        assert models[0].name == "text-embedding-3-small"
