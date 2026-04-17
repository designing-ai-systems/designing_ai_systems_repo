"""
HuggingFace embedding provider adapter.

Uses sentence-transformers for local embedding generation. The dependency
is optional — install with: pip install genai-platform[huggingface]

Models are downloaded on first use and cached locally by sentence-transformers.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Chapter 5: Embedding generation for Data Service
  - Chapter 3 extension: EmbeddingProvider for HuggingFace
"""

from __future__ import annotations

from typing import Dict, List

from services.models.embedding_providers.base import EmbeddingProvider
from services.models.models import (
    EmbeddingResponse,
    ModelCapability,
    ModelInfo,
)


def _load_sentence_transformer(model_name: str):
    """Lazy-load sentence-transformers, giving a clear error if not installed."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for HuggingFace embeddings. "
            "Install with: pip install sentence-transformers"
        )
    return SentenceTransformer(model_name)


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """Adapter for HuggingFace sentence-transformers models."""

    def __init__(self, model_names: List[str]):
        self._model_names = model_names
        self._models: Dict[str, object] = {}

    def _get_model(self, model_name: str):
        if model_name not in self._model_names:
            raise ValueError(
                f"Model '{model_name}' not configured. "
                f"Available: {self._model_names}"
            )
        if model_name not in self._models:
            self._models[model_name] = _load_sentence_transformer(model_name)
        return self._models[model_name]

    def embed(self, texts: List[str], model: str) -> EmbeddingResponse:
        st_model = self._get_model(model)
        raw = st_model.encode(texts)
        embeddings = [vec.tolist() for vec in raw]
        return EmbeddingResponse(
            embeddings=embeddings,
            model=model,
            provider="huggingface",
        )

    def get_supported_embedding_models(self) -> List[ModelInfo]:
        return [
            ModelInfo(
                name=name,
                provider="huggingface",
                capabilities=ModelCapability(context_window=512),
            )
            for name in self._model_names
        ]
