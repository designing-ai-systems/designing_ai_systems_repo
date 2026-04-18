"""
OpenAI embedding provider adapter.

Uses the OpenAI Embeddings API (client.embeddings.create) to generate
text embeddings. Separate from OpenAIProvider (chat) because they are
different API surfaces and different capabilities.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Chapter 5: Embedding generation for Data Service
  - Chapter 3 extension: EmbeddingProvider for OpenAI
"""

from __future__ import annotations

from typing import List, Optional

from openai import OpenAI

from services.models.embedding_providers.base import EmbeddingProvider
from services.models.models import (
    EmbeddingResponse,
    ModelCapability,
    ModelInfo,
    TokenUsage,
)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Adapter for OpenAI Embeddings API."""

    def __init__(self, api_key: str, base_url: Optional[str] = None):
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)

    def embed(self, texts: List[str], model: str) -> EmbeddingResponse:
        response = self._client.embeddings.create(
            model=model,
            input=texts,
        )
        embeddings = [item.embedding for item in response.data]
        usage = TokenUsage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=0,
            total_tokens=response.usage.total_tokens,
        )
        return EmbeddingResponse(
            embeddings=embeddings,
            model=response.model,
            provider="openai",
            usage=usage,
        )

    def get_supported_embedding_models(self) -> List[ModelInfo]:
        return [
            ModelInfo(
                name="text-embedding-3-small",
                provider="openai",
                capabilities=ModelCapability(context_window=8191),
            ),
            ModelInfo(
                name="text-embedding-3-large",
                provider="openai",
                capabilities=ModelCapability(context_window=8191),
            ),
            ModelInfo(
                name="text-embedding-ada-002",
                provider="openai",
                capabilities=ModelCapability(context_window=8191),
            ),
        ]
