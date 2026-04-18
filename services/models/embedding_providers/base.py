"""
Embedding provider adapter base class.

Defines a unified interface for all embedding providers using domain types.
Separate from ModelProvider (LLM inference) because embedding is a
fundamentally different capability: text in, vectors out.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Chapter 5 references Model Service embed() method
  - Chapter 3 extension: EmbeddingProvider ABC
"""

from abc import ABC, abstractmethod
from typing import List

from services.models.models import EmbeddingResponse, ModelInfo


class EmbeddingProvider(ABC):
    """Abstract base class that all embedding provider adapters implement."""

    @abstractmethod
    def embed(self, texts: List[str], model: str) -> EmbeddingResponse:
        """Generate embeddings for a list of texts."""
        raise NotImplementedError

    @abstractmethod
    def get_supported_embedding_models(self) -> List[ModelInfo]:
        """Report which embedding models this provider supports."""
        raise NotImplementedError
