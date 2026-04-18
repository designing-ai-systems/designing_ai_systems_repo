"""Embedding provider adapters."""

from .base import EmbeddingProvider
from .huggingface_provider import HuggingFaceEmbeddingProvider
from .openai_provider import OpenAIEmbeddingProvider

__all__ = [
    "EmbeddingProvider",
    "HuggingFaceEmbeddingProvider",
    "OpenAIEmbeddingProvider",
]
