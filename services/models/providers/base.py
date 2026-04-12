"""
Provider adapter base class.

Defines a unified interface for all model providers using domain types.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 3.8: Provider adapter interface (ModelProvider ABC)
"""

from abc import ABC, abstractmethod
from typing import Iterator, List, Optional

from services.models.models import (
    ChatChunk,
    ChatConfig,
    ChatMessage,
    ChatResponse,
    ModelInfo,
    ResponseFormat,
    ToolDefinition,
)


class ModelProvider(ABC):
    """Abstract base class that all provider adapters implement."""

    @abstractmethod
    def chat(
        self,
        model: str,
        messages: List[ChatMessage],
        config: ChatConfig,
        tools: Optional[List[ToolDefinition]] = None,
        response_format: Optional[ResponseFormat] = None,
        system_prompt: Optional[str] = None,
    ) -> ChatResponse:
        """Generate a response synchronously."""
        raise NotImplementedError

    @abstractmethod
    def chat_stream(
        self,
        model: str,
        messages: List[ChatMessage],
        config: ChatConfig,
        tools: Optional[List[ToolDefinition]] = None,
        response_format: Optional[ResponseFormat] = None,
        system_prompt: Optional[str] = None,
    ) -> Iterator[ChatChunk]:
        """Generate a response with streaming tokens."""
        raise NotImplementedError

    @abstractmethod
    def get_supported_models(self) -> List[ModelInfo]:
        """Report which models this provider supports."""
        raise NotImplementedError
