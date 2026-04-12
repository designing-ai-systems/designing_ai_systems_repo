"""
Anthropic (Claude) provider adapter.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 3.9: Anthropic provider adapter implementation
"""

from __future__ import annotations

import json
from typing import Iterator, List, Optional, Tuple

from anthropic import Anthropic

from services.models.models import (
    ChatChunk,
    ChatConfig,
    ChatMessage,
    ChatResponse,
    ModelCapability,
    ModelInfo,
    ResponseFormat,
    TokenUsage,
    ToolDefinition,
)
from services.models.providers.base import ModelProvider


class AnthropicProvider(ModelProvider):
    """Adapter for Anthropic Messages API."""

    def __init__(self, api_key: str):
        self._client = Anthropic(api_key=api_key)

    def chat(
        self,
        model: str,
        messages: List[ChatMessage],
        config: ChatConfig,
        tools: Optional[List[ToolDefinition]] = None,
        response_format: Optional[ResponseFormat] = None,
        system_prompt: Optional[str] = None,
    ) -> ChatResponse:
        system_text, conversation = self._split_system_messages(messages, system_prompt)
        payload = {
            "model": model,
            "messages": [self._convert_message(m) for m in conversation],
            "max_tokens": config.max_tokens if config.max_tokens > 0 else 4096,
        }
        if config.stop:
            payload["stop_sequences"] = config.stop
        if config.temperature > 0:
            payload["temperature"] = config.temperature
        elif config.top_p > 0 and config.top_p < 1:
            payload["top_p"] = config.top_p
        if system_text:
            payload["system"] = system_text
        if tools:
            payload["tools"] = [self._convert_tool(t) for t in tools]

        response = self._client.messages.create(**payload)
        text = self._extract_text(response.content)
        usage = response.usage

        return ChatResponse(
            content=text,
            model=response.model,
            provider="anthropic",
            usage=TokenUsage(
                prompt_tokens=usage.input_tokens,
                completion_tokens=usage.output_tokens,
                total_tokens=usage.input_tokens + usage.output_tokens,
            ),
            finish_reason=response.stop_reason or "stop",
        )

    def chat_stream(
        self,
        model: str,
        messages: List[ChatMessage],
        config: ChatConfig,
        tools: Optional[List[ToolDefinition]] = None,
        response_format: Optional[ResponseFormat] = None,
        system_prompt: Optional[str] = None,
    ) -> Iterator[ChatChunk]:
        system_text, conversation = self._split_system_messages(messages, system_prompt)
        payload = {
            "model": model,
            "messages": [self._convert_message(m) for m in conversation],
            "max_tokens": config.max_tokens if config.max_tokens > 0 else 4096,
        }
        if config.stop:
            payload["stop_sequences"] = config.stop
        if config.temperature > 0:
            payload["temperature"] = config.temperature
        elif config.top_p > 0 and config.top_p < 1:
            payload["top_p"] = config.top_p
        if system_text:
            payload["system"] = system_text
        if tools:
            payload["tools"] = [self._convert_tool(t) for t in tools]

        with self._client.messages.stream(**payload) as stream:
            for text in stream.text_stream:
                yield ChatChunk(token=text, model=model)
            final = stream.get_final_message()

        usage = final.usage
        yield ChatChunk(
            token="",
            model=model,
            finish_reason=final.stop_reason or "stop",
            usage=TokenUsage(
                prompt_tokens=usage.input_tokens,
                completion_tokens=usage.output_tokens,
                total_tokens=usage.input_tokens + usage.output_tokens,
            ),
        )

    def get_supported_models(self) -> List[ModelInfo]:
        return [
            ModelInfo(
                name="claude-sonnet-4-5",
                provider="anthropic",
                capabilities=ModelCapability(
                    context_window=200000, supports_vision=True, supports_tools=True
                ),
            ),
            ModelInfo(
                name="claude-haiku-4-5",
                provider="anthropic",
                capabilities=ModelCapability(
                    context_window=200000, supports_vision=True, supports_tools=True
                ),
            ),
            ModelInfo(
                name="claude-opus-4-5",
                provider="anthropic",
                capabilities=ModelCapability(
                    context_window=200000, supports_vision=True, supports_tools=True
                ),
            ),
        ]

    def _split_system_messages(
        self,
        messages: List[ChatMessage],
        system_prompt: Optional[str],
    ) -> Tuple[Optional[str], List[ChatMessage]]:
        system_parts = []
        conversation = []
        for msg in messages:
            if msg.role == "system":
                system_parts.append(msg.content or "")
            else:
                conversation.append(msg)
        if system_prompt:
            system_parts.insert(0, system_prompt)
        combined = "\n".join(p for p in system_parts if p)
        return combined if combined else None, conversation

    def _convert_message(self, message: ChatMessage) -> dict:
        return {"role": message.role, "content": message.content}

    def _convert_tool(self, tool: ToolDefinition) -> dict:
        return {
            "name": tool.function.name,
            "description": tool.function.description,
            "input_schema": tool.function.parameters or {},
        }

    def _extract_text(self, blocks) -> str:
        return "".join(b.text for b in blocks if b.type == "text")
