"""
OpenAI provider adapter.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 3.9 pattern: Provider adapter implementation (book shows Anthropic;
    OpenAI follows the same adapter pattern)
"""

from __future__ import annotations

import json
from typing import Iterator, List, Optional

from openai import OpenAI

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


class OpenAIProvider(ModelProvider):
    """Adapter for OpenAI Chat Completions API."""

    def __init__(self, api_key: str, base_url: Optional[str] = None):
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)

    def chat(
        self,
        model: str,
        messages: List[ChatMessage],
        config: ChatConfig,
        tools: Optional[List[ToolDefinition]] = None,
        response_format: Optional[ResponseFormat] = None,
        system_prompt: Optional[str] = None,
    ) -> ChatResponse:
        openai_messages = [self._convert_message(m) for m in messages]
        if system_prompt:
            openai_messages.insert(0, {"role": "system", "content": system_prompt})

        payload = {
            "model": model,
            "messages": openai_messages,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }
        if config.stop:
            payload["stop"] = config.stop
        if config.max_tokens > 0:
            payload["max_tokens"] = config.max_tokens
        if tools:
            payload["tools"] = [self._convert_tool(t) for t in tools]
        if response_format:
            payload["response_format"] = self._convert_response_format(response_format)

        response = self._client.chat.completions.create(**payload)
        message = response.choices[0].message
        usage = response.usage

        tool_calls_out = None
        if message.tool_calls:
            tool_calls_out = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]

        return ChatResponse(
            content=message.content or "",
            model=response.model,
            provider="openai",
            usage=TokenUsage(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
            ),
            tool_calls=tool_calls_out,
            finish_reason=response.choices[0].finish_reason or "stop",
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
        openai_messages = [self._convert_message(m) for m in messages]
        if system_prompt:
            openai_messages.insert(0, {"role": "system", "content": system_prompt})

        payload = {
            "model": model,
            "messages": openai_messages,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if config.stop:
            payload["stop"] = config.stop
        if config.max_tokens > 0:
            payload["max_tokens"] = config.max_tokens
        if tools:
            payload["tools"] = [self._convert_tool(t) for t in tools]
        if response_format:
            payload["response_format"] = self._convert_response_format(response_format)

        final_finish = None
        final_usage = None
        stream = self._client.chat.completions.create(**payload)
        for event in stream:
            if not event.choices:
                if event.usage:
                    final_usage = event.usage
                continue
            choice = event.choices[0]
            delta = choice.delta
            if delta and delta.content:
                yield ChatChunk(token=delta.content, model=model)
            if choice.finish_reason:
                final_finish = choice.finish_reason
            if event.usage:
                final_usage = event.usage

        usage = None
        if final_usage:
            usage = TokenUsage(
                prompt_tokens=final_usage.prompt_tokens,
                completion_tokens=final_usage.completion_tokens,
                total_tokens=final_usage.total_tokens,
            )

        yield ChatChunk(
            token="",
            model=model,
            finish_reason=final_finish or "stop",
            usage=usage,
        )

    def get_supported_models(self) -> List[ModelInfo]:
        return [
            ModelInfo(
                name="gpt-4o",
                provider="openai",
                capabilities=ModelCapability(
                    context_window=128000,
                    supports_vision=True,
                    supports_tools=True,
                ),
            ),
            ModelInfo(
                name="gpt-4o-mini",
                provider="openai",
                capabilities=ModelCapability(
                    context_window=128000,
                    supports_vision=True,
                    supports_tools=True,
                ),
            ),
        ]

    def _convert_message(self, message: ChatMessage) -> dict:
        data = {"role": message.role, "content": message.content}
        if message.tool_calls:
            data["tool_calls"] = message.tool_calls
        if message.tool_call_id:
            data["tool_call_id"] = message.tool_call_id
        return data

    def _convert_tool(self, tool: ToolDefinition) -> dict:
        return {
            "type": "function",
            "function": {
                "name": tool.function.name,
                "description": tool.function.description,
                "parameters": tool.function.parameters or {},
            },
        }

    def _convert_response_format(self, response_format: ResponseFormat) -> dict:
        return {"type": response_format.type}
