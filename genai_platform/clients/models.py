"""
Model Service client.

Returns domain dataclasses (ChatResponse, ChatChunk, etc.),
never exposing Protocol Buffers to the caller.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 3.17: ModelClient initialization (class + __init__)
  - Listing 3.18: ModelClient.chat() method
  - Listing 3.19: ModelClient.chat_stream() method
"""

import json
import logging
from typing import Any, Dict, Iterator, List, Optional

import grpc

from proto import models_pb2, models_pb2_grpc
from services.models.models import (
    ChatChunk,
    ChatResponse,
    EmbeddingResponse,
    FallbackConfig,
    ModelCapability,
    ModelInfo,
    TokenUsage,
)

from .base import BaseClient

logger = logging.getLogger(__name__)


class ModelClient(BaseClient):
    """Client for Model Service."""

    def __init__(self, platform):
        super().__init__(platform, "models")
        self._stub = models_pb2_grpc.ModelServiceStub(self._channel)

    # ==================== Core Inference ====================

    def chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[str] = None,
        system_prompt_name: Optional[str] = None,
        fallback_config: Optional[FallbackConfig] = None,
        **kwargs,
    ) -> ChatResponse:
        """Generate a chat response. Returns ChatResponse dataclass.

        When *fallback_config* is provided and enabled, the client will
        iterate through ``fallback_config.providers`` on RPC failure,
        transparently retrying with alternative models.

        Example:
            response = platform.models.chat(
                model="claude-sonnet-4-5",
                messages=[{"role": "user", "content": "Hello!"}],
                fallback_config=FallbackConfig(
                    providers=["gpt-4o"],
                ),
            )
            print(response.content)
        """
        models_to_try = self._build_model_chain(model, fallback_config)
        last_error: Optional[Exception] = None

        for try_model in models_to_try:
            request = self._build_chat_request(
                try_model,
                messages,
                temperature,
                max_tokens,
                tools,
                response_format,
                system_prompt_name,
            )
            try:
                resp = self._stub.Chat(request, metadata=self._metadata)
                return self._proto_to_chat_response(resp)
            except grpc.RpcError as e:
                last_error = e
                if self._should_fail_fast(e, fallback_config):
                    raise
                logger.warning("chat failed for model %s: %s", try_model, e)
                continue

        raise last_error

    def chat_stream(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        fallback_config: Optional[FallbackConfig] = None,
        **kwargs,
    ) -> Iterator[ChatChunk]:
        """Stream a chat response. Yields ChatChunk dataclasses.

        Supports the same *fallback_config* as :meth:`chat`.

        Example:
            for chunk in platform.models.chat_stream(
                model="claude-sonnet-4-5",
                messages=[{"role": "user", "content": "Hello"}],
                fallback_config=FallbackConfig(providers=["gpt-4o"]),
            ):
                print(chunk.token, end="", flush=True)
        """
        models_to_try = self._build_model_chain(model, fallback_config)
        last_error: Optional[Exception] = None

        for try_model in models_to_try:
            pb_messages = [self._dict_to_proto_msg(m) for m in messages]
            config = models_pb2.ChatConfig(temperature=temperature)
            if max_tokens:
                config.max_tokens = max_tokens

            request = models_pb2.ChatRequest(
                model=try_model,
                messages=pb_messages,
                config=config,
            )
            try:
                for chunk in self._stub.ChatStream(request, metadata=self._metadata):
                    usage = None
                    if chunk.HasField("usage"):
                        usage = TokenUsage(
                            prompt_tokens=chunk.usage.prompt_tokens,
                            completion_tokens=chunk.usage.completion_tokens,
                            total_tokens=chunk.usage.total_tokens,
                        )
                    yield ChatChunk(
                        token=chunk.token,
                        model=chunk.model,
                        finish_reason=chunk.finish_reason
                        if chunk.HasField("finish_reason")
                        else None,
                        usage=usage,
                    )
                return  # stream completed successfully
            except grpc.RpcError as e:
                last_error = e
                if self._should_fail_fast(e, fallback_config):
                    raise
                logger.warning("chat_stream failed for model %s: %s", try_model, e)
                continue

        raise last_error

    # ==================== Discovery ====================

    def list_models(self) -> List[ModelInfo]:
        """List available models. Returns list of ModelInfo dataclasses."""
        resp = self._stub.ListModels(models_pb2.ListModelsRequest(), metadata=self._metadata)
        return [
            ModelInfo(
                name=m.name,
                provider=m.provider,
                capabilities=ModelCapability(
                    context_window=m.capabilities.context_window,
                    supports_vision=m.capabilities.supports_vision,
                    supports_tools=m.capabilities.supports_tools,
                ),
            )
            for m in resp.models
        ]

    def get_model_capabilities(self, model: str) -> ModelCapability:
        """Get capabilities for a specific model."""
        request = models_pb2.GetCapabilitiesRequest(model=model)
        caps = self._stub.GetModelCapabilities(request, metadata=self._metadata)
        return ModelCapability(
            context_window=caps.context_window,
            supports_vision=caps.supports_vision,
            supports_tools=caps.supports_tools,
        )

    # ==================== Embedding ====================

    def embed(
        self,
        texts: List[str],
        model: str,
    ) -> EmbeddingResponse:
        """Generate embeddings for a list of texts. Returns EmbeddingResponse dataclass."""
        request = models_pb2.EmbedRequest(texts=texts, model=model)
        resp = self._stub.Embed(request, metadata=self._metadata)
        usage = None
        if resp.HasField("usage"):
            usage = TokenUsage(
                prompt_tokens=resp.usage.prompt_tokens,
                completion_tokens=resp.usage.completion_tokens,
                total_tokens=resp.usage.total_tokens,
            )
        return EmbeddingResponse(
            embeddings=[list(e.values) for e in resp.embeddings],
            model=resp.model,
            provider=resp.provider,
            usage=usage,
        )

    def list_embedding_models(self) -> List[ModelInfo]:
        """List available embedding models. Returns list of ModelInfo dataclasses."""
        resp = self._stub.ListEmbeddingModels(
            models_pb2.ListEmbeddingModelsRequest(), metadata=self._metadata
        )
        return [
            ModelInfo(
                name=m.name,
                provider=m.provider,
                capabilities=ModelCapability(
                    context_window=m.capabilities.context_window,
                ),
            )
            for m in resp.models
        ]

    # ==================== Prompt Management ====================

    def register_prompt(
        self,
        name: str,
        content: str,
        author: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Register a system prompt with versioning."""
        metadata_pb = models_pb2.PromptMetadata(author=author or "", tags=tags or [])
        request = models_pb2.RegisterPromptRequest(name=name, content=content, metadata=metadata_pb)
        resp = self._stub.RegisterPrompt(request, metadata=self._metadata)
        return {"name": resp.name, "version": resp.version, "created_at": resp.created_at}

    def get_prompt(self, name: str, version: Optional[int] = None) -> Dict[str, Any]:
        """Retrieve a prompt by name and version."""
        request = models_pb2.GetPromptRequest(name=name, version=version or 0)
        prompt = self._stub.GetPrompt(request, metadata=self._metadata)
        return {
            "name": prompt.name,
            "version": prompt.version,
            "content": prompt.content,
            "metadata": {
                "author": prompt.metadata.author,
                "tags": list(prompt.metadata.tags),
            },
            "created_at": prompt.created_at,
        }

    def list_prompts(self) -> List[Dict[str, Any]]:
        """List all prompts (latest version per prompt)."""
        resp = self._stub.ListPrompts(models_pb2.ListPromptsRequest(), metadata=self._metadata)
        return [
            {
                "name": p.name,
                "version": p.version,
                "content": p.content,
                "created_at": p.created_at,
            }
            for p in resp.prompts
        ]

    # ==================== Model Registry ====================

    def register_model(
        self,
        name: str,
        endpoint: str,
        adapter_type: str,
        context_window: int = 8192,
        supports_vision: bool = False,
        supports_tools: bool = True,
        provider: Optional[str] = None,
        health_check: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Register a custom model endpoint."""
        capabilities = models_pb2.ModelCapabilities(
            context_window=context_window,
            supports_vision=supports_vision,
            supports_tools=supports_tools,
        )
        request = models_pb2.RegisterModelRequest(
            name=name,
            endpoint=endpoint,
            capabilities=capabilities,
            health_check=health_check or "",
            adapter_type=adapter_type,
            provider=provider or "",
        )
        resp = self._stub.RegisterModel(request, metadata=self._metadata)
        return {"name": resp.name, "status": resp.status, "registered_at": resp.registered_at}

    def list_registered_models(self) -> List[Dict[str, Any]]:
        """List all registered custom models."""
        resp = self._stub.ListRegisteredModels(
            models_pb2.ListRegisteredModelsRequest(), metadata=self._metadata
        )
        return [
            {
                "name": m.name,
                "endpoint": m.endpoint,
                "provider": m.provider,
                "adapter_type": m.adapter_type,
                "status": m.status,
            }
            for m in resp.models
        ]

    def get_model_status(self, name: str) -> Dict[str, Any]:
        """Get status for a registered model."""
        request = models_pb2.GetModelStatusRequest(name=name)
        status = self._stub.GetModelStatus(request, metadata=self._metadata)
        return {
            "name": status.name,
            "status": status.status,
            "last_checked": status.last_checked,
            "endpoint": status.endpoint,
        }

    # ==================== Fallback Helpers ====================

    @staticmethod
    def _build_model_chain(primary: str, fallback_config: Optional[FallbackConfig]) -> List[str]:
        """Return the ordered list of models to try: [primary, *fallbacks]."""
        if fallback_config is None or not fallback_config.enabled or not fallback_config.providers:
            return [primary]
        return [primary] + list(fallback_config.providers)

    @staticmethod
    def _should_fail_fast(error: grpc.RpcError, fallback_config: Optional[FallbackConfig]) -> bool:
        """Return True if the error should skip fallbacks entirely."""
        if fallback_config is None or not fallback_config.enabled:
            return True
        if fallback_config.fail_on:
            code_name = error.code().name
            if code_name in fallback_config.fail_on:
                return True
        return False

    def _build_chat_request(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: Optional[int],
        tools: Optional[List[Dict[str, Any]]],
        response_format: Optional[str],
        system_prompt_name: Optional[str],
    ) -> models_pb2.ChatRequest:
        pb_messages = [self._dict_to_proto_msg(m) for m in messages]
        config = models_pb2.ChatConfig(temperature=temperature)
        if max_tokens:
            config.max_tokens = max_tokens

        request = models_pb2.ChatRequest(
            model=model,
            messages=pb_messages,
            config=config,
        )
        if system_prompt_name:
            request.system_prompt_name = system_prompt_name
        if tools:
            for t in tools:
                func = t.get("function", {})
                request.tools.append(
                    models_pb2.ToolDefinition(
                        type=t.get("type", "function"),
                        function=models_pb2.FunctionDef(
                            name=func.get("name", ""),
                            description=func.get("description", ""),
                            parameters_json=json.dumps(func.get("parameters", {})),
                        ),
                    )
                )
        if response_format:
            request.response_format.CopyFrom(models_pb2.ResponseFormat(type=response_format))
        return request

    # ==================== Conversion Helpers ====================

    def _dict_to_proto_msg(self, msg_dict: Dict[str, Any]) -> models_pb2.ChatMessage:
        proto_msg = models_pb2.ChatMessage(role=msg_dict.get("role", "user"))
        content = msg_dict.get("content")
        if content is not None:
            proto_msg.content = content
        if msg_dict.get("tool_call_id"):
            proto_msg.tool_call_id = msg_dict["tool_call_id"]
        if msg_dict.get("name"):
            proto_msg.name = msg_dict["name"]
        if msg_dict.get("tool_calls"):
            for tc in msg_dict["tool_calls"]:
                func = tc.get("function", {})
                proto_msg.tool_calls.append(
                    models_pb2.ToolCall(
                        id=tc.get("id", ""),
                        type=tc.get("type", "function"),
                        function=models_pb2.ToolCallFunction(
                            name=func.get("name", ""),
                            arguments=func.get("arguments", ""),
                        ),
                    )
                )
        return proto_msg

    def _proto_to_chat_response(self, resp) -> ChatResponse:
        usage = None
        if resp.HasField("usage"):
            usage = TokenUsage(
                prompt_tokens=resp.usage.prompt_tokens,
                completion_tokens=resp.usage.completion_tokens,
                total_tokens=resp.usage.total_tokens,
            )
        tool_calls = None
        if resp.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in resp.tool_calls
            ]
        return ChatResponse(
            content=resp.content if resp.content else None,
            model=resp.model,
            provider=resp.provider,
            usage=usage,
            tool_calls=tool_calls,
            finish_reason=resp.finish_reason,
        )
