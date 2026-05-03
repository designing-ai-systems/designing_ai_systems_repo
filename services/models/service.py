"""
Model Service - Business logic layer.

Architecture:
- Provider adapters auto-discover their models (OpenAI, Anthropic)
- ModelRegistry stores explicit registrations (custom + overrides)
- Resolution: Check registry first (overrides), then adapters (defaults)
- Servicer converts between proto and domain types at the boundary.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 3.5: ModelService gRPC definition (proto side)
  - Listing 3.6: ChatRequest Protocol Buffer definition (proto side)
  - Listing 3.7: ChatResponse Protocol Buffer definition (proto side)
  - Listing 3.8: Provider adapter interface (used by this servicer)
  - Listing 7.3: log_fallback_triggered helper (provider fallback log)
  - Listing 7.7: Chat method with automatic generation tracking
  - Listing 7.8: Metrics publisher integration
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional

import grpc

from proto import models_pb2, models_pb2_grpc
from services.models.embedding_providers.base import EmbeddingProvider
from services.models.embedding_providers.openai_provider import OpenAIEmbeddingProvider
from services.models.metrics_publisher import (
    ModelRequestMetrics,
    ModelServiceMetricsPublisher,
)
from services.models.models import (
    ChatChunk,
    ChatConfig,
    ChatMessage,
    ChatResponse,
    FunctionDefinition,
    ResponseFormat,
    ToolDefinition,
)
from services.models.providers import AnthropicProvider, ModelProvider, OpenAIProvider
from services.models.store import ModelRegistry, PromptRegistry
from services.shared.observability_client import ObservabilityClient
from services.shared.servicer_base import BaseServicer
from services.shared.traced_service import TraceContext, TracedService


class ModelService(models_pb2_grpc.ModelServiceServicer, BaseServicer, TracedService):
    """The Model Service.

    Inherits from :class:`TracedService` so every Chat call produces a
    Generation in the observability backend (Listing 7.7). Pass an
    :class:`ObservabilityClient` to wire telemetry; the default
    :py:meth:`ObservabilityClient.null` keeps the service usable in
    tests and in deployments where observability is disabled.
    """

    def __init__(self, observability: Optional[ObservabilityClient] = None):
        observability = observability or ObservabilityClient.null()
        TracedService.__init__(self, "models", observability)
        self._providers: Dict[str, ModelProvider] = {}
        self._embedding_providers: Dict[str, EmbeddingProvider] = {}
        self._initialize_providers()
        self._initialize_embedding_providers()
        self._model_registry = ModelRegistry()
        self._prompts = PromptRegistry()
        self._metrics_publisher = ModelServiceMetricsPublisher(observability)

    def add_to_server(self, server: grpc.Server):
        models_pb2_grpc.add_ModelServiceServicer_to_server(self, server)

    # ==================== Proto <-> Domain Conversion ====================

    def _proto_msg_to_domain(self, proto_msg) -> ChatMessage:
        tool_calls = None
        if proto_msg.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in proto_msg.tool_calls
            ]
        return ChatMessage(
            role=proto_msg.role,
            content=proto_msg.content if proto_msg.HasField("content") else None,
            tool_calls=tool_calls,
            tool_call_id=proto_msg.tool_call_id or None,
            name=proto_msg.name if proto_msg.HasField("name") else None,
        )

    def _proto_config_to_domain(self, proto_config) -> ChatConfig:
        return ChatConfig(
            temperature=proto_config.temperature,
            max_tokens=proto_config.max_tokens,
            top_p=proto_config.top_p,
            stop=list(proto_config.stop_sequences) if proto_config.stop_sequences else None,
            frequency_penalty=proto_config.frequency_penalty,
            presence_penalty=proto_config.presence_penalty,
        )

    def _proto_tools_to_domain(
        self,
        proto_tools,
    ) -> Optional[List[ToolDefinition]]:
        if not proto_tools:
            return None
        result = []
        for t in proto_tools:
            params = None
            if t.function.parameters_json:
                params = json.loads(t.function.parameters_json)
            result.append(
                ToolDefinition(
                    type=t.type or "function",
                    function=FunctionDefinition(
                        name=t.function.name,
                        description=t.function.description,
                        parameters=params,
                    ),
                )
            )
        return result

    def _domain_response_to_proto(self, resp: ChatResponse) -> models_pb2.ChatResponse:
        usage = None
        if resp.usage:
            usage = models_pb2.TokenUsage(
                prompt_tokens=resp.usage.prompt_tokens,
                completion_tokens=resp.usage.completion_tokens,
                total_tokens=resp.usage.total_tokens,
            )
        tool_calls = []
        if resp.tool_calls:
            for tc in resp.tool_calls:
                tool_calls.append(
                    models_pb2.ToolCall(
                        id=tc.get("id", ""),
                        type=tc.get("type", "function"),
                        function=models_pb2.ToolCallFunction(
                            name=tc["function"]["name"],
                            arguments=tc["function"]["arguments"],
                        ),
                    )
                )
        return models_pb2.ChatResponse(
            content=resp.content or "",
            model=resp.model,
            provider=resp.provider,
            usage=usage,
            tool_calls=tool_calls,
            finish_reason=resp.finish_reason or "stop",
        )

    def _domain_chunk_to_proto(self, chunk: ChatChunk) -> models_pb2.ChatChunk:
        proto_chunk = models_pb2.ChatChunk(
            token=chunk.token,
            model=chunk.model,
        )
        if chunk.finish_reason:
            proto_chunk.finish_reason = chunk.finish_reason
        if chunk.usage:
            proto_chunk.usage.CopyFrom(
                models_pb2.TokenUsage(
                    prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                    total_tokens=chunk.usage.total_tokens,
                )
            )
        return proto_chunk

    # ==================== Core Inference ====================

    def Chat(self, request: models_pb2.ChatRequest, context) -> models_pb2.ChatResponse:
        model_name = request.model or self._get_default_model()
        if not model_name:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details("No model specified and no providers configured.")
            return models_pb2.ChatResponse()

        provider = self._resolve_provider(model_name)
        if provider is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"No provider found for model '{model_name}'.")
            return models_pb2.ChatResponse()

        system_prompt = self._resolve_system_prompt(request, context)
        if request.system_prompt_name and system_prompt is None:
            return models_pb2.ChatResponse()

        config = (
            self._proto_config_to_domain(request.config)
            if request.HasField("config")
            else ChatConfig()
        )
        domain_msgs = [self._proto_msg_to_domain(m) for m in request.messages]
        tools = self._proto_tools_to_domain(request.tools)
        response_format = None
        if request.HasField("response_format"):
            response_format = ResponseFormat(type=request.response_format.type)

        # Listing 7.7: wrap the provider call in trace_generation so every
        # workflow that calls platform.models.chat(...) gets a generation
        # automatically, with token counts, cost, and timing, without the
        # workflow developer writing a single line of tracing code.
        trace_context = self._extract_trace_context(context)
        with self.trace_generation(
            trace_context,
            model=model_name,
            requested_model=request.model,
        ) as gen_ctx:
            domain_resp = provider.chat(
                model=model_name,
                messages=domain_msgs,
                config=config,
                tools=tools,
                response_format=response_format,
                system_prompt=system_prompt,
            )
            cost_usd = self._estimate_cost(model_name, domain_resp)
            gen_ctx.update(
                provider=getattr(provider, "name", domain_resp.provider or ""),
                prompt_tokens=domain_resp.usage.prompt_tokens if domain_resp.usage else 0,
                completion_tokens=(domain_resp.usage.completion_tokens if domain_resp.usage else 0),
                cost_usd=cost_usd,
            )

        # Listing 7.8: emit aggregate metrics so dashboards stay current.
        self._metrics_publisher.publish(
            ModelRequestMetrics(
                provider=getattr(provider, "name", domain_resp.provider or "unknown"),
                model=model_name,
                workflow_id=trace_context.workflow_id or "",
                cache_hit=False,
                fallback_used=False,
                prompt_tokens=domain_resp.usage.prompt_tokens if domain_resp.usage else 0,
                completion_tokens=(domain_resp.usage.completion_tokens if domain_resp.usage else 0),
                cost_usd=cost_usd,
                duration_ms=0.0,
            )
        )

        return self._domain_response_to_proto(domain_resp)

    def ChatStream(
        self, request: models_pb2.ChatRequest, context
    ) -> Iterable[models_pb2.ChatChunk]:
        model_name = request.model or self._get_default_model()
        if not model_name:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details("No model specified and no providers configured.")
            return

        provider = self._resolve_provider(model_name)
        if provider is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"No provider found for model '{model_name}'.")
            return

        system_prompt = self._resolve_system_prompt(request, context)
        if request.system_prompt_name and system_prompt is None:
            return

        config = (
            self._proto_config_to_domain(request.config)
            if request.HasField("config")
            else ChatConfig()
        )
        domain_msgs = [self._proto_msg_to_domain(m) for m in request.messages]
        tools = self._proto_tools_to_domain(request.tools)
        response_format = None
        if request.HasField("response_format"):
            response_format = ResponseFormat(type=request.response_format.type)

        for chunk in provider.chat_stream(
            model=model_name,
            messages=domain_msgs,
            config=config,
            tools=tools,
            response_format=response_format,
            system_prompt=system_prompt,
        ):
            yield self._domain_chunk_to_proto(chunk)

    # ==================== Discovery ====================

    def ListModels(self, request, context) -> models_pb2.ListModelsResponse:
        models_by_name = {}
        for provider in self._providers.values():
            for info in provider.get_supported_models():
                models_by_name[info.name] = models_pb2.ModelInfo(
                    name=info.name,
                    provider=info.provider,
                    capabilities=models_pb2.ModelCapabilities(
                        context_window=info.capabilities.context_window,
                        supports_vision=info.capabilities.supports_vision,
                        supports_tools=info.capabilities.supports_tools,
                        supports_streaming=info.capabilities.supports_streaming,
                        supports_json_mode=info.capabilities.supports_json_mode,
                    ),
                )
        for registered in self._model_registry.list_all():
            models_by_name[registered.name] = models_pb2.ModelInfo(
                name=registered.name,
                provider=registered.provider,
                capabilities=registered.capabilities,
            )
        return models_pb2.ListModelsResponse(models=list(models_by_name.values()))

    def GetModelCapabilities(self, request, context) -> models_pb2.ModelCapabilities:
        registered = self._model_registry.get(request.model)
        if registered:
            return registered.capabilities
        for provider in self._providers.values():
            for info in provider.get_supported_models():
                if info.name == request.model:
                    return models_pb2.ModelCapabilities(
                        context_window=info.capabilities.context_window,
                        supports_vision=info.capabilities.supports_vision,
                        supports_tools=info.capabilities.supports_tools,
                    )
        context.set_code(grpc.StatusCode.NOT_FOUND)
        context.set_details(f"Model '{request.model}' not found")
        return models_pb2.ModelCapabilities()

    # ==================== Embedding ====================

    def Embed(self, request: models_pb2.EmbedRequest, context) -> models_pb2.EmbedResponse:
        model_name = request.model
        if not model_name:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Model name is required for embedding.")
            return models_pb2.EmbedResponse()

        provider = self._resolve_embedding_provider(model_name)
        if provider is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"No embedding provider found for model '{model_name}'.")
            return models_pb2.EmbedResponse()

        domain_resp = provider.embed(list(request.texts), model_name)

        proto_embeddings = [models_pb2.Embedding(values=vec) for vec in domain_resp.embeddings]
        usage = None
        if domain_resp.usage:
            usage = models_pb2.TokenUsage(
                prompt_tokens=domain_resp.usage.prompt_tokens,
                completion_tokens=domain_resp.usage.completion_tokens,
                total_tokens=domain_resp.usage.total_tokens,
            )
        return models_pb2.EmbedResponse(
            embeddings=proto_embeddings,
            model=domain_resp.model,
            provider=domain_resp.provider,
            usage=usage,
        )

    def ListEmbeddingModels(self, request, context) -> models_pb2.ListEmbeddingModelsResponse:
        models_by_name = {}
        for provider in self._embedding_providers.values():
            for info in provider.get_supported_embedding_models():
                models_by_name[info.name] = models_pb2.ModelInfo(
                    name=info.name,
                    provider=info.provider,
                    capabilities=models_pb2.ModelCapabilities(
                        context_window=info.capabilities.context_window,
                    ),
                )
        return models_pb2.ListEmbeddingModelsResponse(models=list(models_by_name.values()))

    # ==================== Prompt Management ====================

    def RegisterPrompt(self, request, context) -> models_pb2.RegisterPromptResponse:
        prompt = self._prompts.register(
            name=request.name,
            content=request.content,
            metadata=request.metadata if request.HasField("metadata") else None,
        )
        return models_pb2.RegisterPromptResponse(
            name=prompt.name, version=prompt.version, created_at=prompt.created_at
        )

    def GetPrompt(self, request, context) -> models_pb2.Prompt:
        prompt = self._prompts.get(request.name, request.version)
        if prompt is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Prompt '{request.name}' not found")
            return models_pb2.Prompt()
        return prompt

    def ListPrompts(self, request, context) -> models_pb2.ListPromptsResponse:
        return models_pb2.ListPromptsResponse(prompts=self._prompts.list_latest())

    # ==================== Model Registry ====================

    def RegisterModel(self, request, context) -> models_pb2.RegisterModelResponse:
        model = self._model_registry.register(
            name=request.name,
            endpoint=request.endpoint,
            capabilities=request.capabilities,
            health_check=request.health_check,
            adapter_type=request.adapter_type or "openai",
            provider=request.provider if request.provider else None,
        )
        return models_pb2.RegisterModelResponse(
            name=model.name, status=model.status, registered_at=model.registered_at
        )

    def ListRegisteredModels(self, request, context) -> models_pb2.ListRegisteredModelsResponse:
        return models_pb2.ListRegisteredModelsResponse(models=self._model_registry.list_all())

    def GetModelStatus(self, request, context) -> models_pb2.ModelStatus:
        model = self._model_registry.get(request.name)
        if model is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Model '{request.name}' not registered")
            return models_pb2.ModelStatus()
        return models_pb2.ModelStatus(
            name=model.name,
            status=model.status,
            last_checked=model.registered_at,
            endpoint=model.endpoint,
        )

    # ==================== Private Helpers ====================

    def _initialize_providers(self) -> None:
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            self._providers["openai"] = OpenAIProvider(
                api_key=openai_key, base_url=os.getenv("OPENAI_BASE_URL")
            )
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            self._providers["anthropic"] = AnthropicProvider(api_key=anthropic_key)

    def _initialize_embedding_providers(self) -> None:
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            self._embedding_providers["openai"] = OpenAIEmbeddingProvider(
                api_key=openai_key, base_url=os.getenv("OPENAI_BASE_URL")
            )
        hf_models = os.getenv("HF_EMBEDDING_MODELS")
        if hf_models:
            from services.models.embedding_providers.huggingface_provider import (
                HuggingFaceEmbeddingProvider,
            )

            model_names = [m.strip() for m in hf_models.split(",") if m.strip()]
            if model_names:
                self._embedding_providers["huggingface"] = HuggingFaceEmbeddingProvider(
                    model_names=model_names
                )

    def _resolve_provider(self, model_name: str) -> Optional[ModelProvider]:
        registered = self._model_registry.get(model_name)
        if registered:
            adapter_type = registered.adapter_type
            if adapter_type in self._providers:
                return self._providers[adapter_type]
            return None
        for provider in self._providers.values():
            for info in provider.get_supported_models():
                if info.name == model_name:
                    return provider
        return None

    def _resolve_embedding_provider(self, model_name: str) -> Optional[EmbeddingProvider]:
        for provider in self._embedding_providers.values():
            for info in provider.get_supported_embedding_models():
                if info.name == model_name:
                    return provider
        return None

    def _resolve_system_prompt(self, request, context) -> Optional[str]:
        if not request.system_prompt_name:
            return None
        prompt = self._prompts.get(request.system_prompt_name)
        if prompt is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Prompt '{request.system_prompt_name}' not found")
            return None
        return prompt.content

    def _get_default_model(self) -> Optional[str]:
        if "openai" in self._providers:
            return "gpt-4o"
        if "anthropic" in self._providers:
            return "claude-sonnet-4-5"
        return None

    def _extract_trace_context(self, context) -> TraceContext:
        """Read trace context from incoming gRPC metadata (if any).

        The Gateway seeds ``x-trace-id`` for every request; downstream
        services pass ``x-parent-span-id`` so child spans nest correctly.
        Missing values fall back to fresh defaults so the call still
        produces a self-contained trace.
        """
        import uuid

        metadata: Dict[str, str] = {}
        invocation = getattr(context, "invocation_metadata", None)
        if callable(invocation):
            try:
                metadata = {k: v for k, v in invocation()}
            except Exception:
                metadata = {}
        return TraceContext(
            trace_id=metadata.get("x-trace-id") or uuid.uuid4().hex,
            span_id=metadata.get("x-parent-span-id") or None,
            parent_span_id=metadata.get("x-parent-span-id") or None,
            workflow_id=metadata.get("x-workflow-id", ""),
            user_id=metadata.get("x-user-id", ""),
            session_id=metadata.get("x-session-id", ""),
        )

    def _estimate_cost(self, model_name: str, response: ChatResponse) -> float:
        """Best-effort cost estimate from token counts.

        Uses a small built-in price table for the supported commercial
        models; unknown models return 0.0 so the metric is still emitted
        but doesn't pretend to know the price.
        """
        if response.usage is None:
            return 0.0
        # Prices are USD per 1M tokens (input, output).
        prices = {
            "gpt-4o": (2.5, 10.0),
            "gpt-4o-mini": (0.15, 0.6),
            "claude-sonnet-4-5": (3.0, 15.0),
            "claude-opus-4-5": (15.0, 75.0),
            "claude-haiku-4-5": (0.8, 4.0),
        }
        price = prices.get(model_name)
        if price is None:
            return 0.0
        input_per_token = price[0] / 1_000_000
        output_per_token = price[1] / 1_000_000
        return (
            response.usage.prompt_tokens * input_per_token
            + response.usage.completion_tokens * output_per_token
        )

    def _log_fallback_triggered(
        self,
        original: str,
        fallback: str,
        error: Exception,
        trace_context: TraceContext,
    ) -> None:
        """Listing 7.3: emit a structured WARNING when a provider fallback fires."""
        self.observability.log(
            event_type="model_fallback",
            severity="WARNING",
            message=f"Fallback: {original} -> {fallback}",
            trace_id=trace_context.trace_id,
            span_id=trace_context.span_id or "",
            workflow_id=trace_context.workflow_id or "",
            user_id=trace_context.user_id or "",
            attributes={
                "original_provider": original,
                "fallback_provider": fallback,
                "error_type": type(error).__name__,
            },
        )
