"""Tests for ModelProvider ABC and adapter interfaces."""

import pytest
from services.models.models import (
    ChatConfig,
    ChatMessage,
    ChatResponse,
    ChatChunk,
    ModelInfo,
)
from services.models.providers.base import ModelProvider


class StubProvider(ModelProvider):
    """Minimal provider for testing the ABC contract."""

    def chat(self, model, messages, config, tools=None, response_format=None, system_prompt=None):
        return ChatResponse(content="stub reply", model=model, provider="stub")

    def chat_stream(self, model, messages, config, tools=None, response_format=None, system_prompt=None):
        yield ChatChunk(token="chunk", model=model)
        yield ChatChunk(token="", model=model, finish_reason="stop")

    def get_supported_models(self):
        return [ModelInfo(name="stub-model", provider="stub")]


class TestModelProviderABC:
    def test_chat_returns_domain_type(self):
        provider = StubProvider()
        msg = ChatMessage(role="user", content="Hello")
        resp = provider.chat("stub-model", [msg], ChatConfig())
        assert isinstance(resp, ChatResponse)
        assert resp.content == "stub reply"
        assert resp.provider == "stub"

    def test_chat_stream_yields_chunks(self):
        provider = StubProvider()
        msg = ChatMessage(role="user", content="Hello")
        chunks = list(provider.chat_stream("stub-model", [msg], ChatConfig()))
        assert len(chunks) == 2
        assert isinstance(chunks[0], ChatChunk)
        assert chunks[0].token == "chunk"
        assert chunks[1].finish_reason == "stop"

    def test_get_supported_models(self):
        provider = StubProvider()
        models = provider.get_supported_models()
        assert len(models) == 1
        assert isinstance(models[0], ModelInfo)

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            ModelProvider()
