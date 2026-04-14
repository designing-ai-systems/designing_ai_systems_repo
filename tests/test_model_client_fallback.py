"""
Tests for client-side fallback logic in ModelClient.

RED/GREEN TDD: These tests are written first and must FAIL before implementation.
"""

from unittest.mock import MagicMock, patch

import grpc
import pytest

from genai_platform.clients.models import ModelClient
from services.models.models import (
    FallbackConfig,
)


def _make_proto_response(content="Hello", model="gpt-4o", provider="openai"):
    """Build a mock protobuf ChatResponse."""
    resp = MagicMock()
    resp.content = content
    resp.model = model
    resp.provider = provider
    resp.finish_reason = "stop"
    resp.HasField = lambda f: f == "usage"
    resp.usage.prompt_tokens = 10
    resp.usage.completion_tokens = 5
    resp.usage.total_tokens = 15
    resp.tool_calls = []
    return resp


def _make_proto_chunk(token="Hi", model="gpt-4o", finish_reason=None):
    """Build a mock protobuf StreamChunk."""
    chunk = MagicMock()
    chunk.token = token
    chunk.model = model
    chunk.finish_reason = finish_reason
    chunk.HasField = lambda f: False
    return chunk


def _make_client():
    """Create a ModelClient with a mocked platform and stub."""
    platform = MagicMock()
    platform.gateway_url = "localhost:50051"
    with patch("genai_platform.clients.models.models_pb2_grpc.ModelServiceStub"):
        client = ModelClient(platform)
    client._stub = MagicMock()
    return client


class TestFallbackPrimarySucceeds:
    """When the primary model succeeds, fallbacks are never tried."""

    def test_chat_returns_primary_response(self):
        client = _make_client()
        client._stub.Chat.return_value = _make_proto_response(
            content="primary ok", model="claude-sonnet-4-5", provider="anthropic"
        )

        fallback = FallbackConfig(
            providers=["gpt-4o", "gpt-4o-mini"],
        )
        resp = client.chat(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "Hi"}],
            fallback_config=fallback,
        )
        assert resp.content == "primary ok"
        assert resp.model == "claude-sonnet-4-5"
        assert client._stub.Chat.call_count == 1

    def test_chat_stream_returns_primary_chunks(self):
        client = _make_client()
        client._stub.ChatStream.return_value = [
            _make_proto_chunk(token="Hi", model="claude-sonnet-4-5"),
        ]

        fallback = FallbackConfig(providers=["gpt-4o"])
        chunks = list(
            client.chat_stream(
                model="claude-sonnet-4-5",
                messages=[{"role": "user", "content": "Hi"}],
                fallback_config=fallback,
            )
        )
        assert len(chunks) == 1
        assert chunks[0].token == "Hi"
        assert client._stub.ChatStream.call_count == 1


class TestFallbackTriggered:
    """When primary fails, the client should try fallback providers."""

    def test_chat_falls_back_on_rpc_error(self):
        client = _make_client()

        rpc_error = grpc.RpcError()
        rpc_error.code = lambda: grpc.StatusCode.UNAVAILABLE
        rpc_error.details = lambda: "Connection refused"

        client._stub.Chat.side_effect = [
            rpc_error,
            _make_proto_response(content="fallback ok", model="gpt-4o", provider="openai"),
        ]

        fallback = FallbackConfig(providers=["gpt-4o"])
        resp = client.chat(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "Hi"}],
            fallback_config=fallback,
        )
        assert resp.content == "fallback ok"
        assert resp.model == "gpt-4o"
        assert client._stub.Chat.call_count == 2

    def test_chat_stream_falls_back_on_rpc_error(self):
        client = _make_client()

        rpc_error = grpc.RpcError()
        rpc_error.code = lambda: grpc.StatusCode.UNAVAILABLE
        rpc_error.details = lambda: "Connection refused"

        client._stub.ChatStream.side_effect = [
            rpc_error,
            [_make_proto_chunk(token="fallback", model="gpt-4o")],
        ]

        fallback = FallbackConfig(providers=["gpt-4o"])
        chunks = list(
            client.chat_stream(
                model="claude-sonnet-4-5",
                messages=[{"role": "user", "content": "Hi"}],
                fallback_config=fallback,
            )
        )
        assert len(chunks) == 1
        assert chunks[0].token == "fallback"
        assert client._stub.ChatStream.call_count == 2

    def test_chat_tries_multiple_fallbacks(self):
        client = _make_client()

        rpc_error = grpc.RpcError()
        rpc_error.code = lambda: grpc.StatusCode.INTERNAL
        rpc_error.details = lambda: "Server error"

        client._stub.Chat.side_effect = [
            rpc_error,  # primary fails
            rpc_error,  # first fallback fails
            _make_proto_response(content="third ok", model="gpt-4o-mini", provider="openai"),
        ]

        fallback = FallbackConfig(providers=["gpt-4o", "gpt-4o-mini"])
        resp = client.chat(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "Hi"}],
            fallback_config=fallback,
        )
        assert resp.content == "third ok"
        assert resp.model == "gpt-4o-mini"
        assert client._stub.Chat.call_count == 3


class TestFailOnShortCircuit:
    """When error matches fail_on, skip fallback and raise immediately."""

    def test_chat_raises_immediately_on_fail_on_match(self):
        client = _make_client()

        rpc_error = grpc.RpcError()
        rpc_error.code = lambda: grpc.StatusCode.PERMISSION_DENIED
        rpc_error.details = lambda: "authentication_error"

        client._stub.Chat.side_effect = rpc_error

        fallback = FallbackConfig(
            providers=["gpt-4o"],
            fail_on=["PERMISSION_DENIED"],
        )
        with pytest.raises(grpc.RpcError):
            client.chat(
                model="claude-sonnet-4-5",
                messages=[{"role": "user", "content": "Hi"}],
                fallback_config=fallback,
            )
        assert client._stub.Chat.call_count == 1

    def test_chat_stream_raises_immediately_on_fail_on_match(self):
        client = _make_client()

        rpc_error = grpc.RpcError()
        rpc_error.code = lambda: grpc.StatusCode.PERMISSION_DENIED
        rpc_error.details = lambda: "authentication_error"

        client._stub.ChatStream.side_effect = rpc_error

        fallback = FallbackConfig(
            providers=["gpt-4o"],
            fail_on=["PERMISSION_DENIED"],
        )
        with pytest.raises(grpc.RpcError):
            list(
                client.chat_stream(
                    model="claude-sonnet-4-5",
                    messages=[{"role": "user", "content": "Hi"}],
                    fallback_config=fallback,
                )
            )
        assert client._stub.ChatStream.call_count == 1


class TestFallbackDisabled:
    """When fallback is disabled, errors propagate immediately."""

    def test_chat_raises_when_fallback_disabled(self):
        client = _make_client()

        rpc_error = grpc.RpcError()
        rpc_error.code = lambda: grpc.StatusCode.UNAVAILABLE
        rpc_error.details = lambda: "down"

        client._stub.Chat.side_effect = rpc_error

        fallback = FallbackConfig(enabled=False, providers=["gpt-4o"])
        with pytest.raises(grpc.RpcError):
            client.chat(
                model="claude-sonnet-4-5",
                messages=[{"role": "user", "content": "Hi"}],
                fallback_config=fallback,
            )
        assert client._stub.Chat.call_count == 1


class TestNoFallbackConfig:
    """Without fallback_config, errors propagate as before (no regression)."""

    def test_chat_raises_without_fallback(self):
        client = _make_client()

        rpc_error = grpc.RpcError()
        rpc_error.code = lambda: grpc.StatusCode.UNAVAILABLE
        rpc_error.details = lambda: "down"

        client._stub.Chat.side_effect = rpc_error

        with pytest.raises(grpc.RpcError):
            client.chat(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hi"}],
            )


class TestAllFallbacksFail:
    """When primary and all fallbacks fail, the last error is raised."""

    def test_chat_raises_last_error(self):
        client = _make_client()

        rpc_error = grpc.RpcError()
        rpc_error.code = lambda: grpc.StatusCode.UNAVAILABLE
        rpc_error.details = lambda: "down"

        client._stub.Chat.side_effect = rpc_error

        fallback = FallbackConfig(providers=["gpt-4o", "gpt-4o-mini"])
        with pytest.raises(grpc.RpcError):
            client.chat(
                model="claude-sonnet-4-5",
                messages=[{"role": "user", "content": "Hi"}],
                fallback_config=fallback,
            )
        # primary + 2 fallbacks = 3 attempts
        assert client._stub.Chat.call_count == 3


class TestBuildChatRequestTools:
    """ChatRequest must carry OpenAI-style tool definitions for Model Service."""

    def test_build_chat_request_serializes_tools(self):
        client = _make_client()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "scheduling_check_availability",
                    "description": "Check provider availability.",
                    "parameters": {
                        "type": "object",
                        "required": ["provider_id"],
                        "properties": {
                            "provider_id": {"type": "string"},
                            "date_range": {"type": "string"},
                        },
                    },
                },
            }
        ]
        req = client._build_chat_request(
            "gpt-4o",
            [{"role": "user", "content": "Use the tool."}],
            0.0,
            256,
            tools,
            None,
            None,
        )
        assert len(req.tools) == 1
        assert req.tools[0].type == "function"
        assert req.tools[0].function.name == "scheduling_check_availability"
        assert req.tools[0].function.description == "Check provider availability."
        assert '"provider_id"' in req.tools[0].function.parameters_json
