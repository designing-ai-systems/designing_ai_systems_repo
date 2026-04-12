"""Tests for Model Service domain models (book Listings 3.2-3.4, 3.11-3.15)."""

from services.models.models import (
    CacheConfig,
    ChatChunk,
    ChatConfig,
    ChatMessage,
    ChatResponse,
    FallbackConfig,
    FunctionDefinition,
    ModelCapability,
    ModelInfo,
    RateLimitConfig,
    RequestMetrics,
    ResponseFormat,
    RetryConfig,
    RoutingConfig,
    TokenUsage,
    ToolDefinition,
)


class TestChatMessage:
    def test_user_message(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.tool_calls is None

    def test_system_message(self):
        msg = ChatMessage(role="system", content="You are helpful.")
        assert msg.role == "system"


class TestChatConfig:
    def test_defaults(self):
        cfg = ChatConfig()
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 512
        assert cfg.top_p == 1.0

    def test_override(self):
        cfg = ChatConfig(temperature=0.0, max_tokens=4096)
        assert cfg.temperature == 0.0
        assert cfg.max_tokens == 4096


class TestTokenUsage:
    def test_usage(self):
        u = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        assert u.total_tokens == 30


class TestChatResponse:
    def test_basic_response(self):
        resp = ChatResponse(
            content="Hello there!",
            model="gpt-4o",
            provider="openai",
            usage=TokenUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
            finish_reason="stop",
        )
        assert resp.content == "Hello there!"
        assert resp.model == "gpt-4o"
        assert resp.usage.total_tokens == 8

    def test_tool_call_response(self):
        resp = ChatResponse(
            content=None,
            model="gpt-4o",
            provider="openai",
            finish_reason="tool_calls",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                }
            ],
        )
        assert resp.finish_reason == "tool_calls"
        assert len(resp.tool_calls) == 1


class TestChatChunk:
    def test_token_chunk(self):
        chunk = ChatChunk(token="Hello", model="gpt-4o")
        assert chunk.token == "Hello"
        assert chunk.finish_reason is None

    def test_final_chunk(self):
        chunk = ChatChunk(
            token="",
            model="gpt-4o",
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        )
        assert chunk.finish_reason == "stop"


class TestModelCapability:
    def test_defaults(self):
        cap = ModelCapability()
        assert cap.context_window == 0
        assert cap.supports_vision is False
        assert cap.supports_tools is False
        assert cap.supports_streaming is True


class TestModelInfo:
    def test_model_info(self):
        info = ModelInfo(
            name="gpt-4o",
            provider="openai",
            capabilities=ModelCapability(context_window=128000, supports_tools=True),
        )
        assert info.name == "gpt-4o"
        assert info.capabilities.context_window == 128000


class TestToolDefinition:
    def test_tool(self):
        func = FunctionDefinition(
            name="get_weather",
            description="Get weather for a city",
            parameters={"type": "object", "properties": {}},
        )
        tool = ToolDefinition(type="function", function=func)
        assert tool.function.name == "get_weather"


class TestResponseFormat:
    def test_json_mode(self):
        fmt = ResponseFormat(type="json_object")
        assert fmt.type == "json_object"


class TestRetryConfig:
    def test_defaults(self):
        cfg = RetryConfig()
        assert cfg.max_retries == 3
        assert cfg.initial_delay == 1.0
        assert cfg.exponential_backoff is True
        assert cfg.max_delay == 60.0
        assert cfg.retry_on is None


class TestFallbackConfig:
    def test_defaults(self):
        cfg = FallbackConfig()
        assert cfg.enabled is True
        assert cfg.providers is None
        assert cfg.retry_config is None
        assert cfg.fail_on is None

    def test_with_providers_and_retry(self):
        cfg = FallbackConfig(
            providers=["claude-sonnet-4-5", "gpt-4o"],
            retry_config=RetryConfig(max_retries=2),
            fail_on=["authentication_error"],
        )
        assert len(cfg.providers) == 2
        assert cfg.retry_config.max_retries == 2
        assert cfg.fail_on == ["authentication_error"]


class TestRoutingConfig:
    def test_strategy(self):
        cfg = RoutingConfig(strategy="round_robin", models=["a", "b"])
        assert cfg.strategy == "round_robin"


class TestRateLimitConfig:
    def test_defaults(self):
        cfg = RateLimitConfig()
        assert cfg.requests_per_minute == 60


class TestCacheConfig:
    def test_defaults(self):
        cfg = CacheConfig()
        assert cfg.enabled is False


class TestRequestMetrics:
    def test_metrics(self):
        m = RequestMetrics(
            latency_ms=150.5,
            model="gpt-4o",
            provider="openai",
            cached=False,
        )
        assert m.latency_ms == 150.5
        assert m.retries == 0
