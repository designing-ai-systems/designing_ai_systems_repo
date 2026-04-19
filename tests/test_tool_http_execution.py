"""Tests for real HTTP execution in ToolServiceImpl (Listing 6.12-6.14)."""

import json

import httpx

from proto import tools_pb2
from services.tools.circuit_breaker import CircuitBreaker
from services.tools.service import ToolServiceImpl


class FakeContext:
    def __init__(self):
        self.code = None
        self.details_str = None

    def set_code(self, code):
        self.code = code

    def set_details(self, details):
        self.details_str = details


def _mock_client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def _register_tool(
    svc,
    *,
    name,
    endpoint="",
    credential_ref="",
    timeout_seconds=0,
    max_response_size_kb=0,
):
    tool = tools_pb2.ToolDefinition(
        name=name,
        description=name,
        endpoint=endpoint,
        credential_ref=credential_ref,
    )
    if timeout_seconds or max_response_size_kb:
        tool.execution_limits.CopyFrom(
            tools_pb2.ExecutionLimits(
                timeout_seconds=timeout_seconds,
                max_response_size_kb=max_response_size_kb,
            )
        )
    await svc.RegisterTool(tools_pb2.RegisterToolRequest(tool=tool), FakeContext())


class TestHttpExecution:
    async def test_execute_posts_to_tool_endpoint(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["method"] = request.method
            captured["body"] = request.content.decode()
            return httpx.Response(200, json={"ok": True})

        svc = ToolServiceImpl(http_client=_mock_client(handler))
        await _register_tool(svc, name="test.tool", endpoint="https://api.example.com/foo")

        resp = await svc.ExecuteTool(
            tools_pb2.ExecuteToolRequest(
                tool_name="test.tool",
                arguments_json='{"patient": "p1"}',
            ),
            FakeContext(),
        )

        assert resp.success is True
        assert captured["method"] == "POST"
        assert captured["url"] == "https://api.example.com/foo"
        assert json.loads(captured["body"]) == {"patient": "p1"}

    async def test_response_returned_verbatim_on_success(self):
        def handler(request):
            return httpx.Response(200, json={"appointment_id": "apt-123", "status": "booked"})

        svc = ToolServiceImpl(http_client=_mock_client(handler))
        await _register_tool(svc, name="test.tool", endpoint="https://api.example.com/book")

        resp = await svc.ExecuteTool(
            tools_pb2.ExecuteToolRequest(tool_name="test.tool", arguments_json="{}"),
            FakeContext(),
        )

        assert resp.success is True
        assert json.loads(resp.result_json) == {
            "appointment_id": "apt-123",
            "status": "booked",
        }

    async def test_non_2xx_returns_success_false(self):
        def handler(request):
            return httpx.Response(500, json={"error": "internal"})

        svc = ToolServiceImpl(http_client=_mock_client(handler))
        await _register_tool(svc, name="test.tool", endpoint="https://api.example.com/fail")

        resp = await svc.ExecuteTool(
            tools_pb2.ExecuteToolRequest(tool_name="test.tool", arguments_json="{}"),
            FakeContext(),
        )

        assert resp.success is False
        assert "500" in resp.error


class TestCredentialInjection:
    async def test_api_key_injected_as_x_api_key_header(self):
        captured = {}

        def handler(request):
            captured["headers"] = dict(request.headers)
            return httpx.Response(200, json={"ok": True})

        svc = ToolServiceImpl(http_client=_mock_client(handler))
        await svc.credential_store.store(
            "sched-key", "api_key", "secret-xyz", allowed_tools=["test.tool"]
        )
        await _register_tool(
            svc,
            name="test.tool",
            endpoint="https://api.example.com/v1",
            credential_ref="sched-key",
        )

        await svc.ExecuteTool(
            tools_pb2.ExecuteToolRequest(tool_name="test.tool", arguments_json="{}"),
            FakeContext(),
        )

        assert captured["headers"].get("x-api-key") == "secret-xyz"
        assert "authorization" not in captured["headers"]

    async def test_bearer_injected_as_authorization_header(self):
        captured = {}

        def handler(request):
            captured["headers"] = dict(request.headers)
            return httpx.Response(200, json={"ok": True})

        svc = ToolServiceImpl(http_client=_mock_client(handler))
        await svc.credential_store.store(
            "bearer-key", "bearer", "tok-abc", allowed_tools=["test.tool"]
        )
        await _register_tool(
            svc,
            name="test.tool",
            endpoint="https://api.example.com/v1",
            credential_ref="bearer-key",
        )

        await svc.ExecuteTool(
            tools_pb2.ExecuteToolRequest(tool_name="test.tool", arguments_json="{}"),
            FakeContext(),
        )

        assert captured["headers"].get("authorization") == "Bearer tok-abc"

    async def test_oauth2_injected_as_authorization_bearer(self):
        captured = {}

        def handler(request):
            captured["headers"] = dict(request.headers)
            return httpx.Response(200, json={"ok": True})

        svc = ToolServiceImpl(http_client=_mock_client(handler))
        await svc.credential_store.store(
            "oauth-key", "oauth2", "access-token-xyz", allowed_tools=["test.tool"]
        )
        await _register_tool(
            svc,
            name="test.tool",
            endpoint="https://api.example.com/v1",
            credential_ref="oauth-key",
        )

        await svc.ExecuteTool(
            tools_pb2.ExecuteToolRequest(tool_name="test.tool", arguments_json="{}"),
            FakeContext(),
        )

        assert captured["headers"].get("authorization") == "Bearer access-token-xyz"

    async def test_no_credential_sends_no_auth_header(self):
        captured = {}

        def handler(request):
            captured["headers"] = dict(request.headers)
            return httpx.Response(200, json={"ok": True})

        svc = ToolServiceImpl(http_client=_mock_client(handler))
        await _register_tool(svc, name="test.tool", endpoint="https://api.example.com/v1")

        await svc.ExecuteTool(
            tools_pb2.ExecuteToolRequest(tool_name="test.tool", arguments_json="{}"),
            FakeContext(),
        )

        assert "authorization" not in captured["headers"]
        assert "x-api-key" not in captured["headers"]


class TestExecutionLimits:
    async def test_timeout_enforced_from_execution_limits(self):
        def handler(request):
            raise httpx.ReadTimeout("simulated timeout", request=request)

        svc = ToolServiceImpl(http_client=_mock_client(handler))
        await _register_tool(
            svc,
            name="test.tool",
            endpoint="https://api.example.com/slow",
            timeout_seconds=1,
        )

        resp = await svc.ExecuteTool(
            tools_pb2.ExecuteToolRequest(tool_name="test.tool", arguments_json="{}"),
            FakeContext(),
        )

        assert resp.success is False
        assert "timeout" in resp.error.lower()

    async def test_oversized_response_rejected(self):
        # 5 KB body, limit is 1 KB
        big_payload = {"blob": "x" * (5 * 1024)}

        def handler(request):
            return httpx.Response(200, json=big_payload)

        svc = ToolServiceImpl(http_client=_mock_client(handler))
        await _register_tool(
            svc,
            name="test.tool",
            endpoint="https://api.example.com/big",
            max_response_size_kb=1,
        )

        resp = await svc.ExecuteTool(
            tools_pb2.ExecuteToolRequest(tool_name="test.tool", arguments_json="{}"),
            FakeContext(),
        )

        assert resp.success is False
        assert "max_response_size" in resp.error or "too large" in resp.error.lower()


class TestCircuitBreakerIntegration:
    async def test_failed_http_call_records_into_circuit_breaker(self):
        cb = CircuitBreaker(failure_threshold=2)

        def handler(request):
            return httpx.Response(500, json={"err": "boom"})

        svc = ToolServiceImpl(http_client=_mock_client(handler), circuit_breaker=cb)
        await _register_tool(svc, name="test.tool", endpoint="https://api.example.com/fail")

        # First failure: allowed but recorded
        resp1 = await svc.ExecuteTool(
            tools_pb2.ExecuteToolRequest(tool_name="test.tool", arguments_json="{}"),
            FakeContext(),
        )
        assert resp1.success is False

        # Second failure: trips the breaker
        resp2 = await svc.ExecuteTool(
            tools_pb2.ExecuteToolRequest(tool_name="test.tool", arguments_json="{}"),
            FakeContext(),
        )
        assert resp2.success is False

        # Third call: breaker open, rejected before HTTP attempt
        resp3 = await svc.ExecuteTool(
            tools_pb2.ExecuteToolRequest(tool_name="test.tool", arguments_json="{}"),
            FakeContext(),
        )
        assert resp3.success is False
        assert "Circuit breaker" in resp3.error
