"""Tests for MCP server registration and MCP-routed execution (Listing 6.18)."""

import json

from mcp import types

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


class FakeMCPClient:
    """In-memory stand-in for an MCP session.

    Matches the protocol our real MCPClient exposes: list_tools() returns
    mcp.types.Tool; call_tool() returns mcp.types.CallToolResult.
    """

    def __init__(self, tools, call_handler=None):
        self._tools = tools
        self._call_handler = call_handler or (lambda name, args: {"ok": True, "name": name})
        self.calls: list[tuple[str, dict]] = []
        self.closed = False

    async def list_tools(self):
        return list(self._tools)

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        payload = self._call_handler(name, args)
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(payload))]
        )

    async def close(self):
        self.closed = True


def _connector(client):
    async def _connect(server_url: str, auth_token: str = ""):
        client._last_server_url = server_url
        client._last_auth_token = auth_token
        return client

    return _connect


GITHUB_TOOLS = [
    types.Tool(
        name="create_issue",
        description="Create a GitHub issue",
        inputSchema={
            "type": "object",
            "required": ["title"],
            "properties": {"title": {"type": "string"}, "body": {"type": "string"}},
        },
    ),
    types.Tool(
        name="list_repos",
        description="List repos for an owner",
        inputSchema={
            "type": "object",
            "required": ["owner"],
            "properties": {"owner": {"type": "string"}},
        },
    ),
    types.Tool(
        name="close_issue",
        description="Close an issue",
        inputSchema={
            "type": "object",
            "required": ["issue_number"],
            "properties": {"issue_number": {"type": "integer"}},
        },
    ),
]


class TestRegisterMcpServer:
    async def test_imports_tools_into_registry(self):
        client = FakeMCPClient(GITHUB_TOOLS)
        svc = ToolServiceImpl(mcp_connector=_connector(client))

        resp = await svc.RegisterMcpServer(
            tools_pb2.RegisterMcpServerRequest(
                server_url="https://mcp.github.example/mcp",
                namespace="devtools.github",
            ),
            FakeContext(),
        )

        assert set(resp.imported_tool_names) == {
            "devtools.github.create_issue",
            "devtools.github.list_repos",
            "devtools.github.close_issue",
        }
        discovered = svc.registry.discover(namespace="devtools.github.*")
        assert {t.name for t in discovered} == set(resp.imported_tool_names)

    async def test_imported_tools_carry_policy_overrides(self):
        client = FakeMCPClient(GITHUB_TOOLS[:1])
        svc = ToolServiceImpl(mcp_connector=_connector(client))

        request = tools_pb2.RegisterMcpServerRequest(
            server_url="https://mcp.github.example/mcp",
            namespace="devtools.github",
        )
        request.policy_overrides.CopyFrom(
            tools_pb2.ToolBehavior(requires_confirmation=True)
        )
        request.rate_limit_overrides.CopyFrom(
            tools_pb2.RateLimits(requests_per_minute=30)
        )

        await svc.RegisterMcpServer(request, FakeContext())

        tool = svc.registry.get("devtools.github.create_issue")
        assert tool is not None
        assert tool.behavior is not None
        assert tool.behavior.requires_confirmation is True
        assert tool.rate_limits is not None
        assert tool.rate_limits.requests_per_minute == 30

    async def test_unreachable_server_returns_error_and_no_partial_state(self):
        async def bad_connector(server_url, auth_token=""):
            raise ConnectionError("connect refused")

        svc = ToolServiceImpl(mcp_connector=bad_connector)
        ctx = FakeContext()

        resp = await svc.RegisterMcpServer(
            tools_pb2.RegisterMcpServerRequest(
                server_url="https://unreachable.example/mcp",
                namespace="devtools.github",
            ),
            ctx,
        )

        assert list(resp.imported_tool_names) == []
        assert ctx.code is not None
        assert svc.registry.discover(namespace="devtools.github.*") == []

    async def test_credential_ref_passed_to_connector(self):
        client = FakeMCPClient(GITHUB_TOOLS[:1])
        svc = ToolServiceImpl(mcp_connector=_connector(client))
        await svc.credential_store.store(
            "github-mcp-token",
            "bearer",
            "tok-abc",
            allowed_tools=["mcp://https://mcp.github.example/mcp"],
        )

        await svc.RegisterMcpServer(
            tools_pb2.RegisterMcpServerRequest(
                server_url="https://mcp.github.example/mcp",
                namespace="devtools.github",
                credential_ref="github-mcp-token",
            ),
            FakeContext(),
        )

        assert client._last_auth_token == "tok-abc"


class TestMcpToolExecution:
    async def test_execute_mcp_tool_routes_through_mcp_client(self):
        client = FakeMCPClient(
            GITHUB_TOOLS,
            call_handler=lambda name, args: {"created": True, "issue_id": 42, "got": args},
        )
        svc = ToolServiceImpl(mcp_connector=_connector(client))

        await svc.RegisterMcpServer(
            tools_pb2.RegisterMcpServerRequest(
                server_url="https://mcp.github.example/mcp",
                namespace="devtools.github",
            ),
            FakeContext(),
        )

        resp = await svc.ExecuteTool(
            tools_pb2.ExecuteToolRequest(
                tool_name="devtools.github.create_issue",
                arguments_json='{"title": "Bug"}',
            ),
            FakeContext(),
        )

        assert resp.success is True
        result = json.loads(resp.result_json)
        assert result["created"] is True
        assert result["issue_id"] == 42
        # Tool name passed to MCP is the original (no namespace prefix)
        assert client.calls == [("create_issue", {"title": "Bug"})]

    async def test_mcp_tool_failure_records_into_circuit_breaker(self):
        def boom(name, args):
            raise RuntimeError(f"boom on {name}")

        client = FakeMCPClient(GITHUB_TOOLS[:1], call_handler=boom)
        cb = CircuitBreaker(failure_threshold=2)
        svc = ToolServiceImpl(mcp_connector=_connector(client), circuit_breaker=cb)

        await svc.RegisterMcpServer(
            tools_pb2.RegisterMcpServerRequest(
                server_url="https://mcp.github.example/mcp",
                namespace="devtools.github",
            ),
            FakeContext(),
        )

        # Two failures to trip the breaker.
        for _ in range(2):
            r = await svc.ExecuteTool(
                tools_pb2.ExecuteToolRequest(
                    tool_name="devtools.github.create_issue",
                    arguments_json='{"title": "X"}',
                ),
                FakeContext(),
            )
            assert r.success is False

        # Third attempt: circuit open, no call made.
        r3 = await svc.ExecuteTool(
            tools_pb2.ExecuteToolRequest(
                tool_name="devtools.github.create_issue",
                arguments_json='{"title": "Y"}',
            ),
            FakeContext(),
        )
        assert r3.success is False
        assert "Circuit breaker" in r3.error
        assert len(client.calls) == 2  # third call never reached the MCP client
