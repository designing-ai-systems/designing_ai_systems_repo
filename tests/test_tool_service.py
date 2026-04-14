"""Tests for ToolServiceImpl gRPC servicer (grpc.aio)."""

import json

import grpc

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


class TestRegisterTool:
    async def test_register_returns_status(self):
        svc = ToolServiceImpl()
        request = tools_pb2.RegisterToolRequest(
            tool=tools_pb2.ToolDefinition(
                name="test.tool",
                version="1.0.0",
                description="A test",
            )
        )
        resp = await svc.RegisterTool(request, FakeContext())
        assert resp.name == "test.tool"
        assert resp.status == "registered"


class TestDiscoverTools:
    async def _register_tools(self, svc):
        for name, caps, tags, read_only in [
            ("scheduling.book", ["scheduling"], ["patient"], False),
            ("scheduling.check", ["scheduling", "availability"], ["patient"], True),
            ("billing.verify", ["insurance"], ["hipaa"], True),
        ]:
            tool = tools_pb2.ToolDefinition(
                name=name,
                description=name,
                capabilities=caps,
                tags=tags,
            )
            tool.behavior.CopyFrom(tools_pb2.ToolBehavior(is_read_only=read_only))
            await svc.RegisterTool(tools_pb2.RegisterToolRequest(tool=tool), FakeContext())

    async def test_discover_all(self):
        svc = ToolServiceImpl()
        await self._register_tools(svc)
        resp = await svc.DiscoverTools(tools_pb2.DiscoverToolsRequest(), FakeContext())
        assert len(resp.tools) == 3

    async def test_discover_by_capability(self):
        svc = ToolServiceImpl()
        await self._register_tools(svc)
        resp = await svc.DiscoverTools(
            tools_pb2.DiscoverToolsRequest(capabilities=["insurance"]),
            FakeContext(),
        )
        assert len(resp.tools) == 1
        assert resp.tools[0].name == "billing.verify"

    async def test_discover_read_only(self):
        svc = ToolServiceImpl()
        await self._register_tools(svc)
        resp = await svc.DiscoverTools(
            tools_pb2.DiscoverToolsRequest(read_only=True), FakeContext()
        )
        assert len(resp.tools) == 2


class TestExecuteTool:
    async def test_execute_registered_tool(self):
        svc = ToolServiceImpl()
        await svc.RegisterTool(
            tools_pb2.RegisterToolRequest(
                tool=tools_pb2.ToolDefinition(name="test.tool", description="A test")
            ),
            FakeContext(),
        )
        resp = await svc.ExecuteTool(
            tools_pb2.ExecuteToolRequest(
                tool_name="test.tool",
                arguments_json='{"key": "value"}',
            ),
            FakeContext(),
        )
        assert resp.success is True
        result = json.loads(resp.result_json)
        assert result["tool"] == "test.tool"
        assert result.get("credential_injected") is None

    async def test_execute_with_credential_injection(self):
        svc = ToolServiceImpl()
        await svc.credential_store.store(
            "scheduling-api-prod",
            "api_key",
            "secret-xyz",
            allowed_tools=["test.tool"],
        )
        await svc.RegisterTool(
            tools_pb2.RegisterToolRequest(
                tool=tools_pb2.ToolDefinition(
                    name="test.tool",
                    description="A test",
                    credential_ref="scheduling-api-prod",
                )
            ),
            FakeContext(),
        )
        resp = await svc.ExecuteTool(
            tools_pb2.ExecuteToolRequest(
                tool_name="test.tool",
                arguments_json="{}",
            ),
            FakeContext(),
        )
        assert resp.success is True
        result = json.loads(resp.result_json)
        assert result.get("credential_injected") is True
        assert result.get("credential_type") == "api_key"

    async def test_execute_missing_credential(self):
        svc = ToolServiceImpl()
        await svc.RegisterTool(
            tools_pb2.RegisterToolRequest(
                tool=tools_pb2.ToolDefinition(
                    name="test.tool",
                    description="A test",
                    credential_ref="missing-ref",
                )
            ),
            FakeContext(),
        )
        resp = await svc.ExecuteTool(
            tools_pb2.ExecuteToolRequest(tool_name="test.tool"),
            FakeContext(),
        )
        assert resp.success is False
        assert "not found" in resp.error

    async def test_execute_nonexistent_tool(self):
        svc = ToolServiceImpl()
        ctx = FakeContext()
        resp = await svc.ExecuteTool(tools_pb2.ExecuteToolRequest(tool_name="nope"), ctx)
        assert resp.success is False
        assert ctx.code == grpc.StatusCode.NOT_FOUND

    async def test_circuit_breaker_blocks_requests(self):
        cb = CircuitBreaker(failure_threshold=1)
        svc = ToolServiceImpl(circuit_breaker=cb)
        cb.record_result("blocked.tool", success=False)
        resp = await svc.ExecuteTool(
            tools_pb2.ExecuteToolRequest(tool_name="blocked.tool"),
            FakeContext(),
        )
        assert resp.success is False
        assert "Circuit breaker" in resp.error


class TestValidateTool:
    async def test_validate_registered_tool(self):
        svc = ToolServiceImpl()
        await svc.RegisterTool(
            tools_pb2.RegisterToolRequest(
                tool=tools_pb2.ToolDefinition(
                    name="test.tool",
                    parameters_json='{"required": ["patient_id"]}',
                )
            ),
            FakeContext(),
        )
        resp = await svc.ValidateTool(
            tools_pb2.ValidateToolRequest(
                tool_name="test.tool",
                arguments_json='{"patient_id": "123"}',
            ),
            FakeContext(),
        )
        assert resp.valid is True

    async def test_validate_missing_required_param(self):
        svc = ToolServiceImpl()
        await svc.RegisterTool(
            tools_pb2.RegisterToolRequest(
                tool=tools_pb2.ToolDefinition(
                    name="test.tool",
                    parameters_json='{"required": ["patient_id"]}',
                )
            ),
            FakeContext(),
        )
        resp = await svc.ValidateTool(
            tools_pb2.ValidateToolRequest(
                tool_name="test.tool",
                arguments_json="{}",
            ),
            FakeContext(),
        )
        assert resp.valid is False
        assert any("patient_id" in e for e in resp.errors)

    async def test_validate_nonexistent_tool(self):
        svc = ToolServiceImpl()
        resp = await svc.ValidateTool(
            tools_pb2.ValidateToolRequest(tool_name="nope"),
            FakeContext(),
        )
        assert resp.valid is False
