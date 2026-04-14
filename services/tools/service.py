"""
Tool Service — gRPC service implementation.

Thin translation layer: receives proto requests, delegates to
ToolRegistry/CredentialStore/CircuitBreaker, returns proto responses.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 6.1: ToolService gRPC contract
  - Listing 6.4: Tool registration
  - Listing 6.7: Tool discovery
  - Listing 6.12: Sandboxed tool execution
  - Listing 6.18: CircuitBreaker integration
"""

import json
import logging
import time

import grpc

from proto import tools_pb2, tools_pb2_grpc
from services.shared.servicer_base import BaseServicer
from services.tools.circuit_breaker import CircuitBreaker
from services.tools.credential_store import CredentialStore, InMemoryCredentialStore
from services.tools.models import (
    CostMetadata,
    ExecutionLimits,
    RateLimits,
    ToolBehavior,
    ToolDefinition,
    ToolExecutionResult,
)
from services.tools.store import InMemoryToolRegistry, ToolRegistry

logger = logging.getLogger(__name__)


class ToolServiceImpl(tools_pb2_grpc.ToolServiceServicer, BaseServicer):
    def __init__(
        self,
        registry: ToolRegistry | None = None,
        credential_store: CredentialStore | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ):
        self.registry = registry or InMemoryToolRegistry()
        self.credential_store = credential_store or InMemoryCredentialStore()
        self.circuit_breaker = circuit_breaker or CircuitBreaker()

    def add_to_server(self, server: grpc.Server):
        tools_pb2_grpc.add_ToolServiceServicer_to_server(self, server)

    # --- proto <-> domain ---

    def _proto_to_domain(self, proto: tools_pb2.ToolDefinition) -> ToolDefinition:
        behavior = None
        if proto.HasField("behavior"):
            behavior = ToolBehavior(
                is_read_only=proto.behavior.is_read_only,
                is_idempotent=proto.behavior.is_idempotent,
                requires_confirmation=proto.behavior.requires_confirmation,
                typical_latency_ms=proto.behavior.typical_latency_ms,
                side_effects=list(proto.behavior.side_effects),
            )
        rate_limits = None
        if proto.HasField("rate_limits"):
            rate_limits = RateLimits(
                requests_per_minute=proto.rate_limits.requests_per_minute,
                requests_per_session=proto.rate_limits.requests_per_session,
                daily_limit=proto.rate_limits.daily_limit,
            )
        cost = None
        if proto.HasField("cost"):
            cost = CostMetadata(
                estimated_cost_usd=proto.cost.estimated_cost_usd,
                billing_category=proto.cost.billing_category,
            )
        execution_limits = None
        if proto.HasField("execution_limits"):
            execution_limits = ExecutionLimits(
                timeout_seconds=proto.execution_limits.timeout_seconds,
                memory_limit_mb=proto.execution_limits.memory_limit_mb,
                cpu_limit_millicores=proto.execution_limits.cpu_limit_millicores,
                max_response_size_kb=proto.execution_limits.max_response_size_kb,
                max_retries=proto.execution_limits.max_retries,
            )
        return ToolDefinition(
            name=proto.name,
            version=proto.version or "1.0.0",
            owner=proto.owner,
            description=proto.description,
            parameters=json.loads(proto.parameters_json) if proto.parameters_json else None,
            returns_schema=json.loads(proto.returns_json) if proto.returns_json else None,
            behavior=behavior,
            rate_limits=rate_limits,
            cost=cost,
            required_permissions=list(proto.required_permissions),
            capabilities=list(proto.capabilities),
            tags=list(proto.tags),
            endpoint=proto.endpoint,
            credential_ref=proto.credential_ref,
            execution_limits=execution_limits,
        )

    def _domain_to_proto(self, tool: ToolDefinition) -> tools_pb2.ToolDefinition:
        proto = tools_pb2.ToolDefinition(
            name=tool.name,
            version=tool.version,
            owner=tool.owner,
            description=tool.description,
            parameters_json=json.dumps(tool.parameters) if tool.parameters else "",
            returns_json=json.dumps(tool.returns_schema) if tool.returns_schema else "",
            required_permissions=tool.required_permissions,
            capabilities=tool.capabilities,
            tags=tool.tags,
            endpoint=tool.endpoint,
            credential_ref=tool.credential_ref,
        )
        if tool.behavior:
            proto.behavior.CopyFrom(
                tools_pb2.ToolBehavior(
                    is_read_only=tool.behavior.is_read_only,
                    is_idempotent=tool.behavior.is_idempotent,
                    requires_confirmation=tool.behavior.requires_confirmation,
                    typical_latency_ms=tool.behavior.typical_latency_ms,
                    side_effects=tool.behavior.side_effects,
                )
            )
        if tool.rate_limits:
            proto.rate_limits.CopyFrom(
                tools_pb2.RateLimits(
                    requests_per_minute=tool.rate_limits.requests_per_minute,
                    requests_per_session=tool.rate_limits.requests_per_session,
                    daily_limit=tool.rate_limits.daily_limit,
                )
            )
        if tool.cost:
            proto.cost.CopyFrom(
                tools_pb2.CostMetadata(
                    estimated_cost_usd=tool.cost.estimated_cost_usd,
                    billing_category=tool.cost.billing_category,
                )
            )
        if tool.execution_limits:
            proto.execution_limits.CopyFrom(
                tools_pb2.ExecutionLimits(
                    timeout_seconds=tool.execution_limits.timeout_seconds,
                    memory_limit_mb=tool.execution_limits.memory_limit_mb,
                    cpu_limit_millicores=tool.execution_limits.cpu_limit_millicores,
                    max_response_size_kb=tool.execution_limits.max_response_size_kb,
                    max_retries=tool.execution_limits.max_retries,
                )
            )
        return proto

    # --- RPC implementations ---

    def RegisterTool(self, request, context):
        try:
            tool = self._proto_to_domain(request.tool)
            status = self.registry.register(tool)
            return tools_pb2.RegisterToolResponse(
                name=tool.name, version=tool.version, status=status
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return tools_pb2.RegisterToolResponse()

    def DiscoverTools(self, request, context):
        try:
            tools = self.registry.discover(
                namespace=request.namespace or None,
                capabilities=list(request.capabilities) or None,
                tags=list(request.tags) or None,
                read_only=request.read_only,
                version_constraint=request.version_constraint or None,
            )
            proto_tools = [self._domain_to_proto(t) for t in tools]
            return tools_pb2.DiscoverToolsResponse(tools=proto_tools)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return tools_pb2.DiscoverToolsResponse()

    def ExecuteTool(self, request, context):
        try:
            tool_name = request.tool_name
            if not self.circuit_breaker.allow_request(tool_name):
                return tools_pb2.ExecuteToolResponse(
                    success=False,
                    error=f"Circuit breaker open for '{tool_name}'",
                )

            tool = self.registry.get(tool_name)
            if not tool:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Tool '{tool_name}' not found")
                return tools_pb2.ExecuteToolResponse(success=False, error="Tool not found")

            args = json.loads(request.arguments_json) if request.arguments_json else {}
            start = time.time()
            result = self._execute_sandboxed(tool, args)
            elapsed_ms = int((time.time() - start) * 1000)

            self.circuit_breaker.record_result(tool_name, result.success)

            return tools_pb2.ExecuteToolResponse(
                success=result.success,
                result_json=json.dumps(result.result) if result.result else "",
                error=result.error or "",
                execution_time_ms=elapsed_ms,
            )
        except Exception as e:
            self.circuit_breaker.record_result(request.tool_name, False)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return tools_pb2.ExecuteToolResponse(success=False, error=str(e))

    def ValidateTool(self, request, context):
        try:
            tool = self.registry.get(request.tool_name)
            if not tool:
                return tools_pb2.ValidateToolResponse(
                    valid=False, errors=[f"Tool '{request.tool_name}' not found"]
                )
            errors = []
            if request.arguments_json:
                args = json.loads(request.arguments_json)
                if tool.parameters:
                    for required_key in tool.parameters.get("required", []):
                        if required_key not in args:
                            errors.append(f"Missing required parameter: {required_key}")
            return tools_pb2.ValidateToolResponse(valid=len(errors) == 0, errors=errors)
        except json.JSONDecodeError as e:
            return tools_pb2.ValidateToolResponse(valid=False, errors=[f"Invalid JSON: {e}"])
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return tools_pb2.ValidateToolResponse(valid=False, errors=[str(e)])

    def _execute_sandboxed(self, tool: ToolDefinition, args: dict) -> ToolExecutionResult:
        """
        Sandboxed tool execution stub (Listing 6.12).

        In production this would run in an isolated container/subprocess.
        Here we return a placeholder result.
        """
        logger.info("Executing tool %s with args %s", tool.name, args)
        return ToolExecutionResult(
            tool_name=tool.name,
            success=True,
            result={"status": "executed", "tool": tool.name, "args": args},
        )
