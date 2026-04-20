"""
Tool Service client.

Manages external tool registration, discovery, and execution.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 6.4: Tool registration
  - Listing 6.7: Tool discovery
  - Listing 6.11: Tool deprecation / migration
  - Listing 6.12: Tool execution
"""

import json
from typing import Any, Dict, List, Optional, Tuple

from proto import tools_pb2, tools_pb2_grpc
from services.tools.model_sync import tool_definitions_to_model_tools
from services.tools.models import (
    CostMetadata,
    ExecutionLimits,
    RateLimits,
    ToolBehavior,
    ToolDefinition,
    ToolExecutionResult,
)

from .base import BaseClient


class ToolClient(BaseClient):
    """Client for Tool Service — registration, discovery, execution."""

    def __init__(self, platform):
        super().__init__(platform, "tools")
        self._stub = tools_pb2_grpc.ToolServiceStub(self._channel)

    # --- Registration (Listing 6.4) ---

    def register(
        self,
        name: str,
        description: str,
        version: str = "1.0.0",
        owner: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        returns_schema: Optional[Dict[str, Any]] = None,
        behavior: Optional[ToolBehavior] = None,
        rate_limits: Optional[RateLimits] = None,
        cost: Optional[CostMetadata] = None,
        required_permissions: Optional[List[str]] = None,
        capabilities: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        endpoint: str = "",
        credential_ref: str = "",
        execution_limits: Optional[ExecutionLimits] = None,
    ) -> Dict[str, str]:
        proto_tool = tools_pb2.ToolDefinition(
            name=name,
            version=version,
            owner=owner,
            description=description,
            parameters_json=json.dumps(parameters) if parameters else "",
            returns_json=json.dumps(returns_schema) if returns_schema else "",
            required_permissions=required_permissions or [],
            capabilities=capabilities or [],
            tags=tags or [],
            endpoint=endpoint,
            credential_ref=credential_ref,
        )
        if behavior:
            proto_tool.behavior.CopyFrom(
                tools_pb2.ToolBehavior(
                    is_read_only=behavior.is_read_only,
                    is_idempotent=behavior.is_idempotent,
                    requires_confirmation=behavior.requires_confirmation,
                    typical_latency_ms=behavior.typical_latency_ms,
                    side_effects=behavior.side_effects,
                )
            )
        if rate_limits:
            proto_tool.rate_limits.CopyFrom(
                tools_pb2.RateLimits(
                    requests_per_minute=rate_limits.requests_per_minute,
                    requests_per_session=rate_limits.requests_per_session,
                    daily_limit=rate_limits.daily_limit,
                )
            )
        if cost:
            proto_tool.cost.CopyFrom(
                tools_pb2.CostMetadata(
                    estimated_cost_usd=cost.estimated_cost_usd,
                    billing_category=cost.billing_category,
                )
            )
        if execution_limits:
            proto_tool.execution_limits.CopyFrom(
                tools_pb2.ExecutionLimits(
                    timeout_seconds=execution_limits.timeout_seconds,
                    memory_limit_mb=execution_limits.memory_limit_mb,
                    cpu_limit_millicores=execution_limits.cpu_limit_millicores,
                    max_response_size_kb=execution_limits.max_response_size_kb,
                    max_retries=execution_limits.max_retries,
                )
            )

        request = tools_pb2.RegisterToolRequest(tool=proto_tool)
        response = self._stub.RegisterTool(request, metadata=self.metadata)
        return {"name": response.name, "version": response.version, "status": response.status}

    # --- Discovery (Listing 6.7) ---

    def discover(
        self,
        namespace: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        read_only: bool = False,
        version_constraint: Optional[str] = None,
    ) -> List[ToolDefinition]:
        request = tools_pb2.DiscoverToolsRequest(
            namespace=namespace or "",
            capabilities=capabilities or [],
            tags=tags or [],
            read_only=read_only,
            version_constraint=version_constraint or "",
        )
        response = self._stub.DiscoverTools(request, metadata=self.metadata)
        return [self._proto_to_domain(t) for t in response.tools]

    def build_model_tools(
        self,
        *,
        names: Optional[List[str]] = None,
        namespace: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        read_only: bool = False,
        version_constraint: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        """
        Discover tools from the registry and convert them to Model Service ``tools`` payloads.

        Uses the same filters as :meth:`discover`. If *names* is set, only those
        platform tool names are kept (must appear in discovery results).

        Returns:
            (model_tools, llm_to_platform) where *llm_to_platform* maps LLM function
            name -> Tool Service ``tool_name`` for :meth:`execute`.
        """
        definitions = self.discover(
            namespace=namespace,
            capabilities=capabilities,
            tags=tags,
            read_only=read_only,
            version_constraint=version_constraint,
        )
        if names is not None:
            wanted = set(names)
            definitions = [d for d in definitions if d.name in wanted]
        return tool_definitions_to_model_tools(definitions)

    # --- Execution (Listing 6.12) ---

    def execute(
        self, tool_name: str, arguments: Optional[Dict[str, Any]] = None, session_id: str = ""
    ) -> ToolExecutionResult:
        request = tools_pb2.ExecuteToolRequest(
            tool_name=tool_name,
            arguments_json=json.dumps(arguments) if arguments else "{}",
            session_id=session_id,
        )
        response = self._stub.ExecuteTool(request, metadata=self.metadata)
        return ToolExecutionResult(
            tool_name=tool_name,
            success=response.success,
            result=json.loads(response.result_json) if response.result_json else None,
            error=response.error or None,
            execution_time_ms=response.execution_time_ms,
        )

    # --- MCP registration (Listing 6.18) ---

    def register_mcp_server(
        self,
        server_url: str,
        namespace: str,
        credential_ref: str = "",
        policy_overrides: Optional[ToolBehavior] = None,
        rate_limit_overrides: Optional[RateLimits] = None,
    ) -> List[str]:
        """Connect to an MCP server and import its tools into the registry.

        Each imported tool is registered under ``<namespace>.<tool_name>``.
        Returns the list of fully qualified platform tool names that were
        imported so applications can discover them immediately.
        """
        request = tools_pb2.RegisterMcpServerRequest(
            server_url=server_url,
            namespace=namespace,
            credential_ref=credential_ref,
        )
        if policy_overrides:
            request.policy_overrides.CopyFrom(
                tools_pb2.ToolBehavior(
                    is_read_only=policy_overrides.is_read_only,
                    is_idempotent=policy_overrides.is_idempotent,
                    requires_confirmation=policy_overrides.requires_confirmation,
                    typical_latency_ms=policy_overrides.typical_latency_ms,
                    side_effects=policy_overrides.side_effects,
                )
            )
        if rate_limit_overrides:
            request.rate_limit_overrides.CopyFrom(
                tools_pb2.RateLimits(
                    requests_per_minute=rate_limit_overrides.requests_per_minute,
                    requests_per_session=rate_limit_overrides.requests_per_session,
                    daily_limit=rate_limit_overrides.daily_limit,
                )
            )
        response = self._stub.RegisterMcpServer(request, metadata=self.metadata)
        return list(response.imported_tool_names)

    # --- Async execution (Listing 6.16) ---

    def execute_async(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        session_id: str = "",
    ) -> Dict[str, str]:
        """Start a tool execution in the background; returns ``{"task_id", "status"}``.

        Poll :meth:`get_task` to observe state transitions
        (pending → running → succeeded | failed | timed_out).
        """
        request = tools_pb2.ExecuteToolRequest(
            tool_name=tool_name,
            arguments_json=json.dumps(arguments) if arguments else "{}",
            session_id=session_id,
        )
        response = self._stub.ExecuteToolAsync(request, metadata=self.metadata)
        return {"task_id": response.task_id, "status": response.status}

    def get_task(self, task_id: str) -> Dict[str, Any]:
        """Return the current state of an async task started by :meth:`execute_async`."""
        request = tools_pb2.GetTaskRequest(task_id=task_id)
        response = self._stub.GetTask(request, metadata=self.metadata)
        return {
            "task_id": response.task_id,
            "status": response.status,
            "result": json.loads(response.result_json) if response.result_json else None,
            "error": response.error or None,
        }

    # --- Validation ---

    def validate(
        self, tool_name: str, arguments: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        request = tools_pb2.ValidateToolRequest(
            tool_name=tool_name,
            arguments_json=json.dumps(arguments) if arguments else "{}",
        )
        response = self._stub.ValidateTool(request, metadata=self.metadata)
        return {"valid": response.valid, "errors": list(response.errors)}

    # --- helpers ---

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
            version=proto.version,
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
            mcp_server_url=proto.mcp_server_url,
        )
