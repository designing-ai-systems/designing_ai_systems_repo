"""
Tool Service — gRPC service implementation (grpc.aio).

Thin translation layer: receives proto requests, delegates to
ToolRegistry/CredentialStore/CircuitBreaker, returns proto responses.

Uses grpc.aio so RPC handlers can await Listing 6.14 (CredentialStore)
during credential injection before external execution.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 6.1: ToolService gRPC contract
  - Listing 6.4: Tool registration
  - Listing 6.7: Tool discovery
  - Listing 6.12: Sandboxed tool execution
  - Listing 6.14: Credential retrieval at execution time
  - Listing 6.18: CircuitBreaker integration
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Awaitable, Callable, Dict, Optional

import grpc
import grpc.aio
import httpx

from proto import tools_pb2, tools_pb2_grpc
from services.shared.servicer_base import BaseAioServicer
from services.tools.circuit_breaker import CircuitBreaker
from services.tools.credential_store import Credential, CredentialStore, InMemoryCredentialStore
from services.tools.mcp_client import MCPClient, connect_streamable_http, extract_text_payload
from services.tools.models import (
    CostMetadata,
    ExecutionLimits,
    RateLimits,
    ToolBehavior,
    ToolDefinition,
    ToolExecutionResult,
    ToolTask,
)
from services.tools.store import InMemoryToolRegistry, ToolRegistry

MCPConnector = Callable[[str, str], Awaitable[MCPClient]]

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_MAX_RESPONSE_SIZE_KB = 1024


class ToolServiceImpl(tools_pb2_grpc.ToolServiceServicer, BaseAioServicer):
    def __init__(
        self,
        registry: ToolRegistry | None = None,
        credential_store: CredentialStore | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        http_client: httpx.AsyncClient | None = None,
        mcp_connector: MCPConnector | None = None,
    ):
        self.registry = registry or InMemoryToolRegistry()
        self.credential_store = credential_store or InMemoryCredentialStore()
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self._http_client = http_client or httpx.AsyncClient()
        # Listing 6.16: async tasks. In-memory dict is adequate for a book
        # demo; production should swap for Redis/Celery/equivalent.
        self._tasks: Dict[str, ToolTask] = {}
        # Listing 6.18: one persistent MCP session per registered server,
        # shared across every ExecuteTool that resolves to a tool imported
        # from that server. Tests inject a connector that returns a fake.
        self._mcp_connector: MCPConnector = mcp_connector or connect_streamable_http
        self._mcp_clients: Dict[str, MCPClient] = {}

    def add_to_aio_server(self, server: grpc.aio.Server) -> None:
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
            mcp_server_url=proto.mcp_server_url,
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
            mcp_server_url=tool.mcp_server_url,
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

    # --- RPC implementations (async / grpc.aio) ---

    async def RegisterTool(self, request, context):
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

    async def DiscoverTools(self, request, context):
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

    async def ExecuteTool(self, request, context):
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
            credential: Optional[Credential] = None
            if tool.credential_ref:
                try:
                    credential = await self.credential_store.retrieve(
                        tool.credential_ref, tool_name
                    )
                except KeyError:
                    return tools_pb2.ExecuteToolResponse(
                        success=False,
                        error=f"Credential '{tool.credential_ref}' not found",
                    )
                except PermissionError as e:
                    return tools_pb2.ExecuteToolResponse(success=False, error=str(e))

            start = time.time()
            result = await self._execute_sandboxed(tool, args, credential)
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

    async def RegisterMcpServer(self, request, context):
        """Listing 6.18: connect to an external MCP server and import its tools.

        Each imported tool is registered under the caller's ``namespace`` with
        its name prefixed (``<namespace>.<tool_name>``). ``policy_overrides``
        and ``rate_limit_overrides`` are copied onto every imported tool so
        platform-level policies apply uniformly across that server.
        """
        server_url = request.server_url
        namespace = request.namespace

        auth_token = ""
        if request.credential_ref:
            try:
                cred = await self.credential_store.retrieve(
                    request.credential_ref, f"mcp://{server_url}"
                )
                auth_token = cred.value
            except (KeyError, PermissionError) as e:
                context.set_code(grpc.StatusCode.PERMISSION_DENIED)
                context.set_details(str(e))
                return tools_pb2.RegisterMcpServerResponse()

        try:
            client = await self._mcp_connector(server_url, auth_token)
            mcp_tools = await client.list_tools()
        except Exception as e:  # noqa: BLE001 — surface any connect/list failure uniformly
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details(f"MCP server unreachable: {e}")
            return tools_pb2.RegisterMcpServerResponse()

        self._mcp_clients[server_url] = client

        behavior_override = None
        if request.HasField("policy_overrides"):
            po = request.policy_overrides
            behavior_override = ToolBehavior(
                is_read_only=po.is_read_only,
                is_idempotent=po.is_idempotent,
                requires_confirmation=po.requires_confirmation,
                typical_latency_ms=po.typical_latency_ms,
                side_effects=list(po.side_effects),
            )
        rate_limits_override = None
        if request.HasField("rate_limit_overrides"):
            rl = request.rate_limit_overrides
            rate_limits_override = RateLimits(
                requests_per_minute=rl.requests_per_minute,
                requests_per_session=rl.requests_per_session,
                daily_limit=rl.daily_limit,
            )

        imported_names = []
        for t in mcp_tools:
            platform_name = f"{namespace}.{t.name}"
            tool_def = ToolDefinition(
                name=platform_name,
                description=t.description or "",
                parameters=t.inputSchema or None,
                behavior=behavior_override,
                rate_limits=rate_limits_override,
                mcp_server_url=server_url,
            )
            self.registry.register(tool_def)
            imported_names.append(platform_name)

        logger.info(
            "Registered %d MCP tools from %s under namespace %s",
            len(imported_names),
            server_url,
            namespace,
        )
        return tools_pb2.RegisterMcpServerResponse(imported_tool_names=imported_names)

    async def ExecuteToolAsync(self, request, context):
        """Listing 6.16 Option 1: returns a task_id immediately; callers poll GetTask."""
        tool_name = request.tool_name
        tool = self.registry.get(tool_name)
        if not tool:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Tool '{tool_name}' not found")
            return tools_pb2.ExecuteToolAsyncResponse()

        args = json.loads(request.arguments_json) if request.arguments_json else {}
        task_id = uuid.uuid4().hex
        task = ToolTask(id=task_id, status="pending", tool_name=tool_name, arguments=args)
        self._tasks[task_id] = task
        asyncio.create_task(self._run_async(task, tool))
        return tools_pb2.ExecuteToolAsyncResponse(task_id=task_id, status="pending")

    async def GetTask(self, request, context):
        task = self._tasks.get(request.task_id)
        if not task:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Task '{request.task_id}' not found")
            return tools_pb2.GetTaskResponse()

        result_json = ""
        error = ""
        if task.result:
            if task.result.result is not None:
                result_json = json.dumps(task.result.result)
            if task.result.error:
                error = task.result.error

        return tools_pb2.GetTaskResponse(
            task_id=task.id,
            status=task.status,
            result_json=result_json,
            error=error,
        )

    async def _run_async(self, task: ToolTask, tool: ToolDefinition) -> None:
        """Background coroutine that drives a single async task to completion."""
        task.status = "running"
        credential: Optional[Credential] = None
        if tool.credential_ref:
            try:
                credential = await self.credential_store.retrieve(
                    tool.credential_ref, tool.name
                )
            except (KeyError, PermissionError) as e:
                task.result = ToolExecutionResult(
                    tool_name=tool.name, success=False, error=str(e)
                )
                task.status = "failed"
                return

        result = await self._execute_sandboxed(tool, task.arguments or {}, credential)
        self.circuit_breaker.record_result(tool.name, result.success)
        task.result = result
        if result.success:
            task.status = "succeeded"
        elif result.error and "timeout" in result.error.lower():
            task.status = "timed_out"
        else:
            task.status = "failed"

    async def ValidateTool(self, request, context):
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

    async def _execute_sandboxed(
        self,
        tool: ToolDefinition,
        args: dict,
        credential: Optional[Credential],
    ) -> ToolExecutionResult:
        """
        Execute a tool by POSTing args to its endpoint (Listing 6.12).

        Injects ``credential`` as an auth header based on ``credential_type``
        (Listing 6.13–6.14). Enforces per-tool ``timeout_seconds`` and
        ``max_response_size_kb`` from ExecutionLimits (Listing 6.17);
        memory / cpu / retries are container-level concerns left to the
        deployment.

        When ``tool.mcp_server_url`` is set, routes through the MCP client
        registered for that server (Listing 6.18) instead of issuing HTTP.
        """
        if tool.mcp_server_url:
            return await self._execute_via_mcp(tool, args)

        if not tool.endpoint:
            return ToolExecutionResult(
                tool_name=tool.name,
                success=False,
                error=f"Tool '{tool.name}' has no endpoint configured",
            )

        headers = _auth_headers(credential)
        limits = tool.execution_limits or ExecutionLimits()
        timeout = limits.timeout_seconds or DEFAULT_TIMEOUT_SECONDS
        max_bytes = (limits.max_response_size_kb or DEFAULT_MAX_RESPONSE_SIZE_KB) * 1024

        logger.info(
            "Executing tool %s POST %s (credential=%s, timeout=%ds)",
            tool.name,
            tool.endpoint,
            credential.name if credential else None,
            timeout,
        )

        try:
            response = await self._http_client.post(
                tool.endpoint,
                json=args,
                headers=headers,
                timeout=timeout,
            )
        except httpx.TimeoutException as e:
            return ToolExecutionResult(
                tool_name=tool.name, success=False, error=f"timeout after {timeout}s: {e}"
            )
        except httpx.RequestError as e:
            return ToolExecutionResult(
                tool_name=tool.name, success=False, error=f"request failed: {e}"
            )

        body = response.content
        if len(body) > max_bytes:
            return ToolExecutionResult(
                tool_name=tool.name,
                success=False,
                error=f"response exceeds max_response_size_kb ({len(body)} > {max_bytes} bytes)",
            )

        if response.status_code >= 400:
            snippet = body.decode("utf-8", errors="replace")[:500]
            return ToolExecutionResult(
                tool_name=tool.name,
                success=False,
                error=f"HTTP {response.status_code}: {snippet}",
            )

        try:
            result = response.json()
        except json.JSONDecodeError:
            result = {"raw": body.decode("utf-8", errors="replace")}

        return ToolExecutionResult(tool_name=tool.name, success=True, result=result)


    async def _execute_via_mcp(
        self, tool: ToolDefinition, args: dict
    ) -> ToolExecutionResult:
        """Route execution through the MCP client that owns this tool (Listing 6.18)."""
        client = self._mcp_clients.get(tool.mcp_server_url)
        if client is None:
            return ToolExecutionResult(
                tool_name=tool.name,
                success=False,
                error=f"No MCP client registered for {tool.mcp_server_url}",
            )
        # The platform tool name is "<namespace>.<original>"; strip the namespace
        # so the MCP server sees the name it actually exposes.
        original_name = tool.name.rsplit(".", 1)[-1]
        try:
            result = await client.call_tool(original_name, args)
        except Exception as e:  # noqa: BLE001
            return ToolExecutionResult(
                tool_name=tool.name, success=False, error=f"MCP call failed: {e}"
            )
        if result.isError:
            text = extract_text_payload(result)
            return ToolExecutionResult(
                tool_name=tool.name, success=False, error=str(text) or "MCP tool error"
            )
        payload = extract_text_payload(result)
        if not isinstance(payload, dict):
            payload = {"result": payload}
        return ToolExecutionResult(tool_name=tool.name, success=True, result=payload)


def _auth_headers(credential: Optional[Credential]) -> dict:
    """Map credential_type → outbound auth header (Listing 6.14)."""
    if credential is None:
        return {}
    ctype = credential.credential_type
    if ctype == "api_key":
        return {"X-API-Key": credential.value}
    if ctype in ("bearer", "oauth2"):
        return {"Authorization": f"Bearer {credential.value}"}
    if ctype == "basic":
        return {"Authorization": f"Basic {credential.value}"}
    logger.warning("Unknown credential_type %r; passing as X-Credential header", ctype)
    return {"X-Credential": credential.value}
