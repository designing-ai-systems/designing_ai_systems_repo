"""
MCP client used by the Tool Service to wrap external MCP servers (Listing 6.18).

The Tool Service plays the role of MCP host-and-client (§6.4): it connects
once per registered server and shares that connection across every
application the platform serves. A real production deployment would likely
add reconnect logic, health checks, and per-server credential isolation —
this thin wrapper gives the book demo the minimum: connect, list tools,
call tools, close.

Transport: streamable HTTP (the transport the chapter's example uses). Other
transports (stdio, SSE) could slot in behind the same Protocol.
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from typing import Any, Optional, Protocol, runtime_checkable

from mcp import types
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client


@runtime_checkable
class MCPClient(Protocol):
    """Protocol the Tool Service calls into. Fakes in tests implement the same shape."""

    async def list_tools(self) -> list[types.Tool]: ...

    async def call_tool(self, name: str, args: dict) -> types.CallToolResult: ...

    async def close(self) -> None: ...


class StreamableHttpMCPClient:
    """Real MCPClient over streamable HTTP.

    Each list_tools / call_tool opens a fresh session, runs the RPC, and
    closes cleanly — simpler than pinning one long-lived session across
    multiple asyncio tasks (which anyio's task-group cancel scopes reject).
    The per-call init cost is a single POST and acceptable for a book demo;
    production would add connection pooling if it mattered.
    """

    def __init__(self, server_url: str, auth_token: str = "") -> None:
        self._server_url = server_url
        self._headers: Optional[dict[str, str]] = (
            {"Authorization": f"Bearer {auth_token}"} if auth_token else None
        )

    async def _open(self, stack: AsyncExitStack) -> ClientSession:
        streams = await stack.enter_async_context(
            streamablehttp_client(self._server_url, headers=self._headers)
        )
        read, write, _ = streams
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        return session

    async def list_tools(self) -> list[types.Tool]:
        async with AsyncExitStack() as stack:
            session = await self._open(stack)
            return (await session.list_tools()).tools

    async def call_tool(self, name: str, args: dict) -> types.CallToolResult:
        async with AsyncExitStack() as stack:
            session = await self._open(stack)
            return await session.call_tool(name, args)

    async def close(self) -> None:
        # No persistent state to release.
        return None


async def connect_streamable_http(server_url: str, auth_token: str = "") -> MCPClient:
    """Default connector used by ToolServiceImpl when no override is supplied.

    Performs one handshake to surface DNS/TLS errors eagerly (tests rely on
    register-time failures to report UNAVAILABLE).
    """
    client = StreamableHttpMCPClient(server_url, auth_token)
    # Eager health-check: a successful list_tools() proves reachability and auth.
    await client.list_tools()
    return client


def extract_text_payload(result: types.CallToolResult) -> Any:
    """Convert an MCP CallToolResult into a plain JSON-serializable payload.

    Prefers ``structuredContent`` when present (the modern MCP way to carry
    typed data). Falls back to concatenated text blocks, parsed as JSON when
    possible so callers get a dict/list instead of a string-wrapped blob.
    """
    if result.structuredContent is not None:
        return result.structuredContent
    texts = [c.text for c in result.content if isinstance(c, types.TextContent)]
    joined = "".join(texts)
    if not joined:
        return None
    import json as _json

    try:
        return _json.loads(joined)
    except _json.JSONDecodeError:
        return joined
