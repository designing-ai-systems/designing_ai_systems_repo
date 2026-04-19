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

    Holds the transport and session open across calls via an AsyncExitStack,
    so subsequent list_tools / call_tool reuse the same initialized session.
    """

    def __init__(self) -> None:
        self._stack: Optional[AsyncExitStack] = None
        self._session: Optional[ClientSession] = None

    async def connect(self, server_url: str, auth_token: str = "") -> None:
        self._stack = AsyncExitStack()
        headers: Optional[dict[str, str]] = (
            {"Authorization": f"Bearer {auth_token}"} if auth_token else None
        )
        streams = await self._stack.enter_async_context(
            streamablehttp_client(server_url, headers=headers)
        )
        read, write, _ = streams
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()

    async def list_tools(self) -> list[types.Tool]:
        assert self._session is not None, "call connect() first"
        return (await self._session.list_tools()).tools

    async def call_tool(self, name: str, args: dict) -> types.CallToolResult:
        assert self._session is not None, "call connect() first"
        return await self._session.call_tool(name, args)

    async def close(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
            self._session = None


async def connect_streamable_http(server_url: str, auth_token: str = "") -> MCPClient:
    """Default connector used by ToolServiceImpl when no override is supplied."""
    client = StreamableHttpMCPClient()
    await client.connect(server_url, auth_token)
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
