"""
Quick-start: registering a public MCP server with the Tool Service.

Connects the platform to DeepWiki's public streamable-HTTP MCP server —
https://mcp.deepwiki.com/mcp — which is maintained by Cognition (the team
behind Devin). DeepWiki exposes read-only tools that answer questions
about GitHub repositories; no credentials are required.

What this demonstrates:
  1. Tool Service connects as an MCP client (Listing 6.18)
  2. DeepWiki's tools land in the platform registry under a platform
     namespace ("docs.deepwiki"), just like any native tool
  3. Applications discover and execute MCP-backed tools through the same
     SDK surface (platform.tools.discover / platform.tools.execute)
  4. Real outbound MCP call returns real content

Expected runtime: ~5–15s (one live call to DeepWiki's ask_question tool).
Stop with Ctrl+C.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from genai_platform import GenAIPlatform
from services.gateway.main import main as start_gateway
from services.shared.server import run_aio_service_main
from services.tools.service import ToolServiceImpl

MCP_SERVER_URL = "https://mcp.deepwiki.com/mcp"
NAMESPACE = "docs.deepwiki"
TARGET_REPO = "modelcontextprotocol/python-sdk"


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def start_service_in_thread(target, name):
    t = threading.Thread(target=target, daemon=True, name=name)
    t.start()
    return t


def start_tools():
    run_aio_service_main("tools", ToolServiceImpl)


def main():
    print("=" * 60)
    print("  MCP Quick-Start: platform + DeepWiki (public MCP server)")
    print("=" * 60)

    print("\nStarting services...")
    start_service_in_thread(start_tools, "ToolService")
    time.sleep(1)
    start_service_in_thread(start_gateway, "Gateway")
    time.sleep(1)
    print("Services ready.\n")

    platform = GenAIPlatform()

    # --------------------------------------------------------------
    # 1. Register DeepWiki's MCP server
    # --------------------------------------------------------------
    section("1. Register DeepWiki as an MCP server")
    print(f"  server_url = {MCP_SERVER_URL}")
    print(f"  namespace  = {NAMESPACE}")
    imported = platform.tools.register_mcp_server(
        server_url=MCP_SERVER_URL,
        namespace=NAMESPACE,
    )
    print(f"\n  Imported {len(imported)} tools from DeepWiki into the registry:")
    for name in imported:
        print(f"    - {name}")

    # --------------------------------------------------------------
    # 2. Discover the imported tools through the standard SDK
    # --------------------------------------------------------------
    section("2. Discover MCP-backed tools")
    tools = platform.tools.discover(namespace=f"{NAMESPACE}.*")
    print(f"  platform.tools.discover(namespace='{NAMESPACE}.*') -> {len(tools)} tools")
    for t in tools:
        params = list((t.parameters or {}).get("properties", {}).keys())
        desc = (t.description or "").split("\n")[0][:70]
        print(f"    - {t.name}")
        print(f"      params: {params}")
        print(f"      desc:   {desc}")

    # --------------------------------------------------------------
    # 3. Execute a real MCP tool against DeepWiki
    # --------------------------------------------------------------
    section("3. Execute docs.deepwiki.ask_question against a real repo")
    question = f"What is the project at {TARGET_REPO}, in one paragraph?"
    print(f"  repoName: {TARGET_REPO}")
    print(f"  question: {question}")
    print("\n  ... calling DeepWiki (live network request) ...")

    result = platform.tools.execute(
        tool_name=f"{NAMESPACE}.ask_question",
        arguments={"repoName": TARGET_REPO, "question": question},
    )

    print(f"\n  success: {result.success}")
    print(f"  time_ms: {result.execution_time_ms}")
    if not result.success:
        print(f"  error:   {result.error}")
        return

    answer_blob = result.result or {}
    # DeepWiki returns {"result": "<markdown answer>"} (unstructured text
    # we wrapped into a dict). Show the first few lines so the user sees
    # this is a real answer, not a stub.
    raw = answer_blob.get("result") if isinstance(answer_blob, dict) else answer_blob
    if isinstance(raw, str):
        print("\n  DeepWiki answered:")
        for line in raw.splitlines()[:12]:
            print(f"    {line}")
        if len(raw.splitlines()) > 12:
            print("    ... (truncated)")
    else:
        print(f"\n  Raw payload: {answer_blob}")

    # --------------------------------------------------------------
    # 4. Execute a different MCP tool (read_wiki_structure)
    # --------------------------------------------------------------
    section("4. Execute docs.deepwiki.read_wiki_structure for table-of-contents")
    result2 = platform.tools.execute(
        tool_name=f"{NAMESPACE}.read_wiki_structure",
        arguments={"repoName": TARGET_REPO},
    )
    print(f"  success: {result2.success}")
    print(f"  time_ms: {result2.execution_time_ms}")
    if result2.success:
        raw2 = result2.result.get("result") if isinstance(result2.result, dict) else result2.result
        preview = str(raw2).splitlines()[:10] if isinstance(raw2, str) else [str(raw2)[:300]]
        print("\n  DeepWiki responded with the repo's wiki structure:")
        for line in preview:
            print(f"    {line}")

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)
    print(
        "\nYou just saw real MCP: the Tool Service connected to a public\n"
        "MCP server, imported its tools into the platform registry, and\n"
        "routed a real execute call through the MCP transport.\n"
    )


if __name__ == "__main__":
    main()
