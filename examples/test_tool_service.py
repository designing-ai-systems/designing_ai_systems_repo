"""
Comprehensive Tool Service test.

Exercises every operation the Tool Service exposes, mostly against a
locally-run mock API so you can see real HTTP calls, real credential
injection, and the circuit breaker tripping in real time. No external
credentials required.

Covers:
  - Registration with operational metadata (Listing 6.2, 6.3)
  - Discovery by namespace, capability, tags, read_only (Listing 6.7)
  - Validation of tool arguments against the registered schema
  - Synchronous HTTP execution with credential injection (Listings 6.12–6.14)
  - Asynchronous execution with task polling (Listing 6.16)
  - Circuit breaker tripping under sustained failure (Listing 6.18)

If ``--mcp`` is passed, also registers the public DeepWiki MCP server and
runs one live execute. Default run stays offline.

Run:  python examples/test_tool_service.py [--mcp]
"""

from __future__ import annotations

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio

from genai_platform import GenAIPlatform
from services.gateway.main import main as start_gateway
from services.shared.server import create_grpc_aio_server, get_service_port
from services.tools.credential_store import InMemoryCredentialStore
from services.tools.models import CostMetadata, RateLimits, ToolBehavior
from services.tools.service import ToolServiceImpl

MOCK_PORT = 8766
MOCK_BASE = f"http://127.0.0.1:{MOCK_PORT}"
BOOKING_PATH = "/bookings"
FLAKY_PATH = "/flaky"  # always 500 — used to demo the circuit breaker

# State the mock API reports back so the demo can prove credential injection.
_calls: list[dict] = []


def print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


# --------------------------------------------------------------------
# Mock scheduling API (in-process) — hit by the Tool Service via httpx
# --------------------------------------------------------------------
def start_mock_api():
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status, payload):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length).decode() if length else ""
            args = json.loads(raw) if raw else {}
            _calls.append(
                {
                    "path": self.path,
                    "args": args,
                    "x_api_key": self.headers.get("X-API-Key"),
                    "authorization": self.headers.get("Authorization"),
                }
            )
            if self.path.endswith(FLAKY_PATH):
                self._reply(500, {"error": "simulated downstream outage"})
            elif self.path.endswith(BOOKING_PATH):
                self._reply(
                    200,
                    {
                        "appointment_id": f"apt-{int(time.time() * 1000) % 10_000}",
                        "status": "booked",
                        "args_received": args,
                        "auth_header_present": bool(self.headers.get("X-API-Key")),
                    },
                )
            else:
                self._reply(404, {"error": "unknown path"})

        def log_message(self, *_a, **_kw):
            pass

    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer(("127.0.0.1", MOCK_PORT), Handler)
    server.serve_forever()


# --------------------------------------------------------------------
# Tool Service lifecycle — started in-process with pre-seeded credentials
# --------------------------------------------------------------------
async def _tools_main():
    cred_store = InMemoryCredentialStore()
    await cred_store.store(
        name="scheduling-api-prod",
        credential_type="api_key",
        value="demo-api-key-REDACTED",
        allowed_tools=["healthcare.scheduling.*"],
    )
    port = get_service_port("tools")
    svc = ToolServiceImpl(credential_store=cred_store)
    server = create_grpc_aio_server(servicer=svc, port=port, service_name="tools")
    print(f"Starting tools Service (grpc.aio) on port {port}")
    await server.start()
    await server.wait_for_termination()


def start_tools():
    try:
        asyncio.run(_tools_main())
    except KeyboardInterrupt:
        pass


def start_service_in_thread(target, name):
    threading.Thread(target=target, daemon=True, name=name).start()


# --------------------------------------------------------------------
# Test sections
# --------------------------------------------------------------------
def test_registration(platform):
    print_section("TEST 1: Registration (Listing 6.2, 6.3)")
    result = platform.tools.register(
        name="healthcare.scheduling.book_appointment",
        description="Book an appointment for a patient",
        version="1.0.0",
        owner="scheduling-team",
        parameters={
            "type": "object",
            "required": ["patient_id", "slot_id"],
            "properties": {
                "patient_id": {"type": "string"},
                "slot_id": {"type": "string"},
            },
        },
        behavior=ToolBehavior(
            is_read_only=False,
            is_idempotent=False,
            requires_confirmation=True,
            typical_latency_ms=400,
            side_effects=["calendar_update", "email_notification"],
        ),
        rate_limits=RateLimits(requests_per_minute=60, requests_per_session=3),
        cost=CostMetadata(estimated_cost_usd=0.02, billing_category="scheduling"),
        capabilities=["scheduling", "booking"],
        tags=["patient-facing", "hipaa-compliant"],
        endpoint=f"{MOCK_BASE}{BOOKING_PATH}",
        credential_ref="scheduling-api-prod",
    )
    print(f"  ✓ Registered: {result['name']} v{result['version']} ({result['status']})")

    platform.tools.register(
        name="healthcare.scheduling.check_availability",
        description="Check provider availability",
        version="1.0.0",
        owner="scheduling-team",
        behavior=ToolBehavior(is_read_only=True, is_idempotent=True),
        capabilities=["scheduling", "availability"],
        tags=["patient-facing", "read-only"],
        endpoint=f"{MOCK_BASE}{BOOKING_PATH}",
    )
    print("  ✓ Registered: healthcare.scheduling.check_availability (read-only)")

    platform.tools.register(
        name="flaky.tool",
        description="Always returns 500 — used to trip the circuit breaker",
        owner="demo",
        endpoint=f"{MOCK_BASE}{FLAKY_PATH}",
    )
    print("  ✓ Registered: flaky.tool (always 500)")


def test_discovery(platform):
    print_section("TEST 2: Discovery (Listing 6.7)")

    tools = platform.tools.discover()
    print(f"  All tools: {len(tools)}")
    for t in tools:
        print(
            f"    - {t.name}  [caps={t.capabilities}  ro={bool(t.behavior and t.behavior.is_read_only)}]"
        )

    scheduling = platform.tools.discover(namespace="healthcare.scheduling.*")
    print(f"\n  discover(namespace='healthcare.scheduling.*') -> {len(scheduling)}")

    by_capability = platform.tools.discover(capabilities=["booking"])
    print(f"  discover(capabilities=['booking']) -> {len(by_capability)}")

    ro = platform.tools.discover(read_only=True)
    print(f"  discover(read_only=True) -> {len(ro)}")
    for t in ro:
        print(f"    - {t.name}")


def test_validation(platform):
    print_section("TEST 3: Validation")
    ok = platform.tools.validate(
        tool_name="healthcare.scheduling.book_appointment",
        arguments={"patient_id": "p1", "slot_id": "s42"},
    )
    print(f"  valid args → valid={ok['valid']}  errors={ok['errors']}")

    bad = platform.tools.validate(
        tool_name="healthcare.scheduling.book_appointment",
        arguments={"patient_id": "p1"},  # missing slot_id
    )
    print(f"  missing slot_id → valid={bad['valid']}  errors={bad['errors']}")


def test_sync_execution(platform):
    print_section("TEST 4: Sync execution with credential injection (Listing 6.12–6.14)")

    _calls.clear()
    result = platform.tools.execute(
        tool_name="healthcare.scheduling.book_appointment",
        arguments={"patient_id": "patient-123", "slot_id": "slot-2026-04-20-10am"},
    )
    print(f"  execute success={result.success}  time_ms={result.execution_time_ms}")
    print(f"  result: {result.result}")

    assert _calls, "Mock API should have seen at least one request"
    last = _calls[-1]
    print("\n  What the mock scheduling API actually saw on the wire:")
    print(f"    path        : {last['path']}")
    print(f"    args        : {last['args']}")
    print(f"    X-API-Key   : {last['x_api_key'] or '<missing>'}")
    print(f"    Authorization: {last['authorization'] or '<missing>'}")
    assert last["x_api_key"] == "demo-api-key-REDACTED", "Credential was not injected"
    print("\n  ✓ Tool Service attached the credential — the app never touched the secret")


def test_async_execution(platform):
    print_section("TEST 5: Async execution with polling (Listing 6.16)")
    started = platform.tools.execute_async(
        tool_name="healthcare.scheduling.book_appointment",
        arguments={"patient_id": "patient-456", "slot_id": "slot-async"},
    )
    print(f"  started: task_id={started['task_id']}  status={started['status']}")

    last_status = None
    for _ in range(60):
        task = platform.tools.get_task(started["task_id"])
        if task["status"] != last_status:
            print(f"    poll → {task['status']}")
            last_status = task["status"]
        if task["status"] not in ("pending", "running"):
            break
        time.sleep(0.05)

    print(f"\n  final: {task['status']}")
    print(f"  result: {task['result']}")


def test_circuit_breaker(platform):
    print_section("TEST 6: Circuit breaker trips under sustained failure (Listing 6.18)")
    print("  flaky.tool always returns 500; default breaker opens after 5 consecutive failures.")
    print("  Each call below is a real HTTP attempt until the breaker opens.\n")
    for i in range(1, 8):
        r = platform.tools.execute(tool_name="flaky.tool")
        label = (
            "FAIL (HTTP)"
            if r.error and "500" in r.error
            else ("OPEN " if "Circuit" in (r.error or "") else "OTHER")
        )
        err = (r.error or "").split(":", 1)[0]
        print(f"  attempt {i:>2}: success={r.success:<5}  [{label}] {err}")


def test_mcp_deepwiki(platform):
    print_section("TEST 7 (optional): MCP against DeepWiki — LIVE network call")
    server_url = "https://mcp.deepwiki.com/mcp"
    namespace = "docs.deepwiki"
    print(f"  register_mcp_server(server_url='{server_url}', namespace='{namespace}')")
    imported = platform.tools.register_mcp_server(server_url=server_url, namespace=namespace)
    print(f"  imported: {imported}")

    print("\n  Asking DeepWiki: 'What is the MCP Python SDK?'")
    result = platform.tools.execute(
        tool_name=f"{namespace}.ask_question",
        arguments={
            "repoName": "modelcontextprotocol/python-sdk",
            "question": "What is the MCP Python SDK in one sentence?",
        },
    )
    print(f"  success: {result.success}  time_ms: {result.execution_time_ms}")
    if result.success:
        raw = result.result.get("result") if isinstance(result.result, dict) else result.result
        first_para = (raw or "").split("\n\n", 1)[0] if isinstance(raw, str) else str(raw)[:400]
        print(f"  answer: {first_para[:500]}")


def main():
    do_mcp = "--mcp" in sys.argv

    print("=" * 60)
    print("  Tool Service Comprehensive Test")
    print("=" * 60)
    print("\nStarting mock scheduling API, Tool Service, and Gateway...")
    start_service_in_thread(start_mock_api, "MockAPI")
    start_service_in_thread(start_tools, "ToolService")
    time.sleep(1)
    start_service_in_thread(start_gateway, "Gateway")
    time.sleep(1)
    print("Services ready.\n")

    platform = GenAIPlatform()
    try:
        test_registration(platform)
        test_discovery(platform)
        test_validation(platform)
        test_sync_execution(platform)
        test_async_execution(platform)
        test_circuit_breaker(platform)
        if do_mcp:
            test_mcp_deepwiki(platform)
        else:
            print_section("TEST 7: MCP demo (skipped — pass --mcp to enable)")
            print("  Adds a real live call to https://mcp.deepwiki.com/mcp.")

        print("\n" + "=" * 60)
        print("  ✓ All Tool Service tests completed")
        print("=" * 60)
    except Exception as e:  # noqa: BLE001
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
