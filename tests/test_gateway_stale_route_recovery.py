"""Tests for the gateway's stale-route recovery path.

When a workflow's container goes away outside of `genai-platform deploy`
(manual `docker rm`, OOM, daemon restart, ...), the gateway's local
cache and the Workflow Service's `_routes` are both stale. The recovery
flow:

  external curl → gateway forwards → httpx.ConnectError →
    gateway calls WorkflowService.ListRoutes (which prunes dead routes) →
    refresh local cache → either retry with fresh address or return clearer 404.

Two layers of coverage here:

  - Unit-level: ``refresh_routes_from_workflow_service`` against a real
    in-process WorkflowServiceImpl behind a real grpc.Server, with the
    health-check stubbed so we control which routes survive pruning.
  - Integration-level: a small fake "workflow container" HTTPServer the
    gateway forwards to. Tests kill the upstream HTTPServer, then issue
    a request and assert the gateway returns the recovery 404 instead
    of a 502 ``Connection refused``.
"""

import http.server
import json
import socket
import threading
from concurrent import futures

import grpc
import httpx
import pytest

from proto import workflow_pb2, workflow_pb2_grpc
from services.gateway import http_handler as gateway_http
from services.gateway.http_handler import (
    WorkflowHTTPHandler,
    refresh_routes_from_workflow_service,
)
from services.gateway.registry import ServiceRegistry
from services.workflow.service import WorkflowServiceImpl


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


# ---- Unit: refresh_routes_from_workflow_service ----------------------------


class _StubGrpcWorkflowService:
    """Minimal grpc.Server harness around a WorkflowServiceImpl."""

    def __init__(self, *, all_alive: bool):
        self.servicer = WorkflowServiceImpl(route_health_check=lambda _: all_alive)
        self.port = _free_port()
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
        workflow_pb2_grpc.add_WorkflowServiceServicer_to_server(self.servicer, self.server)
        self.server.add_insecure_port(f"[::]:{self.port}")

    def __enter__(self):
        self.server.start()
        return self

    def __exit__(self, *args):
        self.server.stop(0)


def test_refresh_replaces_local_cache_with_live_routes():
    """ListRoutes returns alive routes; the gateway's cache adopts them."""
    with _StubGrpcWorkflowService(all_alive=True) as svc:
        # Register some routes on the (real, in-process) Workflow Service.
        ctx = type("Ctx", (), {"set_code": lambda *_: None, "set_details": lambda *_: None})()
        svc.servicer.RegisterRoute(
            workflow_pb2.RegisterRouteRequest(api_path="/a", endpoint="host-a:1"), ctx
        )
        svc.servicer.RegisterRoute(
            workflow_pb2.RegisterRouteRequest(api_path="/b", endpoint="host-b:2"), ctx
        )

        # Gateway has stale data.
        registry = ServiceRegistry()
        registry.register_workflow("/a", "stale-a:99")
        registry.register_workflow("/old", "removed:99")

        refresh_routes_from_workflow_service(registry, f"localhost:{svc.port}")

        # Local cache replaced with the Workflow Service's view.
        assert registry._workflows == {"/a": ["host-a:1"], "/b": ["host-b:2"]}


def test_refresh_drops_dead_routes_due_to_prune():
    """When the Workflow Service prunes a dead route, the gateway also drops it."""
    with _StubGrpcWorkflowService(all_alive=False) as svc:
        ctx = type("Ctx", (), {"set_code": lambda *_: None, "set_details": lambda *_: None})()
        svc.servicer.RegisterRoute(
            workflow_pb2.RegisterRouteRequest(api_path="/dead", endpoint="dead:1"), ctx
        )

        registry = ServiceRegistry()
        registry.register_workflow("/dead", "dead:1")

        refresh_routes_from_workflow_service(registry, f"localhost:{svc.port}")

        assert registry._workflows == {}


def test_refresh_soft_fails_when_workflow_service_unreachable():
    """A down Workflow Service should not crash the gateway."""
    registry = ServiceRegistry()
    registry.register_workflow("/x", "host:1")
    # Address that nothing is listening on; refresh should log and return.
    refresh_routes_from_workflow_service(registry, "127.0.0.1:1")
    # Cache is left intact when refresh fails — no worse than not refreshing.
    assert registry._workflows == {"/x": ["host:1"]}


# ---- Integration: full do_POST recovery path -------------------------------


class _FakeContainer:
    """Tiny HTTP server the gateway forwards to. Can be alive() or stopped."""

    def __init__(self):
        self.port = _free_port()

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/health/ready":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"status":"ok"}')
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                payload = json.dumps({"answered": self.path}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args, **kwargs):
                pass

        self._server = http.server.HTTPServer(("127.0.0.1", self.port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._server.shutdown()
        self._server.server_close()


def _gateway_thread(handler_factory, port: int):
    server = http.server.HTTPServer(("127.0.0.1", port), handler_factory)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


@pytest.fixture
def fresh_http_client():
    """Reset the module-level _HTTP_CLIENT so each test gets a fresh one."""
    gateway_http._HTTP_CLIENT = None
    yield
    gateway_http._HTTP_CLIENT = None


def test_gateway_returns_clear_404_when_upstream_dead_and_no_replacement(
    monkeypatch, fresh_http_client
):
    """Container is dead, Workflow Service's pruned ListRoutes is empty → clearer 404."""
    container = _FakeContainer()
    container.start()

    with _StubGrpcWorkflowService(all_alive=False) as svc:
        ctx = type("Ctx", (), {"set_code": lambda *_: None, "set_details": lambda *_: None})()
        svc.servicer.RegisterRoute(
            workflow_pb2.RegisterRouteRequest(
                api_path="/echo", endpoint=f"127.0.0.1:{container.port}"
            ),
            ctx,
        )
        monkeypatch.setenv("WORKFLOW_SERVICE_ADDR", f"localhost:{svc.port}")

        registry = ServiceRegistry()
        # Stale route: gateway's cache points at the (now-stopped) container.
        registry.register_workflow("/echo", f"127.0.0.1:{container.port}")
        container.stop()  # ← container dies before any request lands

        gw_port = _free_port()
        gw_server, _ = _gateway_thread(
            lambda *a, **kw: WorkflowHTTPHandler(registry, *a, **kw), gw_port
        )
        try:
            r = httpx.post(f"http://localhost:{gw_port}/echo", json={"q": "x"}, timeout=5)
            assert r.status_code == 404
            assert "no longer reachable" in r.json()["error"]
            assert "redeploy" in r.json()["error"]
        finally:
            gw_server.shutdown()


def test_gateway_retries_with_fresh_endpoint_when_workflow_service_has_one(
    monkeypatch, fresh_http_client
):
    """Stale endpoint dies, Workflow Service knows the new one → gateway retries and succeeds."""
    fresh_container = _FakeContainer()
    fresh_container.start()

    with _StubGrpcWorkflowService(all_alive=True) as svc:
        ctx = type("Ctx", (), {"set_code": lambda *_: None, "set_details": lambda *_: None})()
        # Workflow Service has the FRESH route (post-redeploy).
        svc.servicer.RegisterRoute(
            workflow_pb2.RegisterRouteRequest(
                api_path="/echo", endpoint=f"127.0.0.1:{fresh_container.port}"
            ),
            ctx,
        )
        monkeypatch.setenv("WORKFLOW_SERVICE_ADDR", f"localhost:{svc.port}")

        registry = ServiceRegistry()
        # Gateway's cache points at a STALE port.
        stale_port = _free_port()  # nothing listening here
        registry.register_workflow("/echo", f"127.0.0.1:{stale_port}")

        gw_port = _free_port()
        gw_server, _ = _gateway_thread(
            lambda *a, **kw: WorkflowHTTPHandler(registry, *a, **kw), gw_port
        )
        try:
            r = httpx.post(f"http://localhost:{gw_port}/echo", json={"q": "x"}, timeout=5)
            assert r.status_code == 200, r.text
            assert r.json() == {"answered": "/echo"}
            # Cache was refreshed to the fresh endpoint.
            assert registry._workflows["/echo"] == [f"127.0.0.1:{fresh_container.port}"]
        finally:
            gw_server.shutdown()
            fresh_container.stop()
