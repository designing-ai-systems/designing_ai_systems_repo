"""
Gateway HTTP handler — forwards external requests to workflow containers.

Three response shapes are supported, picked by the workflow's runtime
server based on its ``response_mode`` (Listings 8.4 / 8.6 / 8.8):

- **sync** — buffered JSON pass-through.
- **stream** — Server-Sent Events; the gateway writes each upstream chunk
  to ``wfile`` as soon as it arrives so progressive tokens reach the
  external client without buffering.
- **async** — 202 Accepted with ``{job_id, status_url}``; client polls
  ``GET /jobs/{job_id}`` (handled here by proxying to
  ``WorkflowService.GetJobStatus`` over gRPC).

The gateway's routing table (``registry._workflows``) is populated by the
Workflow Service via ``RegisterRoute`` (commit 3) and re-hydrated on
startup via ``WorkflowService.ListRoutes`` so that gateway restarts are
non-disruptive.
"""

import json
import logging
import os
from http.server import BaseHTTPRequestHandler

import grpc
import httpx

from proto import workflow_pb2, workflow_pb2_grpc
from services.gateway.registry import ServiceRegistry

logger = logging.getLogger(__name__)


# Forwarding HTTP client is shared across requests; HTTPServer re-instantiates
# WorkflowHTTPHandler per request so the client must outlive the handler.
_HTTP_CLIENT: httpx.Client | None = None


def _get_http_client() -> httpx.Client:
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None:
        _HTTP_CLIENT = httpx.Client(timeout=httpx.Timeout(30.0, read=None))
    return _HTTP_CLIENT


def _workflow_service_addr() -> str:
    return os.getenv("WORKFLOW_SERVICE_ADDR", "localhost:50058")


def _job_id_from_path(path: str) -> str | None:
    """Return the trailing job id from ``/jobs/{id}``, or None if not a jobs path."""
    if not path.startswith("/jobs/"):
        return None
    rest = path[len("/jobs/") :]
    return rest.split("?", 1)[0].rstrip("/") or None


class WorkflowHTTPHandler(BaseHTTPRequestHandler):
    """Per-request HTTP handler. Stateless; reads from the shared `registry`."""

    def __init__(self, registry: ServiceRegistry, *args, **kwargs):
        self.registry = registry
        super().__init__(*args, **kwargs)

    # ---- GET /jobs/{id} → WorkflowService.GetJobStatus -------------------

    def do_GET(self) -> None:
        job_id = _job_id_from_path(self.path)
        if job_id is None:
            self._send_json({"error": "not found"}, status=404)
            return
        try:
            channel = grpc.insecure_channel(_workflow_service_addr())
            stub = workflow_pb2_grpc.WorkflowServiceStub(channel)
            resp = stub.GetJobStatus(workflow_pb2.GetJobStatusRequest(job_id=job_id))
            j = resp.job
            payload = {
                "job": {
                    "job_id": j.job_id,
                    "workflow_id": j.workflow_id,
                    "status": j.status,
                    "progress_message": j.progress_message,
                    "result_json": j.result_json,
                    "error": j.error,
                    "created_at": j.created_at,
                    "updated_at": j.updated_at,
                }
            }
            self._send_json(payload, status=200)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                self._send_json({"error": "job not found"}, status=404)
            else:
                self._send_json({"error": str(e)}, status=502)
        except Exception as e:  # noqa: BLE001
            logger.exception("GET %s failed", self.path)
            self._send_json({"error": str(e)}, status=500)

    # ---- POST /<api_path> → workflow container ---------------------------

    def do_POST(self) -> None:
        api_path = self.path
        # Internal endpoint: Workflow Service pushes new (api_path, endpoint)
        # mappings here after a successful deploy. Documented as a gateway-
        # internal contract; not for external clients.
        if api_path == "/__platform/register-route":
            self._handle_register_route()
            return
        try:
            target_addr = self.registry.get_workflow_address(api_path)
        except ValueError:
            self._send_json({"error": f"no workflow registered for {api_path}"}, status=404)
            return

        content_length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(content_length) if content_length > 0 else b""

        upstream_url = f"http://{target_addr}{api_path}"
        try:
            self._forward(upstream_url, body)
        except httpx.HTTPError as e:
            logger.exception("forward to %s failed", upstream_url)
            self._send_json({"error": f"workflow unreachable: {e}"}, status=502)
        except Exception as e:  # noqa: BLE001
            logger.exception("POST %s failed", api_path)
            self._send_json({"error": str(e)}, status=500)

    def _forward(self, upstream_url: str, body: bytes) -> None:
        """Stream-aware forward: decides between buffered and SSE pass-through."""
        forward_headers = {"content-type": "application/json"}
        with _get_http_client().stream(
            "POST", upstream_url, content=body, headers=forward_headers
        ) as upstream:
            content_type = upstream.headers.get("content-type", "")
            if content_type.startswith("text/event-stream"):
                self._stream_through(upstream)
                return
            # Buffered pass-through for JSON (sync/200 or async/202 alike).
            buffered = b"".join(upstream.iter_bytes())
            self.send_response(upstream.status_code)
            self.send_header("Content-Type", content_type or "application/json")
            self.send_header("Content-Length", str(len(buffered)))
            self.end_headers()
            self.wfile.write(buffered)

    def _handle_register_route(self) -> None:
        """Workflow Service → gateway push: ``{"api_path": ..., "endpoint": ...}``.

        Internal-only (no external auth in the demo); matches the routing
        model in chapter 8: source of truth is the Workflow Service, the
        gateway holds a fast local cache populated by push.
        """
        content_length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(content_length) if content_length > 0 else b""
        try:
            payload = json.loads(body)
            api_path = payload["api_path"]
            endpoint = payload["endpoint"]
        except (KeyError, json.JSONDecodeError) as e:
            self._send_json({"error": f"bad payload: {e}"}, status=400)
            return
        self.registry.register_workflow(api_path, endpoint)
        self._send_json({"success": True}, status=200)

    def _stream_through(self, upstream: httpx.Response) -> None:
        """Write SSE chunks to the external client as they arrive upstream."""
        self.send_response(upstream.status_code)
        self.send_header("Content-Type", upstream.headers.get("content-type", "text/event-stream"))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        for chunk in upstream.iter_bytes():
            if not chunk:
                continue
            self.wfile.write(chunk)
            self.wfile.flush()

    # ---- helpers ---------------------------------------------------------

    def _send_json(self, payload: dict, *, status: int) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A002
        """Override to reduce noise."""
        pass
