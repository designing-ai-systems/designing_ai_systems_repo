"""
Local demo harness — boots the Workflow Service + gateway + runtime server
in-process so the chapter-8 quickstart scripts can be run with `python
examples/quickstart_workflow.py` without a docker stack.

Once commit 4 lands ``docker-compose.yml`` and commit 3 lands the
``genai-platform deploy`` CLI, the runtime parts of this harness go away —
you'll ``docker compose up`` and run ``genai-platform deploy …`` instead.
For commit 2 the harness keeps the demos self-contained.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from contextlib import contextmanager
from http.server import HTTPServer
from pathlib import Path
from typing import Callable, Iterator, Optional

# Make ``services`` and ``proto`` importable when running from a clone.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn  # noqa: E402

from genai_platform import GenAIPlatform  # noqa: E402
from genai_platform.runtime.server import build_app  # noqa: E402
from services.gateway.registry import ServiceRegistry  # noqa: E402
from services.gateway.servers import create_grpc_server as create_gateway_grpc  # noqa: E402
from services.gateway.servers import create_http_server  # noqa: E402
from services.shared.server import create_grpc_server  # noqa: E402
from services.workflow.service import WorkflowServiceImpl  # noqa: E402

# Use non-default ports so the demos do not collide with anything the user
# already has running on localhost (and so two demos in the same shell can
# bind cleanly without TIME_WAIT issues).
_WORKFLOW_PORT = 50158
_GATEWAY_GRPC_PORT = 50151
_GATEWAY_HTTP_PORT = 8180
_RUNTIME_BASE_PORT = 8200


class _UvicornInThread:
    """Run a FastAPI app on a threaded uvicorn server we can stop cleanly."""

    def __init__(self, app, host: str, port: int):
        config = uvicorn.Config(app=app, host=host, port=port, log_level="error")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def start(self) -> None:
        self._thread.start()
        # Wait for the server to start listening.
        for _ in range(50):
            if self._server.started:
                return
            time.sleep(0.05)
        raise RuntimeError("uvicorn did not start within 2.5s")

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=5.0)


@contextmanager
def local_platform(
    workflow_func: Callable,
    *,
    runtime_port: Optional[int] = None,
) -> Iterator[GenAIPlatform]:
    """Boot Workflow Service + gateway + runtime server. Yields a configured platform.

    On exit, all servers are shut down. Routes are pre-registered in the
    gateway because the deploy CLI (commit 3) does not exist yet.
    """
    HTTPServer.allow_reuse_address = True

    runtime_port = runtime_port or _RUNTIME_BASE_PORT

    # 1. Workflow Service (gRPC)
    workflow_service = create_grpc_server(
        servicer=WorkflowServiceImpl(),
        port=_WORKFLOW_PORT,
        service_name="workflow",
    )
    workflow_service.start()

    # 2. Gateway: gRPC for SDK->platform-service, HTTP for client->workflow.
    registry = ServiceRegistry()
    registry.register_platform_service("workflow", f"localhost:{_WORKFLOW_PORT}")
    api_path = workflow_func._workflow_metadata["api_path"]
    registry.register_workflow(api_path, f"localhost:{runtime_port}")

    gateway_grpc = create_gateway_grpc(registry, _GATEWAY_GRPC_PORT)
    gateway_grpc.start()
    gateway_http = create_http_server(registry, _GATEWAY_HTTP_PORT)
    gateway_http_thread = threading.Thread(target=gateway_http.serve_forever, daemon=True)
    gateway_http_thread.start()

    # 3. Runtime server for this one workflow.
    os.environ["WORKFLOW_SERVICE_ADDR"] = f"localhost:{_WORKFLOW_PORT}"
    platform = GenAIPlatform(gateway_url=f"localhost:{_GATEWAY_GRPC_PORT}")
    runtime_app = build_app(workflow_func, platform=platform)
    runtime = _UvicornInThread(runtime_app, host="127.0.0.1", port=runtime_port)
    runtime.start()

    # The composition client uses the gateway HTTP URL.
    os.environ["GENAI_GATEWAY_HTTP_URL"] = f"http://localhost:{_GATEWAY_HTTP_PORT}"

    try:
        yield platform
    finally:
        runtime.stop()
        gateway_http.shutdown()
        gateway_grpc.stop(0)
        workflow_service.stop(0)


def gateway_http_url() -> str:
    return os.environ.get("GENAI_GATEWAY_HTTP_URL", f"http://localhost:{_GATEWAY_HTTP_PORT}")
