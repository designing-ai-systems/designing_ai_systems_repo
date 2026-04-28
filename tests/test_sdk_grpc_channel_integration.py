"""End-to-end test of `BaseClient` against a real ``grpc.Server``.

The rest of the suite mocks one of the layers between SDK and servicer:
either swaps the gRPC stub for a `_DirectStub`, calls the servicer
directly with a `FakeContext`, or uses `httpx.MockTransport` for the
HTTP path. None of those exercise `BaseClient`'s actual channel
construction.

That gap let a real bug ship: `BaseClient` defaulted to a TLS channel
for any URL that didn't start with `localhost` / `127.0.0.1`, including
compose-network hostnames like `gateway:50051`. The fix added a
``GENAI_GATEWAY_INSECURE=1`` env var as the explicit opt-out. This
test runs the SDK against an actual in-process insecure ``grpc.Server``
with that env var set, verifying that:

1. The SDK can construct an insecure channel against a non-loopback
   URL when the env var is set,
2. End-to-end calls round-trip through the real gRPC stack — which is
   the layer where the original bug lived.

No Docker, no network setup. Runs everywhere ``pytest`` runs.
"""

import socket
from concurrent import futures

import grpc
import pytest

from genai_platform import GenAIPlatform
from genai_platform.clients.workflow import WorkflowClient
from proto import workflow_pb2_grpc
from services.workflow.service import WorkflowServiceImpl


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def insecure_grpc_server():
    """In-process WorkflowServiceImpl behind a real plaintext grpc.Server."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    workflow_pb2_grpc.add_WorkflowServiceServicer_to_server(
        WorkflowServiceImpl(route_health_check=lambda _: True),
        server,
    )
    port = _free_port()
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    yield port
    server.stop(0)


def test_sdk_call_succeeds_with_insecure_env_var_against_non_loopback_url(
    insecure_grpc_server, monkeypatch
):
    """Non-loopback URL + ``GENAI_GATEWAY_INSECURE=1`` → SDK uses plaintext
    channel and the real gRPC call goes through.

    Without the env var, ``BaseClient`` would build a TLS channel and
    the call would fail with the TLS-handshake error (the bug we shipped
    and then fixed). This test fences that regression.
    """
    monkeypatch.setenv("GENAI_GATEWAY_INSECURE", "1")
    # 0.0.0.0 routes to localhost via the loopback adapter but does NOT
    # start with "localhost" / "127.0.0.1", so it forces BaseClient down
    # the env-var path instead of the loopback shortcut.
    platform = GenAIPlatform(gateway_url=f"0.0.0.0:{insecure_grpc_server}")
    client = WorkflowClient(platform)

    job_id = client.create_job(workflow_id="w-1", input_json='{"q": "x"}')
    assert job_id  # non-empty round-trip

    job = client.get_job_status(job_id)
    assert job.workflow_id == "w-1"
    assert job.input_json == '{"q": "x"}'
    assert job.status == "pending"


def test_sdk_call_succeeds_against_loopback_url_without_env_var(insecure_grpc_server, monkeypatch):
    """Loopback URLs always get plaintext, no env var needed (production
    keeps secure-by-default for non-loopback hostnames; this exercises
    the local-dev shortcut path)."""
    monkeypatch.delenv("GENAI_GATEWAY_INSECURE", raising=False)
    platform = GenAIPlatform(gateway_url=f"localhost:{insecure_grpc_server}")
    client = WorkflowClient(platform)

    job_id = client.create_job(workflow_id="w-2", input_json="{}")
    assert client.get_job_status(job_id).workflow_id == "w-2"
