"""Tests for the runtime server's shallow health endpoints.

Per Chapter 8: probes are intentionally shallow — they check only that the
process is up and accepting traffic, not that downstream dependencies are
reachable. Coupling the probe to dependency health causes cascading
failures (a 3-second gateway restart drops every workflow from the load
balancer simultaneously).
"""

from fastapi.testclient import TestClient

from genai_platform import workflow
from genai_platform.runtime.server import build_app


@workflow(name="healthy", api_path="/h", response_mode="sync")
def healthy() -> dict:
    return {"ok": True}


def test_liveness_returns_200():
    client = TestClient(build_app(healthy))
    r = client.get("/health/live")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_readiness_returns_200():
    client = TestClient(build_app(healthy))
    r = client.get("/health/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
