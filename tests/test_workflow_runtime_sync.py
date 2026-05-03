"""Tests for the runtime server's sync handler (Listing 8.4)."""

import time

from fastapi.testclient import TestClient

from genai_platform import workflow
from genai_platform.runtime.server import build_app


@workflow(name="echo", api_path="/echo", response_mode="sync", timeout_seconds=2)
def echo(message: str) -> dict:
    return {"echoed": message}


@workflow(
    name="adder",
    api_path="/add",
    response_mode="sync",
    timeout_seconds=2,
)
def adder(a: int, b: int) -> dict:
    return {"sum": a + b}


@workflow(name="slow", api_path="/slow", response_mode="sync", timeout_seconds=1)
def slow_workflow(seconds: float = 5) -> dict:
    time.sleep(seconds)
    return {"finished": True}


@workflow(name="boom", api_path="/boom", response_mode="sync", timeout_seconds=2)
def boom() -> dict:
    raise RuntimeError("intentional failure")


class TestSyncHandler:
    def test_returns_200_and_json_body(self):
        client = TestClient(build_app(echo))
        r = client.post("/echo", json={"message": "hi"})
        assert r.status_code == 200
        assert r.json() == {"echoed": "hi"}

    def test_passes_kwargs_through(self):
        client = TestClient(build_app(adder))
        r = client.post("/add", json={"a": 2, "b": 3})
        assert r.status_code == 200
        assert r.json() == {"sum": 5}

    def test_timeout_returns_504(self):
        client = TestClient(build_app(slow_workflow))
        r = client.post("/slow", json={"seconds": 5})
        assert r.status_code == 504
        assert "timed out" in r.json()["error"].lower()

    def test_exception_returns_500(self):
        client = TestClient(build_app(boom))
        r = client.post("/boom", json={})
        assert r.status_code == 500
        assert "intentional failure" in r.json()["error"]

    def test_invalid_json_returns_422(self):
        client = TestClient(build_app(echo))
        r = client.post("/echo", content=b"not json")
        # FastAPI / starlette translates the body parse failure to 400 or 422.
        assert r.status_code in (400, 422)
