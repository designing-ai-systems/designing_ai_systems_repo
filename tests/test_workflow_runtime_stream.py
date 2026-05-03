"""Tests for the runtime server's stream/SSE handler (Listing 8.6)."""

import json
import time
from typing import Iterator

from fastapi.testclient import TestClient

from genai_platform import workflow
from genai_platform.runtime.server import build_app


@workflow(name="counter", api_path="/count", response_mode="stream", timeout_seconds=5)
def counter(n: int = 3) -> Iterator[dict]:
    for i in range(n):
        yield {"i": i}


@workflow(name="bumper", api_path="/bump", response_mode="stream", timeout_seconds=2)
def bumper() -> Iterator[dict]:
    yield {"phase": "start"}
    raise RuntimeError("intentional mid-stream failure")


@workflow(name="laggy_stream", api_path="/lag", response_mode="stream", timeout_seconds=1)
def laggy_stream() -> Iterator[dict]:
    yield {"phase": "first"}
    time.sleep(5)  # Will trip the chunk timeout.
    yield {"phase": "never"}


def _parse_sse(body: bytes) -> list:
    """Return a list of payloads decoded from `data: {json}\\n\\n` events."""
    out = []
    for line in body.splitlines():
        if line.startswith(b"data: "):
            out.append(json.loads(line[len(b"data: ") :].decode("utf-8")))
    return out


class TestStreamHandler:
    def test_yields_chunks_as_sse_events(self):
        client = TestClient(build_app(counter))
        r = client.post("/count", json={"n": 3})
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        events = _parse_sse(r.content)
        assert events == [{"i": 0}, {"i": 1}, {"i": 2}]

    def test_mid_stream_exception_emits_error_event_and_closes(self):
        client = TestClient(build_app(bumper))
        r = client.post("/bump", json={})
        events = _parse_sse(r.content)
        assert events[0] == {"phase": "start"}
        assert "error" in events[-1]
        assert "intentional mid-stream failure" in events[-1]["error"]

    def test_chunk_timeout_emits_error_event(self):
        client = TestClient(build_app(laggy_stream))
        r = client.post("/lag", json={})
        events = _parse_sse(r.content)
        # First chunk made it; second one timed out.
        assert events[0] == {"phase": "first"}
        assert "error" in events[-1]
        assert "timed out" in events[-1]["error"].lower()
