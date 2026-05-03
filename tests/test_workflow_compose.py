"""Tests for workflow composition (Listings 8.17–8.20).

Uses ``httpx.MockTransport`` to fake child workflow endpoints — the parent
workflow's ``platform.workflows.call(...)`` issues HTTP through the mock
transport, exercising the real routing logic (sync JSON / SSE stream /
202+poll) without binding a port.
"""

import json

import httpx

from genai_platform import GenAIPlatform
from genai_platform.clients.workflow import WorkflowClient


def _client_with_transport(handler):
    """Build a WorkflowClient whose composition HTTP client uses MockTransport."""
    platform = GenAIPlatform(gateway_url="localhost:50051")
    client = WorkflowClient(platform)
    client._http_client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="http://gateway",
    )
    return client


# ---- Sync (200 + JSON) -----------------------------------------------------


class TestCallSync:
    def test_returns_parsed_json_body(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/child/sync"
            assert json.loads(request.content) == {"q": "x"}
            return httpx.Response(200, json={"answer": 42})

        client = _client_with_transport(handler)
        result = client.call("/child/sync", {"q": "x"})
        assert result == {"answer": 42}


# ---- Async (202 + poll) ----------------------------------------------------


class TestCallAsync:
    def test_polls_jobs_endpoint_until_complete(self):
        # Each request increments a counter to drive the state machine.
        state = {"polls": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST" and request.url.path == "/child/async":
                return httpx.Response(
                    202,
                    json={
                        "job_id": "j-1",
                        "status": "pending",
                        "status_url": "/jobs/j-1",
                    },
                )
            if request.method == "GET" and request.url.path == "/jobs/j-1":
                state["polls"] += 1
                if state["polls"] < 3:
                    return httpx.Response(
                        200, json={"job": {"status": "running", "result_json": ""}}
                    )
                return httpx.Response(
                    200,
                    json={
                        "job": {
                            "status": "succeeded",
                            "result_json": json.dumps({"answer": 99}),
                        }
                    },
                )
            return httpx.Response(404)

        client = _client_with_transport(handler)
        result = client.call("/child/async", {}, poll_interval=0.001)
        assert result == {"answer": 99}
        assert state["polls"] == 3

    def test_failed_job_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                return httpx.Response(
                    202,
                    json={"job_id": "j-2", "status": "pending", "status_url": "/jobs/j-2"},
                )
            return httpx.Response(200, json={"job": {"status": "failed", "error": "boom"}})

        client = _client_with_transport(handler)
        try:
            client.call("/child/async", {}, poll_interval=0.001)
        except RuntimeError as e:
            assert "boom" in str(e)
        else:
            raise AssertionError("expected RuntimeError")


# ---- Stream (text/event-stream) --------------------------------------------


class TestCallStream:
    def test_consumes_sse_events_into_list(self):
        def handler(request: httpx.Request) -> httpx.Response:
            body = b'data: {"i": 0}\n\ndata: {"i": 1}\n\ndata: {"i": 2}\n\n'
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=body,
            )

        client = _client_with_transport(handler)
        chunks = client.call("/child/stream", {})
        assert chunks == [{"i": 0}, {"i": 1}, {"i": 2}]


# ---- HTTP retry on 5xx -----------------------------------------------------


class TestHttpRetry:
    def test_retries_on_5xx_then_succeeds(self):
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            if attempts["n"] < 3:
                return httpx.Response(503, json={"err": "down"})
            return httpx.Response(200, json={"ok": True})

        client = _client_with_transport(handler)
        result = client.call("/child/sync", {})
        assert result == {"ok": True}
        assert attempts["n"] == 3

    def test_does_not_retry_on_4xx(self):
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            return httpx.Response(400, json={"err": "bad"})

        client = _client_with_transport(handler)
        try:
            client.call("/child/sync", {})
        except httpx.HTTPStatusError:
            assert attempts["n"] == 1  # exactly one attempt
        else:
            raise AssertionError("expected HTTPStatusError on 4xx")


# ---- Parallel composition --------------------------------------------------


class TestCallParallel:
    def test_returns_results_in_submission_order(self):
        # Each path returns its name, no matter which thread submits first.
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"path": request.url.path})

        client = _client_with_transport(handler)
        results = client.call_parallel(
            [
                ("/papers", {"q": "x"}),
                ("/news", {"q": "x"}),
                ("/patents", {"q": "x"}),
            ]
        )
        assert [r["path"] for r in results] == ["/papers", "/news", "/patents"]

    def test_propagates_first_failure(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/bad":
                return httpx.Response(400, json={"err": "bad"})
            return httpx.Response(200, json={"path": request.url.path})

        client = _client_with_transport(handler)
        try:
            client.call_parallel([("/ok", {}), ("/bad", {})])
        except httpx.HTTPStatusError:
            pass
        else:
            raise AssertionError("expected HTTPStatusError")
