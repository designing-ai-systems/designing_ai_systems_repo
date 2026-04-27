"""Tests for the runtime server's async handler (Listing 8.8).

Uses an in-process WorkflowServiceImpl + a stub WorkflowClient that
bypasses the gateway gRPC channel (same _DirectStub trick as
test_workflow_client.py). After the POST returns 202, the test awaits
the spawned asyncio.Task that the handler stashed on app.state, then
asserts the final job state in the in-process service.
"""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from genai_platform import GenAIPlatform, workflow
from genai_platform.clients.workflow import WorkflowClient
from genai_platform.runtime.server import build_app
from services.workflow.service import WorkflowServiceImpl


class _DirectStub:
    def __init__(self, servicer):
        self._servicer = servicer

    def __getattr__(self, method):
        impl = getattr(self._servicer, method)

        def call(request, metadata=None):  # noqa: ARG001
            return impl(request, _FakeContext())

        return call


class _FakeContext:
    def __init__(self):
        self.code = None
        self.details_str = None

    def set_code(self, c):
        self.code = c

    def set_details(self, d):
        self.details_str = d


@pytest.fixture
def platform_with_in_process_workflow_service():
    """Returns (platform, servicer) where platform.workflows talks directly to servicer."""
    platform = GenAIPlatform(gateway_url="localhost:50051")
    servicer = WorkflowServiceImpl()
    client = WorkflowClient(platform)
    client._stub = _DirectStub(servicer)
    platform._workflows = client
    return platform, servicer


@workflow(name="researcher", api_path="/research", response_mode="async", timeout_seconds=5)
def researcher(topic: str) -> dict:
    return {"summary": f"summary of {topic}"}


@workflow(name="bad_async", api_path="/bad", response_mode="async", timeout_seconds=5)
def bad_async() -> dict:
    raise RuntimeError("intentional async failure")


@pytest.fixture
def _drain_async_task():
    """Helper: synchronously drain the background task left on app.state."""

    def drain(client):
        task = client.app.state.last_async_task
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(task)
        finally:
            loop.close()

    return drain


class TestAsyncHandler:
    def test_returns_202_with_job_id(self, platform_with_in_process_workflow_service):
        platform, _ = platform_with_in_process_workflow_service
        client = TestClient(build_app(researcher, platform=platform))

        r = client.post("/research", json={"topic": "vector dbs"})

        assert r.status_code == 202
        body = r.json()
        assert body["status"] == "pending"
        assert body["job_id"]
        assert body["status_url"] == f"/jobs/{body['job_id']}"

    def test_background_task_completes_job_with_result(
        self, platform_with_in_process_workflow_service
    ):
        platform, servicer = platform_with_in_process_workflow_service
        client = TestClient(build_app(researcher, platform=platform))

        r = client.post("/research", json={"topic": "embedding models"})
        job_id = r.json()["job_id"]

        # asyncio.create_task uses the same loop the handler was on; in
        # TestClient that loop is short-lived. Wait for the task to be done
        # by polling the in-process job store.
        import time

        deadline = time.time() + 5.0
        while time.time() < deadline:
            job = servicer.jobs.get_job(job_id)
            if job is not None and job.status == "succeeded":
                break
            time.sleep(0.05)

        job = servicer.jobs.get_job(job_id)
        assert job is not None
        assert job.status == "succeeded", f"got status={job.status}, error={job.error}"
        assert json.loads(job.result_json) == {"summary": "summary of embedding models"}

    def test_background_task_marks_failed_on_exception(
        self, platform_with_in_process_workflow_service
    ):
        platform, servicer = platform_with_in_process_workflow_service
        client = TestClient(build_app(bad_async, platform=platform))

        r = client.post("/bad", json={})
        job_id = r.json()["job_id"]

        import time

        deadline = time.time() + 5.0
        while time.time() < deadline:
            job = servicer.jobs.get_job(job_id)
            if job is not None and job.status == "failed":
                break
            time.sleep(0.05)

        job = servicer.jobs.get_job(job_id)
        assert job is not None
        assert job.status == "failed"
        assert "intentional async failure" in job.error
