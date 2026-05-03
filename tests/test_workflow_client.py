"""Tests for WorkflowClient job-lifecycle methods (commit 1 surface).

The runtime-server-driven composition methods (call, call_parallel) land in
commit 2 along with the FastAPI runtime server. These tests exercise the
gRPC round-trip against an in-process WorkflowServiceImpl, with the
WorkflowClient stub swapped to skip the gateway routing layer.
"""

import json

import pytest

from genai_platform import GenAIPlatform
from genai_platform.clients.workflow import WorkflowClient
from services.workflow.service import WorkflowServiceImpl


class _DirectStub:
    """Adapter that lets WorkflowClient call WorkflowServiceImpl directly.

    The real client's gRPC stub takes (request, metadata=...) keyword args.
    The servicer takes (request, context). This stub bridges the two so the
    client logic can be tested without a live gRPC server.
    """

    def __init__(self, servicer: WorkflowServiceImpl):
        self._servicer = servicer

    def __getattr__(self, method):
        impl = getattr(self._servicer, method)

        def call(request, metadata=None):  # noqa: ARG001 — metadata is irrelevant in tests
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
def client_with_servicer():
    platform = GenAIPlatform(gateway_url="localhost:50051")
    client = WorkflowClient(platform)
    servicer = WorkflowServiceImpl()
    # Bypass the real gRPC channel; route all stub calls into the in-process servicer.
    client._stub = _DirectStub(servicer)
    return client, servicer


class TestJobLifecycle:
    def test_create_get_update_complete(self, client_with_servicer):
        client, _ = client_with_servicer

        jid = client.create_job(workflow_id="w-1", input_json='{"q": "x"}')
        assert jid

        job = client.get_job_status(jid)
        assert job.status == "pending"
        assert job.input_json == '{"q": "x"}'

        assert client.update_job_progress(jid, "step 1") is True
        assert client.save_checkpoint(jid, '{"phase": "retrieved"}') is True
        assert client.complete_job(jid, '{"answer": 42}') is True

        job = client.get_job_status(jid)
        assert job.status == "succeeded"
        assert json.loads(job.result_json) == {"answer": 42}
        assert job.checkpoint_json == '{"phase": "retrieved"}'

    def test_fail(self, client_with_servicer):
        client, _ = client_with_servicer
        jid = client.create_job(workflow_id="w-1", input_json="{}")
        assert client.fail_job(jid, "boom") is True
        assert client.get_job_status(jid).error == "boom"

    def test_cancel(self, client_with_servicer):
        client, _ = client_with_servicer
        jid = client.create_job(workflow_id="w-1", input_json="{}")
        assert client.cancel_job(jid) is True
        assert client.get_job_status(jid).status == "cancelled"


class TestPlatformIntegration:
    def test_platform_exposes_workflows_property(self):
        platform = GenAIPlatform(gateway_url="localhost:50051")
        assert platform.workflows is not None
        assert isinstance(platform.workflows, WorkflowClient)
        # Lazy: same instance returned on second access
        assert platform.workflows is platform.workflows
