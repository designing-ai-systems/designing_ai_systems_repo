"""
Workflow Service client — job-lifecycle surface (commit 1 of the chapter-8 plan).

This client is what `@workflow`-decorated functions reach for via
``platform.workflows.update_job_progress(...)`` etc. when running inside the
SDK runtime server (added in commit 2). The runtime server sets the active
``job_id`` in a `ContextVar`; the convenience helpers here read that
variable so workflow authors never have to plumb the id manually.

Composition methods (`call`, `call_parallel` — Listings 8.17–8.20) and the
HTTP plumbing they need land in commit 2 alongside the runtime server.

Book: "Designing AI Systems"
  - Listing 8.10: /jobs/{job_id} polling endpoint backed by these RPCs
  - Listing 8.11: progress reporting + checkpointing
"""

from contextvars import ContextVar
from typing import Optional

from proto import workflow_pb2, workflow_pb2_grpc
from services.workflow.models import WorkflowJob

from .base import BaseClient

# Set by the runtime server's async handler before invoking the user's
# function. Workflow authors call `platform.workflows.update_job_progress(...)`
# without passing a job_id; the helper resolves it from this ContextVar.
job_id_var: ContextVar[Optional[str]] = ContextVar("workflow_job_id", default=None)


class WorkflowClient(BaseClient):
    """Client for Workflow Service — job lifecycle (commit 1)."""

    def __init__(self, platform):
        super().__init__(platform, service_name="workflow")
        self._stub = workflow_pb2_grpc.WorkflowServiceStub(self._channel)

    # ---- Job lifecycle ---------------------------------------------------

    def create_job(
        self,
        workflow_id: str,
        input_json: str,
        assigned_endpoint: str = "",
    ) -> str:
        """Create a new async job. Returns the job id.

        The runtime server's async handler (commit 2) calls this when it
        accepts an async request, then returns ``202 Accepted`` to the
        external caller with the resulting ``job_id``.
        """
        resp = self._stub.CreateJob(
            workflow_pb2.CreateJobRequest(
                workflow_id=workflow_id,
                input_json=input_json,
                assigned_endpoint=assigned_endpoint,
            ),
            metadata=self._metadata,
        )
        return resp.job_id

    def get_job_status(self, job_id: str) -> WorkflowJob:
        """Read the current state of a job."""
        resp = self._stub.GetJobStatus(
            workflow_pb2.GetJobStatusRequest(job_id=job_id),
            metadata=self._metadata,
        )
        j = resp.job
        return WorkflowJob(
            job_id=j.job_id,
            workflow_id=j.workflow_id,
            status=j.status,
            progress_message=j.progress_message,
            input_json=j.input_json,
            result_json=j.result_json,
            error=j.error,
            checkpoint_json=j.checkpoint_json,
            assigned_endpoint=j.assigned_endpoint,
            created_at=j.created_at,
            updated_at=j.updated_at,
        )

    def update_job_progress(self, job_id: Optional[str] = None, message: str = "") -> bool:
        """Update progress message. ``job_id`` defaults to the active job."""
        jid = job_id or _require_active_job_id()
        resp = self._stub.UpdateJobProgress(
            workflow_pb2.UpdateJobProgressRequest(job_id=jid, progress_message=message),
            metadata=self._metadata,
        )
        return resp.success

    def save_checkpoint(self, job_id_or_payload, checkpoint_json: Optional[str] = None) -> bool:
        """Persist a checkpoint for the active job.

        Two call shapes for ergonomics inside a workflow:

            platform.workflows.save_checkpoint(job_id, '{"phase": "x"}')
            platform.workflows.save_checkpoint('{"phase": "x"}')   # uses ContextVar
        """
        if checkpoint_json is None:
            jid = _require_active_job_id()
            payload = job_id_or_payload
        else:
            jid = job_id_or_payload
            payload = checkpoint_json
        resp = self._stub.SaveJobCheckpoint(
            workflow_pb2.SaveJobCheckpointRequest(job_id=jid, checkpoint_json=payload),
            metadata=self._metadata,
        )
        return resp.success

    def load_checkpoint(self, job_id: Optional[str] = None) -> str:
        """Return the most recently saved checkpoint payload, or '' if none."""
        jid = job_id or _require_active_job_id()
        return self.get_job_status(jid).checkpoint_json

    def complete_job(self, job_id: str, result_json: str) -> bool:
        resp = self._stub.CompleteJob(
            workflow_pb2.CompleteJobRequest(job_id=job_id, result_json=result_json),
            metadata=self._metadata,
        )
        return resp.success

    def fail_job(self, job_id: str, error: str) -> bool:
        resp = self._stub.FailJob(
            workflow_pb2.FailJobRequest(job_id=job_id, error=error),
            metadata=self._metadata,
        )
        return resp.success

    def cancel_job(self, job_id: str) -> bool:
        resp = self._stub.CancelJob(
            workflow_pb2.CancelJobRequest(job_id=job_id),
            metadata=self._metadata,
        )
        return resp.success


def _require_active_job_id() -> str:
    jid = job_id_var.get()
    if jid is None:
        raise RuntimeError(
            "No active workflow job in context — call from inside an async "
            "@workflow function, or pass job_id explicitly."
        )
    return jid
