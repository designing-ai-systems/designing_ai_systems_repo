"""
Job storage for async workflows (Listings 8.10–8.11).

State machine mirrors the chapter-6 ToolTask in
`services/tools/models.py:85-92`:
    pending → running → succeeded | failed | cancelled | timed_out

In-memory backend is sufficient for the book demo. The matching SQL schema
lives in `services/workflow/schema.sql` (Listing 8.9) for production
deployments; the Postgres-backed `JobStore` is deferred.
"""

import time
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Optional

from services.workflow.models import WorkflowJob


class JobStore(ABC):
    """Abstract async-job store."""

    @abstractmethod
    def create_job(self, workflow_id: str, input_json: str, assigned_endpoint: str = "") -> str:
        """Create a new job in 'pending' state. Returns the job id."""

    @abstractmethod
    def get_job(self, job_id: str) -> Optional[WorkflowJob]:
        """Get a job by id, or None if unknown."""

    @abstractmethod
    def update_progress(self, job_id: str, message: str) -> bool:
        """Update progress message. Returns True if the job exists."""

    @abstractmethod
    def save_checkpoint(self, job_id: str, checkpoint_json: str) -> bool:
        """Persist a checkpoint payload. Returns True if the job exists."""

    @abstractmethod
    def complete(self, job_id: str, result_json: str) -> bool:
        """Mark succeeded with a result. Returns True if the job exists."""

    @abstractmethod
    def fail(self, job_id: str, error: str) -> bool:
        """Mark failed with an error message. Returns True if the job exists."""

    @abstractmethod
    def cancel(self, job_id: str) -> bool:
        """Mark cancelled. Returns True if the job exists."""


class InMemoryJobStore(JobStore):
    def __init__(self) -> None:
        self._jobs: Dict[str, WorkflowJob] = {}

    def create_job(self, workflow_id: str, input_json: str, assigned_endpoint: str = "") -> str:
        jid = uuid.uuid4().hex
        self._jobs[jid] = WorkflowJob(
            job_id=jid,
            workflow_id=workflow_id,
            input_json=input_json,
            assigned_endpoint=assigned_endpoint,
            status="pending",
        )
        return jid

    def get_job(self, job_id: str) -> Optional[WorkflowJob]:
        return self._jobs.get(job_id)

    def _bump(self, job: WorkflowJob) -> None:
        job.updated_at = int(time.time() * 1000)

    def update_progress(self, job_id: str, message: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None:
            return False
        job.progress_message = message
        self._bump(job)
        return True

    def save_checkpoint(self, job_id: str, checkpoint_json: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None:
            return False
        job.checkpoint_json = checkpoint_json
        self._bump(job)
        return True

    def complete(self, job_id: str, result_json: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None:
            return False
        job.status = "succeeded"
        job.result_json = result_json
        self._bump(job)
        return True

    def fail(self, job_id: str, error: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None:
            return False
        job.status = "failed"
        job.error = error
        self._bump(job)
        return True

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None:
            return False
        job.status = "cancelled"
        self._bump(job)
        return True
