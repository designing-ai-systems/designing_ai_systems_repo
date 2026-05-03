"""
Workflow Service client — job lifecycle + workflow composition.

Two distinct surfaces in one client:

1. **Job lifecycle (commit 1, Listings 8.10–8.11)** — what
   `@workflow`-decorated *async* functions reach for via
   ``platform.workflows.update_job_progress(...)`` while running inside
   the SDK runtime server. The runtime server's async handler sets the
   active ``job_id`` in a `ContextVar`; convenience helpers here read it
   so workflow authors never have to plumb the id manually.

2. **Workflow composition (commit 2, Listings 8.17–8.20)** — what *parent*
   workflows use to call *child* workflows: ``call(api_path, body)`` and
   ``call_parallel([(path, body), ...])``. The parent workflow never sees
   the child's response mode; ``call`` auto-detects sync JSON / SSE
   stream / 202+poll and returns a Python value either way.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import httpx

from proto import workflow_pb2, workflow_pb2_grpc
from services.workflow.models import WorkflowJob

from .base import BaseClient

# Set by the runtime server's async handler before invoking the user's
# function. Workflow authors call `platform.workflows.update_job_progress(...)`
# without passing a job_id; the helper resolves it from this ContextVar.
job_id_var: ContextVar[Optional[str]] = ContextVar("workflow_job_id", default=None)


_DEFAULT_GATEWAY_HTTP_URL = "http://localhost:8080"
_DEFAULT_PARALLEL_WORKERS = 8

# HTTP retry layer for workflow→workflow calls (Listing 8.19) — independent of
# the gRPC RetryInterceptor used by SDK→service calls.
_HTTP_RETRY_ATTEMPTS = 3
_HTTP_RETRY_BASE_BACKOFF = 0.1


class WorkflowClient(BaseClient):
    """Workflow Service client — job lifecycle + workflow composition."""

    def __init__(self, platform):
        super().__init__(platform, service_name="workflow")
        self._stub = workflow_pb2_grpc.WorkflowServiceStub(self._channel)
        # Lazily-instantiated HTTP client for composition. Tests override
        # this with a MockTransport-backed client. Real usage points at the
        # gateway's HTTP entry point (port 8080 by default).
        self._http_client: Optional[httpx.Client] = None

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


# ---- Workflow composition (Listings 8.17–8.20) -----------------------------


def _gateway_http_url() -> str:
    return os.environ.get("GENAI_GATEWAY_HTTP_URL", _DEFAULT_GATEWAY_HTTP_URL)


def _ensure_http_client(client: WorkflowClient) -> httpx.Client:
    if client._http_client is None:
        client._http_client = httpx.Client(base_url=_gateway_http_url())
    return client._http_client


def _http_post_with_retry(
    http: httpx.Client,
    path: str,
    body: Dict[str, Any],
    *,
    sleep=time.sleep,
) -> httpx.Response:
    """Listing 8.19: retry transient 5xx + connection errors with exponential backoff.

    4xx errors propagate immediately (caller's contract violation, retrying
    won't fix it). Returns the final ``httpx.Response``; caller is
    responsible for ``raise_for_status`` if needed.
    """
    last: Optional[httpx.Response] = None
    for attempt in range(_HTTP_RETRY_ATTEMPTS):
        try:
            response = http.post(path, json=body)
        except httpx.RequestError:
            if attempt >= _HTTP_RETRY_ATTEMPTS - 1:
                raise
            sleep(_HTTP_RETRY_BASE_BACKOFF * (2**attempt))
            continue

        if response.status_code < 500:
            return response

        last = response
        if attempt < _HTTP_RETRY_ATTEMPTS - 1:
            sleep(_HTTP_RETRY_BASE_BACKOFF * (2**attempt))

    assert last is not None
    return last


def _consume_stream(response: httpx.Response) -> List[Dict[str, Any]]:
    """Listing 8.20: parse a `text/event-stream` body into a list of payloads."""
    out: List[Dict[str, Any]] = []
    body = response.content
    for line in body.splitlines():
        if line.startswith(b"data: "):
            out.append(json.loads(line[len(b"data: ") :].decode("utf-8")))
    return out


def _poll_until_complete(
    http: httpx.Client,
    status_url: str,
    *,
    poll_interval: float = 0.5,
    sleep=time.sleep,
) -> Dict[str, Any]:
    """Listing 8.20: GET ``status_url`` until job leaves pending/running.

    The gateway's `/jobs/{id}` proxy returns ``{"job": {...}}`` shaped like
    the proto ``WorkflowJob``. On terminal states:
      - succeeded → returns the parsed ``result_json`` payload
      - failed    → raises ``RuntimeError(error)``
      - cancelled → raises ``RuntimeError("cancelled")``
    """
    while True:
        response = http.get(status_url)
        response.raise_for_status()
        job = response.json()["job"]
        status = job.get("status")
        if status == "succeeded":
            payload = job.get("result_json") or "{}"
            return json.loads(payload)
        if status == "failed":
            raise RuntimeError(job.get("error") or "workflow failed")
        if status == "cancelled":
            raise RuntimeError("workflow cancelled")
        sleep(poll_interval)


def _call_one(
    client: WorkflowClient,
    api_path: str,
    body: Dict[str, Any],
    *,
    poll_interval: float,
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """The single-call routing logic, factored out for ``call_parallel``."""
    http = _ensure_http_client(client)
    response = _http_post_with_retry(http, api_path, body)

    if response.status_code >= 400 and response.status_code < 500:
        response.raise_for_status()

    if response.status_code == 202:
        # Async child — poll the status URL.
        try:
            status_url = response.json().get("status_url") or ""
        except json.JSONDecodeError:
            status_url = ""
        if not status_url:
            raise RuntimeError(
                f"async child workflow returned 202 without status_url at {api_path}"
            )
        return _poll_until_complete(http, status_url, poll_interval=poll_interval)

    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if content_type.startswith("text/event-stream"):
        return _consume_stream(response)
    return response.json()


def _call(
    self: WorkflowClient,
    api_path: str,
    body: Dict[str, Any],
    *,
    poll_interval: float = 0.5,
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Listing 8.18: call a child workflow over HTTP.

    Auto-detects the child's response mode from the response shape:

    - ``200 + application/json``      → returns the parsed dict.
    - ``200 + text/event-stream``     → returns ``[chunk_dict, ...]``.
    - ``202``                         → polls ``status_url`` until done.

    The parent workflow's author writes the same line of code for every
    child; if the child team flips ``response_mode="sync"`` to ``"async"``
    later, parent code keeps working.
    """
    return _call_one(self, api_path, body, poll_interval=poll_interval)


def _call_parallel(
    self: WorkflowClient,
    specs: Sequence[Tuple[str, Dict[str, Any]]],
    *,
    poll_interval: float = 0.5,
) -> List[Union[Dict[str, Any], List[Dict[str, Any]]]]:
    """Listing 8.17: fan out to multiple child workflows concurrently.

    Results are returned in the submission order. The first failure is
    propagated immediately (other in-flight calls keep running but their
    results are discarded — same semantics as ``concurrent.futures.wait``
    with ``FIRST_EXCEPTION``).
    """
    if not specs:
        return []
    workers = min(_DEFAULT_PARALLEL_WORKERS, len(specs))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_call_one, self, path, body, poll_interval=poll_interval)
            for path, body in specs
        ]
        return [f.result() for f in futures]


# Bind the composition functions onto the class. Defined as module-level
# functions for readability; method form keeps the public API tidy.
WorkflowClient.call = _call
WorkflowClient.call_parallel = _call_parallel
