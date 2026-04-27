"""
The ``@workflow`` decorator — entry point for declaring a deployable AI workflow.

A workflow is a Python function with deployment metadata attached. The
deploy CLI reads that metadata to package the function into a container;
the SDK runtime server (``genai_platform/runtime/server.py``, added in
commit 2 of the chapter-8 plan) reads the same metadata at container
startup to pick the right request handler (sync / stream / async).

Book: "Designing AI Systems" — Listing 8.2.

Field schema diverges from the manuscript draft of Listing 8.2 — this
implementation uses flat keyword arguments instead of nested ``autoscaling_config`` /
``deployment_config`` dicts. Reasons documented in
``chapters/book_discrepancies_chapter8.md`` (#1):

- discoverability and type safety: each kwarg has a name, default and
  checked type; dict keys silently swallow typos and break IDE
  autocomplete.
- no pass-through benefit: nothing in the deploy CLI or runtime server
  treats these as opaque blobs; both pluck specific fields out.
- if the parameter list ever grows enough to need grouping, the right
  next step is a typed ``ScalingConfig``/``ResourceConfig`` dataclass,
  not a free-form dict.

Example::

    @workflow(
        name="patient_intake_assistant",
        api_path="/patient-assistant",
        response_mode="sync",
        min_replicas=1,
        max_replicas=10,
        target_cpu_percent=70,
        cpu="500m",
        memory="512Mi",
        timeout_seconds=30,
        max_retries=3,
    )
    def handle_patient_question(question, patient_id):
        ...
"""

from functools import wraps
from typing import Any, Callable, Dict

VALID_RESPONSE_MODES = {"sync", "stream", "async"}


def workflow(
    name: str,
    api_path: str,
    response_mode: str = "sync",
    *,
    # Scaling
    min_replicas: int = 1,
    max_replicas: int = 10,
    target_cpu_percent: int = 70,
    # Resources
    cpu: str = "500m",
    memory: str = "512Mi",
    gpu_type: str = "",
    num_gpus: int = 0,
    # Reliability
    timeout_seconds: int = 30,
    max_retries: int = 3,
) -> Callable:
    """Decorator that marks a function as a deployable workflow.

    The decorated function still runs unchanged when called directly;
    deployment metadata is attached as ``func._workflow_metadata`` and is
    read by the deploy CLI and the runtime server.

    Args:
        name: Unique workflow identifier.
        api_path: HTTP path the gateway will route to this workflow.
        response_mode: ``"sync"`` (single JSON response), ``"stream"``
            (Server-Sent Events / progressive tokens), or ``"async"``
            (returns 202 + ``job_id``; clients poll ``/jobs/{id}``).
        min_replicas, max_replicas, target_cpu_percent: Horizontal
            autoscaling settings consumed by the generated Kubernetes
            HorizontalPodAutoscaler manifest (commit 3 of the chapter-8
            plan generates these).
        cpu, memory, gpu_type, num_gpus: Per-replica resource limits;
            ``gpu_type=""`` and ``num_gpus=0`` mean CPU-only.
        timeout_seconds, max_retries: Reliability knobs read by the
            runtime server's request handlers.

    Raises:
        ValueError: ``response_mode`` is not one of sync / stream / async.
    """
    if response_mode not in VALID_RESPONSE_MODES:
        raise ValueError(
            f"Invalid response_mode {response_mode!r}; "
            f"must be one of {sorted(VALID_RESPONSE_MODES)}"
        )

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        metadata: Dict[str, Any] = {
            "name": name,
            "api_path": api_path,
            "response_mode": response_mode,
            "min_replicas": min_replicas,
            "max_replicas": max_replicas,
            "target_cpu_percent": target_cpu_percent,
            "cpu": cpu,
            "memory": memory,
            "gpu_type": gpu_type,
            "num_gpus": num_gpus,
            "timeout_seconds": timeout_seconds,
            "max_retries": max_retries,
        }
        wrapper._workflow_metadata = metadata
        return wrapper

    return decorator
