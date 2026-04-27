"""
Workflow Service domain models.

Python dataclasses for workflow specs, deployments, and async jobs.
The gRPC servicer translates between these and the proto messages
defined in ``proto/workflow.proto``.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 8.22: WorkflowSpec, ScalingConfig, ResourceConfig
  - Listing 8.23: WorkflowDeployment
  - Listing 8.10–8.11: WorkflowJob (mirrors the chapter-6 ToolTask state machine
    in services/tools/models.py: pending → running → succeeded/failed/cancelled/timed_out)
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ScalingConfig:
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_percent: int = 70


@dataclass
class ResourceConfig:
    cpu: str = "500m"
    memory: str = "512Mi"
    gpu_type: str = ""
    num_gpus: int = 0


@dataclass
class ReliabilityConfig:
    timeout_seconds: int = 30
    max_retries: int = 3


@dataclass
class WorkflowSpec:
    name: str = ""
    api_path: str = ""
    container_image: str = ""
    response_mode: str = "sync"
    scaling: Optional[ScalingConfig] = None
    resources: Optional[ResourceConfig] = None
    reliability: Optional[ReliabilityConfig] = None
    version: int = 1

    def __post_init__(self) -> None:
        if self.scaling is None:
            self.scaling = ScalingConfig()
        if self.resources is None:
            self.resources = ResourceConfig()
        if self.reliability is None:
            self.reliability = ReliabilityConfig()


@dataclass
class WorkflowDeployment:
    workflow_id: str = ""
    deployment_id: str = ""
    version: int = 1
    status: str = "deploying"
    current_replicas: int = 0
    desired_replicas: int = 0
    healthy_endpoints: List[str] = field(default_factory=list)


@dataclass
class WorkflowJob:
    """
    Async-job record. Status transitions follow the chapter-6 ToolTask model:
        pending → running → succeeded | failed | cancelled | timed_out
    """

    job_id: str = ""
    workflow_id: str = ""
    status: str = "pending"
    progress_message: str = ""
    input_json: str = ""
    result_json: str = ""
    error: str = ""
    checkpoint_json: str = ""
    assigned_endpoint: str = ""
    created_at: int = 0
    updated_at: int = 0

    def __post_init__(self) -> None:
        now_ms = int(time.time() * 1000)
        if self.created_at == 0:
            self.created_at = now_ms
        if self.updated_at == 0:
            self.updated_at = now_ms
