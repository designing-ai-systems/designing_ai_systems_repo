"""Tests for Workflow Service domain dataclasses (Listings 8.22, 8.23)."""

from services.workflow.models import (
    ReliabilityConfig,
    ResourceConfig,
    ScalingConfig,
    WorkflowDeployment,
    WorkflowJob,
    WorkflowSpec,
)


class TestScalingConfig:
    def test_defaults(self):
        s = ScalingConfig()
        assert s.min_replicas == 1
        assert s.max_replicas == 10
        assert s.target_cpu_percent == 70

    def test_explicit_values(self):
        s = ScalingConfig(min_replicas=2, max_replicas=20, target_cpu_percent=50)
        assert (s.min_replicas, s.max_replicas, s.target_cpu_percent) == (2, 20, 50)


class TestResourceConfig:
    def test_defaults(self):
        r = ResourceConfig()
        assert r.cpu == "500m"
        assert r.memory == "512Mi"
        assert r.gpu_type == ""
        assert r.num_gpus == 0

    def test_gpu(self):
        r = ResourceConfig(cpu="2", memory="8Gi", gpu_type="nvidia-t4", num_gpus=1)
        assert r.gpu_type == "nvidia-t4"
        assert r.num_gpus == 1


class TestReliabilityConfig:
    def test_defaults(self):
        r = ReliabilityConfig()
        assert r.timeout_seconds == 30
        assert r.max_retries == 3


class TestWorkflowSpec:
    def test_defaults_for_required_fields_only(self):
        spec = WorkflowSpec(name="patient_intake", api_path="/patient-assistant")
        assert spec.name == "patient_intake"
        assert spec.api_path == "/patient-assistant"
        assert spec.response_mode == "sync"
        assert spec.version == 1
        assert spec.container_image == ""
        # Nested configs auto-populate to defaults if not provided
        assert spec.scaling is not None and spec.scaling.min_replicas == 1
        assert spec.resources is not None and spec.resources.cpu == "500m"
        assert spec.reliability is not None and spec.reliability.timeout_seconds == 30

    def test_response_mode_values(self):
        for mode in ("sync", "stream", "async"):
            spec = WorkflowSpec(name="w", api_path="/w", response_mode=mode)
            assert spec.response_mode == mode


class TestWorkflowDeployment:
    def test_defaults(self):
        d = WorkflowDeployment(workflow_id="w-1", deployment_id="d-1")
        assert d.workflow_id == "w-1"
        assert d.deployment_id == "d-1"
        assert d.version == 1
        assert d.status == "deploying"
        assert d.current_replicas == 0
        assert d.desired_replicas == 0
        assert d.healthy_endpoints == []


class TestWorkflowJob:
    def test_defaults(self):
        j = WorkflowJob(job_id="j-1", workflow_id="w-1")
        assert j.job_id == "j-1"
        assert j.workflow_id == "w-1"
        assert j.status == "pending"
        assert j.progress_message == ""
        assert j.input_json == ""
        assert j.result_json == ""
        assert j.error == ""
        assert j.checkpoint_json == ""
        assert j.assigned_endpoint == ""
        # Timestamps auto-set
        assert j.created_at > 0
        assert j.updated_at > 0
