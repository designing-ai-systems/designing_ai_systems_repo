"""Tests for the deploy CLI's pure functions: scan, Dockerfile/K8s generation.

The actual ``docker build`` is verified by the operator running
``genai-platform deploy <file>`` during demo / acceptance; we don't
mock it as a pytest case (the unit tests below verify the generated
Dockerfile *text* is correct, which is the part `docker build` would
fail on if it regressed).
"""

import textwrap
from pathlib import Path

import pytest
import yaml

from genai_platform.cli.deploy import (
    generate_dockerfile,
    generate_kubernetes_manifests,
    scan_for_workflows,
)


@pytest.fixture
def workflow_module(tmp_path: Path) -> Path:
    src = tmp_path / "patient_intake.py"
    src.write_text(
        textwrap.dedent(
            """
            from genai_platform import workflow

            @workflow(
                name="patient_intake",
                api_path="/patient-assistant",
                response_mode="sync",
                min_replicas=2,
                max_replicas=20,
                target_cpu_percent=60,
                cpu="1",
                memory="2Gi",
                timeout_seconds=15,
                max_retries=2,
            )
            def handle(question: str, patient_id: str) -> dict:
                return {"patient_id": patient_id, "answer": question}
            """
        ).strip()
    )
    return src


@pytest.fixture
def empty_module(tmp_path: Path) -> Path:
    src = tmp_path / "no_workflows.py"
    src.write_text("def hello():\n    return 1\n")
    return src


@pytest.fixture
def two_workflow_module(tmp_path: Path) -> Path:
    src = tmp_path / "two_workflows.py"
    src.write_text(
        textwrap.dedent(
            """
            from genai_platform import workflow

            @workflow(name="a", api_path="/a")
            def fa(): return {}

            @workflow(name="b", api_path="/b")
            def fb(): return {}
            """
        ).strip()
    )
    return src


# ---- Scanning --------------------------------------------------------------


class TestScan:
    def test_finds_a_single_workflow(self, workflow_module):
        items = scan_for_workflows(workflow_module)
        assert len(items) == 1
        meta = items[0].metadata
        assert meta["name"] == "patient_intake"
        assert meta["api_path"] == "/patient-assistant"
        assert meta["min_replicas"] == 2
        assert meta["target_cpu_percent"] == 60

    def test_finds_no_workflows(self, empty_module):
        assert scan_for_workflows(empty_module) == []

    def test_finds_multiple_workflows(self, two_workflow_module):
        items = scan_for_workflows(two_workflow_module)
        names = sorted(i.metadata["name"] for i in items)
        assert names == ["a", "b"]


# ---- Dockerfile generation -------------------------------------------------


class TestDockerfile:
    def test_contains_required_layers(self, workflow_module):
        items = scan_for_workflows(workflow_module)
        text = generate_dockerfile(items[0])
        # Base image + runtime entrypoint per Listing 8.24.
        assert "FROM python:" in text
        assert "RUN pip install" in text
        assert "CMD" in text and "genai_platform.runtime.server" in text
        assert "WORKFLOW_NAME=patient_intake" in text


# ---- Kubernetes manifest generation ----------------------------------------


class TestKubernetesManifests:
    def test_deployment_yaml_uses_workflow_metadata(self, workflow_module):
        items = scan_for_workflows(workflow_module)
        manifests = generate_kubernetes_manifests(items[0])

        assert "Deployment.yaml" in manifests
        deployment = yaml.safe_load(manifests["Deployment.yaml"])
        assert deployment["kind"] == "Deployment"
        assert deployment["metadata"]["name"] == "patient_intake"
        assert deployment["spec"]["replicas"] == 2  # min_replicas
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        assert container["resources"]["limits"]["cpu"] == "1"
        assert container["resources"]["limits"]["memory"] == "2Gi"
        # Container env should pin WORKFLOW_NAME and gateway URL.
        env_names = {e["name"] for e in container["env"]}
        assert {"WORKFLOW_NAME", "GENAI_GATEWAY_URL"}.issubset(env_names)

    def test_hpa_yaml_uses_workflow_metadata(self, workflow_module):
        items = scan_for_workflows(workflow_module)
        manifests = generate_kubernetes_manifests(items[0])

        hpa = yaml.safe_load(manifests["HorizontalPodAutoscaler.yaml"])
        assert hpa["kind"] == "HorizontalPodAutoscaler"
        assert hpa["spec"]["minReplicas"] == 2
        assert hpa["spec"]["maxReplicas"] == 20
        cpu_metric = hpa["spec"]["metrics"][0]
        assert cpu_metric["resource"]["target"]["averageUtilization"] == 60

    def test_service_yaml_routes_to_workflow(self, workflow_module):
        items = scan_for_workflows(workflow_module)
        manifests = generate_kubernetes_manifests(items[0])

        service = yaml.safe_load(manifests["Service.yaml"])
        assert service["kind"] == "Service"
        assert service["metadata"]["name"] == "patient_intake"
