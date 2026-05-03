"""Tests for ``DockerDeployer``'s host-mode vs compose-mode logic.

We don't actually shell out to ``docker run`` here — that's covered by
the opt-in DOCKER_INTEGRATION test in ``test_workflow_deploy_cli.py``
and the live demo flow. These tests just monkeypatch ``subprocess.run``
to capture the command the deployer would have run, and assert the
right flags get passed for each mode.
"""

from __future__ import annotations

import subprocess
from collections import namedtuple

import pytest

from genai_platform.cli import docker_runner as deployer_mod
from genai_platform.cli.docker_runner import DockerDeployer

CompletedProcessLike = namedtuple("CompletedProcessLike", ("returncode", "stdout", "stderr"))


def _patch_docker_runs(
    monkeypatch,
    *,
    container_id: str = "abc123",
    compose_network_exists: bool = False,
):
    """Capture every subprocess.run call into ``calls`` and script docker responses.

    `docker network inspect` returns rc=0 iff ``compose_network_exists``;
    everything else (rm, run, etc.) succeeds and emits the canned container
    id on stdout.
    """
    calls: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        if list(cmd[:3]) == ["docker", "network", "inspect"]:
            rc = 0 if compose_network_exists else 1
            return CompletedProcessLike(returncode=rc, stdout="", stderr="")
        return CompletedProcessLike(returncode=0, stdout=container_id + "\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(deployer_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(deployer_mod.shutil, "which", lambda _: "/usr/local/bin/docker")
    return calls


def _docker_run_cmd(calls: list[list[str]]) -> list[str]:
    """Pick the actual ``docker run`` invocation out of the recorded calls."""
    for cmd in calls:
        if cmd[:3] == ["docker", "run", "-d"]:
            return cmd
    raise AssertionError(f"no docker-run call in {calls}")


class TestHostMode:
    def test_picks_host_port_and_uses_localhost_endpoint(self, monkeypatch):
        """No env var, no compose network detected → host mode."""
        monkeypatch.delenv("WORKFLOW_DOCKER_NETWORK", raising=False)
        calls = _patch_docker_runs(monkeypatch, compose_network_exists=False)

        result = DockerDeployer().deploy(
            image="genai-workflow/x:latest",
            name="x",
            gateway_url="host.docker.internal:50051",
        )

        run_cmd = _docker_run_cmd(calls)
        port_idx = run_cmd.index("-p")
        host_port = int(run_cmd[port_idx + 1].split(":")[0])
        assert host_port > 0
        assert "--network" not in run_cmd
        assert result.endpoint == f"localhost:{host_port}"
        assert result.container_id == "abc123"


class TestComposeMode:
    def test_explicit_env_var_picks_compose_mode(self, monkeypatch):
        monkeypatch.setenv("WORKFLOW_DOCKER_NETWORK", "genai-platform_default")
        calls = _patch_docker_runs(monkeypatch)

        result = DockerDeployer().deploy(
            image="genai-workflow/x:latest",
            name="x",
            gateway_url="gateway:50051",
        )

        run_cmd = _docker_run_cmd(calls)
        assert "--network" in run_cmd
        assert run_cmd[run_cmd.index("--network") + 1] == "genai-platform_default"
        assert "-p" not in run_cmd
        assert "host.docker.internal:host-gateway" not in run_cmd
        assert result.endpoint == "genai-workflow-x:8000"
        env_args = [a for i, a in enumerate(run_cmd) if i > 0 and run_cmd[i - 1] == "-e"]
        assert "GENAI_GATEWAY_URL=gateway:50051" in env_args
        # Compose-network gRPC is plaintext; container must opt out of TLS
        # or the SDK inside it would default to a secure channel.
        assert "GENAI_GATEWAY_INSECURE=1" in env_args

    def test_auto_detected_compose_network(self, monkeypatch):
        """No env var, but `docker network inspect genai-platform_default` says
        the network exists → CLI auto-detects and uses compose mode."""
        monkeypatch.delenv("WORKFLOW_DOCKER_NETWORK", raising=False)
        calls = _patch_docker_runs(monkeypatch, compose_network_exists=True)

        result = DockerDeployer().deploy(
            image="genai-workflow/x:latest",
            name="x",
            gateway_url="gateway:50051",
        )

        run_cmd = _docker_run_cmd(calls)
        assert "--network" in run_cmd
        assert run_cmd[run_cmd.index("--network") + 1] == "genai-platform_default"
        assert result.endpoint == "genai-workflow-x:8000"


class TestStopAndCleanup:
    def test_stop_removes_by_id(self, monkeypatch):
        calls = _patch_docker_runs(monkeypatch)
        DockerDeployer().stop("abc123")
        assert calls == [["docker", "rm", "-f", "abc123"]]


class TestDockerNotInstalled:
    def test_deploy_raises_when_docker_not_on_path(self, monkeypatch):
        monkeypatch.delenv("WORKFLOW_DOCKER_NETWORK", raising=False)
        monkeypatch.setattr(deployer_mod.shutil, "which", lambda _: None)
        with pytest.raises(RuntimeError, match="not on PATH"):
            DockerDeployer().deploy(image="x", name="x", gateway_url="g:1")
