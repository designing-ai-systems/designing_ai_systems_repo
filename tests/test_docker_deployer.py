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

from services.workflow import deployer as deployer_mod
from services.workflow.deployer import DockerDeployer

CompletedProcessLike = namedtuple("CompletedProcessLike", ("returncode", "stdout", "stderr"))


def _patch_docker_runs(monkeypatch, *, container_id: str = "abc123"):
    """Capture every subprocess.run call into ``calls`` and pretend `docker run` succeeded."""
    calls: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        return CompletedProcessLike(returncode=0, stdout=container_id + "\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(deployer_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(deployer_mod.shutil, "which", lambda _: "/usr/local/bin/docker")
    return calls


class TestHostMode:
    def test_picks_host_port_and_uses_localhost_endpoint(self, monkeypatch):
        monkeypatch.delenv("WORKFLOW_DOCKER_NETWORK", raising=False)
        calls = _patch_docker_runs(monkeypatch)

        result = DockerDeployer().deploy(
            image="genai-workflow/x:latest",
            name="x",
            gateway_url="host.docker.internal:50051",
        )

        # The second subprocess call is the actual `docker run`.
        run_cmd = calls[1]
        assert run_cmd[:3] == ["docker", "run", "-d"]
        # `-p HOST:CONTAINER` flag present, host port is non-zero.
        assert any(arg.startswith("-p") or "/" not in arg for arg in run_cmd)
        port_idx = run_cmd.index("-p")
        host_port = int(run_cmd[port_idx + 1].split(":")[0])
        assert host_port > 0
        # No --network flag.
        assert "--network" not in run_cmd
        # Endpoint reflects host port.
        assert result.endpoint == f"localhost:{host_port}"
        assert result.container_id == "abc123"


class TestComposeMode:
    def test_attaches_to_compose_network_and_uses_container_name_endpoint(self, monkeypatch):
        monkeypatch.setenv("WORKFLOW_DOCKER_NETWORK", "genai-platform_default")
        calls = _patch_docker_runs(monkeypatch)

        result = DockerDeployer().deploy(
            image="genai-workflow/x:latest",
            name="x",
            gateway_url="gateway:50051",
        )

        run_cmd = calls[1]
        # --network attaches the new container to the compose network.
        assert "--network" in run_cmd
        net_idx = run_cmd.index("--network")
        assert run_cmd[net_idx + 1] == "genai-platform_default"
        # No host port mapping in compose mode.
        assert "-p" not in run_cmd
        # No host.docker.internal hack — container reaches the gateway by DNS.
        assert "host.docker.internal:host-gateway" not in run_cmd
        # Endpoint is the container's stable name + the runtime port.
        assert result.endpoint == "genai-workflow-x:8000"
        # Gateway URL is plumbed through to the container's env.
        env_args = [a for i, a in enumerate(run_cmd) if i > 0 and run_cmd[i - 1] == "-e"]
        assert "GENAI_GATEWAY_URL=gateway:50051" in env_args


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
