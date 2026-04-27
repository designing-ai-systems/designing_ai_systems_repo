"""
Container deployer used by the Workflow Service's `DeployWorkflow` RPC.

In production this would speak to Kubernetes; for the book demo it shells
out to ``docker run`` against a locally-built image. The seam is an
abstract ``Deployer`` interface so tests can inject a fake.

Book: "Designing AI Systems"
  - Listing 8.23 (Deployment lifecycle messages)
  - Section 8.7 (Deployment pipeline) — local Docker is the demo
    simplification of the K8s rolling-update flow described there.
"""

from __future__ import annotations

import logging
import shutil
import socket
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

# `host.docker.internal` resolves to the host on Docker Desktop (Mac/Win) and
# on Docker for Linux when started with `--add-host`. The gateway runs on the
# host during demos, so containers reach it via this hostname.
DEFAULT_CONTAINER_GATEWAY_URL = "host.docker.internal:50051"
RUNTIME_CONTAINER_PORT = 8000


@dataclass
class DeployResult:
    container_id: str
    host_port: int


class Deployer(ABC):
    @abstractmethod
    def deploy(self, *, image: str, name: str, gateway_url: str) -> DeployResult:
        """Run the container for a workflow. Returns its (container_id, host_port)."""

    @abstractmethod
    def wait_until_ready(self, *, host_port: int, timeout_seconds: float) -> bool:
        """Poll `/health/ready` until it responds 200 or `timeout_seconds` elapses."""

    @abstractmethod
    def stop(self, container_id: str) -> None:
        """Stop and remove a previously-deployed container."""

    def container_logs(self, container_id: str, tail: int = 30) -> str:
        """Optional: return recent container logs for failure diagnostics."""
        return ""


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


class DockerDeployer(Deployer):
    """Spawns workflow containers via the local Docker daemon."""

    def deploy(
        self,
        *,
        image: str,
        name: str,
        gateway_url: str = DEFAULT_CONTAINER_GATEWAY_URL,
    ) -> DeployResult:
        if shutil.which("docker") is None:
            raise RuntimeError("`docker` not on PATH; install Docker to deploy workflows")

        # Idempotent: drop any prior container that bound the well-known name.
        container_name = f"genai-workflow-{name}"
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True,
            text=True,
        )

        host_port = _find_free_port()
        cmd = [
            "docker",
            "run",
            "-d",
            "--name",
            container_name,
            "-p",
            f"{host_port}:{RUNTIME_CONTAINER_PORT}",
            "-e",
            f"WORKFLOW_NAME={name}",
            "-e",
            f"GENAI_GATEWAY_URL={gateway_url}",
            "--add-host",
            "host.docker.internal:host-gateway",
            image,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"docker run failed: {proc.stderr.strip() or proc.stdout.strip()}")
        return DeployResult(container_id=proc.stdout.strip(), host_port=host_port)

    def wait_until_ready(self, *, host_port: int, timeout_seconds: float = 30.0) -> bool:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            try:
                r = httpx.get(f"http://localhost:{host_port}/health/ready", timeout=2.0)
                if r.status_code == 200:
                    return True
            except httpx.RequestError:
                pass
            time.sleep(0.5)
        return False

    def container_logs(self, container_id: str, tail: int = 30) -> str:
        """Return the last ``tail`` lines of a container's stdout+stderr.

        Used by the Workflow Service to surface why a workflow's
        ``/health/ready`` never came up (most often: import error in the
        workflow source file).
        """
        proc = subprocess.run(
            ["docker", "logs", "--tail", str(tail), container_id],
            capture_output=True,
            text=True,
        )
        return (proc.stdout + proc.stderr).strip()

    def stop(self, container_id: str) -> None:
        subprocess.run(["docker", "rm", "-f", container_id], capture_output=True, text=True)


class FakeDeployer(Deployer):
    """Test deployer that records calls and never touches Docker."""

    def __init__(self, host_port: int = 18000):
        self.host_port = host_port
        self.deployed: list[dict] = []
        self.stopped: list[str] = []
        self.ready_returns: bool = True

    def deploy(self, *, image: str, name: str, gateway_url: str) -> DeployResult:
        self.deployed.append({"image": image, "name": name, "gateway_url": gateway_url})
        return DeployResult(container_id=f"fake-{name}", host_port=self.host_port)

    def wait_until_ready(self, *, host_port: int, timeout_seconds: float) -> bool:
        return self.ready_returns

    def stop(self, container_id: str) -> None:
        self.stopped.append(container_id)


class RoutePusher(ABC):
    @abstractmethod
    def push(self, api_path: str, endpoint: str) -> None:
        """Notify the gateway of a new (api_path, endpoint) route."""


class HttpRoutePusher(RoutePusher):
    """Pushes route updates to the gateway's `/__platform/register-route` endpoint."""

    def __init__(self, gateway_http_url: str = "http://localhost:8080"):
        self.gateway_http_url = gateway_http_url

    def push(self, api_path: str, endpoint: str) -> None:
        try:
            httpx.post(
                f"{self.gateway_http_url}/__platform/register-route",
                json={"api_path": api_path, "endpoint": endpoint},
                timeout=5.0,
            )
        except httpx.RequestError as e:
            logger.warning(
                "could not push route %s → %s to gateway %s: %s",
                api_path,
                endpoint,
                self.gateway_http_url,
                e,
            )


class FakeRoutePusher(RoutePusher):
    def __init__(self):
        self.pushed: list[tuple[str, str]] = []

    def push(self, api_path: str, endpoint: str) -> None:
        self.pushed.append((api_path, endpoint))
