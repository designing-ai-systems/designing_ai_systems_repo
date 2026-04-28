"""
Docker runner used by ``genai-platform deploy`` to launch workflow containers.

The CLI runs on the developer's host (where Docker already lives) and is
the thing that actually invokes ``docker run`` / ``docker rm``. The
Workflow Service is bookkeeping-only: it stores the spec, verifies the
container is healthy after the CLI launches it, and pushes the route to
the gateway.

In production (per chapter 8) the Workflow Service uses the Kubernetes
API to deploy workflows, with a scoped service account. We don't have a
Kubernetes cluster in the local-Docker demo, and giving the Workflow
Service direct Docker access (via a mounted ``/var/run/docker.sock``)
would grant it effectively root on the host with no scoping. So for the
demo we keep ``docker run`` on the developer's CLI — which already has
trusted Docker access — and the Workflow Service stays a pure
bookkeeping service. The chapter's architecture is unchanged; this is
just where the demo wires the docker action.

Two operating modes, picked by the ``WORKFLOW_DOCKER_NETWORK`` env var:

- **Host mode** (default — used when running the platform services as
  bare ``python -m services.X.main`` processes): pick a free host port,
  map the container's :8000 to it, endpoint is ``localhost:<host_port>``.
- **Compose-network mode** (used when the platform runs under
  ``docker compose up``): attach new containers to the same compose
  network so the gateway reaches them by DNS name. Endpoint is
  ``<container_name>:8000``; no host port mapping needed.
"""

from __future__ import annotations

import logging
import os
import shutil
import socket
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# `host.docker.internal` resolves to the host on Docker Desktop (Mac/Win) and
# on Docker for Linux when started with `--add-host`. Used in host mode.
DEFAULT_CONTAINER_GATEWAY_URL = "host.docker.internal:50051"
RUNTIME_CONTAINER_PORT = 8000


@dataclass
class DeployResult:
    container_id: str
    endpoint: str  # e.g., "localhost:55512" (host mode) or "genai-workflow-foo:8000" (compose)


class Deployer(ABC):
    @abstractmethod
    def deploy(self, *, image: str, name: str, gateway_url: str) -> DeployResult:
        """Run the container for a workflow. Returns its (container_id, endpoint)."""

    @abstractmethod
    def stop(self, container_id: str) -> None:
        """Stop and remove a previously-deployed container."""

    def container_logs(self, container_id: str, tail: int = 30) -> str:
        """Optional: return recent container logs for failure diagnostics."""
        return ""


DEFAULT_COMPOSE_NETWORK = "genai-platform_default"


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


def _detect_compose_network() -> str | None:
    """Return the compose network name if the platform stack is up, else None.

    Lets the CLI Just Work whether the user is running platform services
    via ``docker compose up`` or via bare ``python -m services.X.main``.
    Explicit ``WORKFLOW_DOCKER_NETWORK`` overrides this detection.
    """
    if shutil.which("docker") is None:
        return None
    proc = subprocess.run(
        ["docker", "network", "inspect", DEFAULT_COMPOSE_NETWORK],
        capture_output=True,
        text=True,
    )
    return DEFAULT_COMPOSE_NETWORK if proc.returncode == 0 else None


class DockerDeployer(Deployer):
    """Spawns workflow containers via the local Docker daemon.

    Host vs. compose mode is picked by ``WORKFLOW_DOCKER_NETWORK``: set it
    to a docker-network name (e.g., ``genai-platform_default``) to run new
    containers on that network and address them by container name; leave
    it unset to fall back to host mode (port mapping + ``localhost``).
    """

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

        compose_network = os.environ.get("WORKFLOW_DOCKER_NETWORK") or _detect_compose_network()
        cmd = [
            "docker",
            "run",
            "-d",
            "--name",
            container_name,
            "-e",
            f"WORKFLOW_NAME={name}",
            "-e",
            f"GENAI_GATEWAY_URL={gateway_url}",
        ]
        if compose_network:
            cmd += [
                "--network",
                compose_network,
                # In-cluster gRPC is plaintext; the SDK inside this container
                # would otherwise default to TLS for any non-localhost URL.
                "-e",
                "GENAI_GATEWAY_INSECURE=1",
            ]
            endpoint = f"{container_name}:{RUNTIME_CONTAINER_PORT}"
        else:
            host_port = _find_free_port()
            cmd += [
                "-p",
                f"{host_port}:{RUNTIME_CONTAINER_PORT}",
                "--add-host",
                "host.docker.internal:host-gateway",
            ]
            endpoint = f"localhost:{host_port}"
        cmd.append(image)

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"docker run failed: {proc.stderr.strip() or proc.stdout.strip()}")
        return DeployResult(container_id=proc.stdout.strip(), endpoint=endpoint)

    def container_logs(self, container_id: str, tail: int = 30) -> str:
        """Return the last ``tail`` lines of a container's stdout+stderr.

        Used by the CLI to surface why a workflow's ``/health/ready`` never
        came up (most often: import error in the workflow source file).
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

    def __init__(self, endpoint: str = "fake-host:18000"):
        self.endpoint = endpoint
        self.deployed: list[dict] = []
        self.stopped: list[str] = []

    def deploy(self, *, image: str, name: str, gateway_url: str) -> DeployResult:
        self.deployed.append({"image": image, "name": name, "gateway_url": gateway_url})
        return DeployResult(container_id=f"fake-{name}", endpoint=self.endpoint)

    def stop(self, container_id: str) -> None:
        self.stopped.append(container_id)
