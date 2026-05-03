"""
Model Service - Main entry point.

Handles model inference and model management operations.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import grpc

from proto import observability_pb2_grpc
from services.models.service import ModelService
from services.shared.observability_client import ObservabilityClient
from services.shared.server import create_grpc_server, get_service_port, run_service


def load_env_file(env_path: Path) -> None:
    """Load environment variables from a local .env file if present."""
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _build_observability_client() -> Optional[ObservabilityClient]:
    """Wire an ObservabilityClient if OBSERVABILITY_SERVICE_ADDR is set."""
    addr = os.getenv("OBSERVABILITY_SERVICE_ADDR")
    if not addr:
        return None
    channel = grpc.insecure_channel(addr)
    stub = observability_pb2_grpc.ObservabilityServiceStub(channel)
    return ObservabilityClient(stub=stub, service_name="models")


def main():
    """Run the Model Service server."""
    project_root = Path(__file__).resolve().parents[2]
    load_env_file(project_root / ".env")

    service_name = "models"
    port = get_service_port(service_name)

    servicer = ModelService(observability=_build_observability_client())
    server = create_grpc_server(
        servicer=servicer,
        port=port,
        service_name=service_name,
    )

    run_service(server, service_name, port)


if __name__ == "__main__":
    main()
