"""
Shared server utilities for platform services.

Provides common patterns for:
- gRPC server creation (sync and grpc.aio)
- Port configuration
- Server lifecycle management
"""

import asyncio
import os
from concurrent import futures
from typing import Optional

import grpc
import grpc.aio

# Port registry to ensure unique ports across services
SERVICE_PORTS = {
    "sessions": 50052,
    "models": 50053,
    "data": 50054,
    "guardrails": 50055,
    "tools": 50056,
    "workflow": 50058,
    "observability": 50059,
    "experiments": 50060,
}


def get_service_port(service_name: str, default_port: Optional[int] = None) -> int:
    """
    Get port for a service, checking environment variable first, then registry.

    Args:
        service_name: Name of the service (e.g., "sessions")
        default_port: Optional default port if not in registry

    Returns:
        Port number to use
    """
    # Check environment variable first (SESSIONS_PORT, MODELS_PORT, etc.)
    env_var = f"{service_name.upper()}_PORT"
    env_port = os.getenv(env_var)
    if env_port:
        return int(env_port)

    # Check registry
    if service_name in SERVICE_PORTS:
        return SERVICE_PORTS[service_name]

    # Use provided default or raise error
    if default_port:
        return default_port

    raise ValueError(
        f"No port configured for service '{service_name}'. "
        f"Set {env_var} environment variable or add to SERVICE_PORTS registry."
    )


def create_grpc_server(
    servicer: object,
    port: Optional[int] = None,
    service_name: Optional[str] = None,
    max_workers: int = 10,
) -> grpc.Server:
    """
    Create and configure a gRPC server for a platform service.

    Args:
        servicer: Servicer instance that implements BaseServicer
        port: Optional explicit port. If None, uses service_name to look up port
        service_name: Service name for port lookup (required if port is None)
        max_workers: Number of worker threads for the server

    Returns:
        Configured gRPC server (not started)
    """
    # Determine port
    if port is None:
        if service_name is None:
            raise ValueError("Either port or service_name must be provided")
        port = get_service_port(service_name)

    # Create server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))

    # Add servicer (servicer implements add_to_server method)
    servicer.add_to_server(server)

    # Configure listening address
    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)

    return server


def run_service(server: grpc.Server, service_name: str, port: Optional[int] = None):
    """
    Run a service server with proper startup/shutdown handling.

    Args:
        server: gRPC server instance
        service_name: Name of the service (for logging)
        port: Optional port (for logging)
    """
    if port is None:
        port = get_service_port(service_name)

    print(f"Starting {service_name} Service on port {port}")

    server.start()
    print(f"{service_name} Service started. Press Ctrl+C to stop.")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print(f"\nStopping {service_name} Service...")
        server.stop(0)
        print(f"{service_name} Service stopped.")


def create_grpc_aio_server(
    servicer,
    port: Optional[int] = None,
    service_name: Optional[str] = None,
) -> grpc.aio.Server:
    """
    Create a grpc.aio server for async servicers (Tool, Guardrails).

    The servicer must implement ``add_to_aio_server(server)`` (see BaseAioServicer).
    """
    if port is None:
        if service_name is None:
            raise ValueError("Either port or service_name must be provided")
        port = get_service_port(service_name)

    server = grpc.aio.server()
    servicer.add_to_aio_server(server)
    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)
    return server


def run_aio_service_main(service_name: str, servicer_factory):
    """
    Entry point for asyncio-based services: ``asyncio.run(_main())``.

    ``servicer_factory`` is a zero-argument callable returning the servicer instance.
    """

    async def _main() -> None:
        port = get_service_port(service_name)
        servicer = servicer_factory()
        server = create_grpc_aio_server(servicer=servicer, port=port)
        print(f"Starting {service_name} Service (grpc.aio) on port {port}")
        await server.start()
        print(f"{service_name} Service started. Press Ctrl+C to stop.")
        try:
            await server.wait_for_termination()
        finally:
            await server.stop(grace=5.0)
            print(f"{service_name} Service stopped.")

    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        print(f"\nStopping {service_name} Service...")
