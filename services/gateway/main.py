"""
API Gateway - Single entry point for all platform traffic.

The gateway serves two purposes (as described in Chapter 2):
1. External HTTP traffic → Routes to workflow containers based on API paths
2. Internal gRPC traffic → Routes to platform services based on x-target-service metadata

This is a single gateway component that handles both types of traffic.
Internally, it runs two servers (HTTP and gRPC) but presents a unified gateway interface.
"""

import os
import threading

from services.gateway.registry import ServiceRegistry
from services.gateway.servers import create_grpc_server, create_http_server


def main():
    """
    Run the API Gateway.

    The gateway is a single component that handles both external HTTP requests
    and internal gRPC calls. It runs two servers internally:
    - HTTP server for external clients → workflows
    - gRPC server for internal workflows → platform services
    """
    registry = ServiceRegistry()

    # Register platform services from environment variables
    sessions_addr = os.getenv("SESSIONS_SERVICE_ADDR", "localhost:50052")
    registry.register_platform_service("sessions", sessions_addr)
    models_addr = os.getenv("MODELS_SERVICE_ADDR", "localhost:50053")
    registry.register_platform_service("models", models_addr)
    guardrails_addr = os.getenv("GUARDRAILS_SERVICE_ADDR", "localhost:50055")
    registry.register_platform_service("guardrails", guardrails_addr)
    tools_addr = os.getenv("TOOLS_SERVICE_ADDR", "localhost:50056")
    registry.register_platform_service("tools", tools_addr)

    # Register workflows (in production, this would come from Workflow Service)
    # For now, we can register manually for testing
    # registry.register_workflow("/patient-assistant", "localhost:8000")

    # Ports
    http_port = int(os.getenv("GATEWAY_HTTP_PORT", "8080"))
    grpc_port = int(os.getenv("GATEWAY_PORT", "50051"))

    print("Starting API Gateway (single component, two purposes)")
    print(f"  External HTTP (clients → workflows): port {http_port}")
    print(f"  Internal gRPC (workflows → platform services): port {grpc_port}")
    print("\nGateway routes:")
    print("  - HTTP requests to workflows based on API path")
    print("  - gRPC requests to platform services based on x-target-service metadata")

    # Start HTTP server in a separate thread
    http_server = create_http_server(registry, http_port)
    http_thread = threading.Thread(target=http_server.serve_forever, daemon=True)
    http_thread.start()
    print(f"HTTP server started on port {http_port}")

    # Start gRPC server
    grpc_server = create_grpc_server(registry, grpc_port)
    grpc_server.start()
    print(f"gRPC server started on port {grpc_port}")

    print("\nGateway started. Press Ctrl+C to stop.")
    try:
        grpc_server.wait_for_termination()
    except KeyboardInterrupt:
        print("\nStopping gateway...")
        http_server.shutdown()
        grpc_server.stop(0)


if __name__ == "__main__":
    main()
