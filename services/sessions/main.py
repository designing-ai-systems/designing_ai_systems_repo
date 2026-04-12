"""
Session Service - Main entry point.

Manages conversation sessions and user state.
"""

from services.sessions.service import SessionService
from services.shared.server import create_grpc_server, get_service_port, run_service


def main():
    """Run the Session Service server."""
    service_name = "sessions"
    port = get_service_port(service_name)

    # Create server with SessionService
    servicer = SessionService()
    server = create_grpc_server(servicer=servicer, port=port, service_name=service_name)

    # Run the service
    run_service(server, service_name, port)


if __name__ == "__main__":
    main()
