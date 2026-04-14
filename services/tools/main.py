"""
Tool Service — Main entry point.

Manages tool registration, discovery, and execution.
"""

from services.shared.server import create_grpc_server, get_service_port, run_service
from services.tools.service import ToolServiceImpl


def main():
    """Run the Tool Service server."""
    service_name = "tools"
    port = get_service_port(service_name)
    servicer = ToolServiceImpl()
    server = create_grpc_server(servicer=servicer, port=port, service_name=service_name)
    run_service(server, service_name, port)


if __name__ == "__main__":
    main()
