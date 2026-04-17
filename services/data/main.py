"""
Data Service - Main entry point.

Manages document ingestion, vector storage, and semantic search.
"""

from services.data.service import DataService
from services.shared.server import create_grpc_server, get_service_port, run_service


def main():
    """Run the Data Service server."""
    service_name = "data"
    port = get_service_port(service_name)

    servicer = DataService()
    server = create_grpc_server(servicer=servicer, port=port, service_name=service_name)

    run_service(server, service_name, port)


if __name__ == "__main__":
    main()
