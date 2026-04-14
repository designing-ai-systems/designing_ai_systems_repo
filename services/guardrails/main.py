"""
Guardrails Service — Main entry point.

Enforces safety policies and compliance checks.
"""

from services.guardrails.service import GuardrailsServiceImpl
from services.shared.server import create_grpc_server, get_service_port, run_service


def main():
    """Run the Guardrails Service server."""
    service_name = "guardrails"
    port = get_service_port(service_name)
    servicer = GuardrailsServiceImpl()
    server = create_grpc_server(servicer=servicer, port=port, service_name=service_name)
    run_service(server, service_name, port)


if __name__ == "__main__":
    main()
