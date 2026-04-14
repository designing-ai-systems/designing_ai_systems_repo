"""
Guardrails Service — Main entry point.

Enforces safety policies and compliance checks.
Runs on grpc.aio (Listing 6.19).
"""

from services.guardrails.service import GuardrailsServiceImpl
from services.shared.server import run_aio_service_main


def main():
    """Run the Guardrails Service server (asyncio + grpc.aio)."""
    run_aio_service_main("guardrails", GuardrailsServiceImpl)


if __name__ == "__main__":
    main()
