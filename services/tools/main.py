"""
Tool Service — Main entry point.

Manages tool registration, discovery, and execution.
Runs on grpc.aio so handlers can await CredentialStore (Listing 6.14).
"""

from services.shared.server import run_aio_service_main
from services.tools.service import ToolServiceImpl


def main():
    """Run the Tool Service server (asyncio + grpc.aio)."""
    run_aio_service_main("tools", ToolServiceImpl)


if __name__ == "__main__":
    main()
