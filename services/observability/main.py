"""
Observability Service — Main entry point.

Collects logs, metrics, traces, and quality scores from every platform
service. Runs on grpc.aio so the ingest path stays cheap and concurrent.
"""

from services.observability.service import ObservabilityServiceImpl
from services.shared.server import run_aio_service_main


def main() -> None:
    """Run the Observability Service server (asyncio + grpc.aio)."""
    run_aio_service_main("observability", ObservabilityServiceImpl)


if __name__ == "__main__":
    main()
