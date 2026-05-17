"""
Observability Service — Main entry point.

Collects logs, metrics, traces, and quality scores from every platform
service. Runs on grpc.aio so the ingest path stays cheap and concurrent.

Backend selection:
- Default: in-memory store (data resets on restart).
- Set ``OBSERVABILITY_POSTGRES_DSN=postgresql://...`` to persist
  traces, generations, logs, metrics, scores, and budget alerts.
"""

import logging
import os

from services.observability.service import ObservabilityServiceImpl
from services.observability.store import InMemoryObservabilityStore
from services.shared.server import run_aio_service_main

logger = logging.getLogger(__name__)


def _build_servicer() -> ObservabilityServiceImpl:
    dsn = os.getenv("OBSERVABILITY_POSTGRES_DSN")
    if dsn:
        try:
            from services.observability.postgres_store import PostgresObservabilityStore

            store = PostgresObservabilityStore(dsn)
            logger.info("Using PostgresObservabilityStore (DSN configured)")
            return ObservabilityServiceImpl(store=store)
        except Exception as exc:  # noqa: BLE001 — log + fall back
            logger.warning(
                "Postgres-backed observability requested but unavailable (%s); "
                "falling back to in-memory store.",
                exc,
            )
    return ObservabilityServiceImpl(store=InMemoryObservabilityStore())


def main() -> None:
    """Run the Observability Service server (asyncio + grpc.aio)."""
    run_aio_service_main("observability", _build_servicer)


if __name__ == "__main__":
    main()
