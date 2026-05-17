"""
Experimentation Service — Main entry point.

Manages target lifecycle, datasets, offline evaluation, online scoring,
A/B testing, and human annotation queues. Reaches out to the
Observability Service over gRPC for trace/score lookups when those
features are exercised.

Backend selection:
- Default: in-memory store (data resets on restart).
- Set ``EXPERIMENTS_POSTGRES_DSN=postgresql://...`` to persist targets,
  datasets, evaluations, scoring rules, experiments, assignments,
  outcomes, and annotation queues.
"""

from __future__ import annotations

import logging
import os

import grpc

from proto import observability_pb2_grpc
from services.experiments.service import ExperimentationServiceImpl
from services.experiments.store import InMemoryExperimentStore
from services.shared.server import run_aio_service_main

logger = logging.getLogger(__name__)


def _make_servicer() -> ExperimentationServiceImpl:
    obs_addr = os.getenv("OBSERVABILITY_SERVICE_ADDR", "")
    obs_stub = None
    if obs_addr:
        channel = grpc.insecure_channel(obs_addr)
        obs_stub = observability_pb2_grpc.ObservabilityServiceStub(channel)

    dsn = os.getenv("EXPERIMENTS_POSTGRES_DSN")
    store = None
    if dsn:
        try:
            from services.experiments.postgres_store import PostgresExperimentStore

            store = PostgresExperimentStore(dsn)
            logger.info("Using PostgresExperimentStore (DSN configured)")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Postgres-backed experiments requested but unavailable (%s); "
                "falling back to in-memory store.",
                exc,
            )
            store = InMemoryExperimentStore()
    return ExperimentationServiceImpl(store=store, observability_stub=obs_stub)


def main() -> None:
    """Run the Experimentation Service server (asyncio + grpc.aio)."""
    run_aio_service_main("experiments", _make_servicer)


if __name__ == "__main__":
    main()
