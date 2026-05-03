"""
Experimentation Service — Main entry point.

Manages target lifecycle, datasets, offline evaluation, online scoring,
A/B testing, and human annotation queues. Reaches out to the
Observability Service over gRPC for trace/score lookups when those
features are exercised.
"""

from __future__ import annotations

import os

import grpc

from proto import observability_pb2_grpc
from services.experiments.service import ExperimentationServiceImpl
from services.shared.server import run_aio_service_main


def _make_servicer() -> ExperimentationServiceImpl:
    obs_addr = os.getenv("OBSERVABILITY_SERVICE_ADDR", "")
    obs_stub = None
    if obs_addr:
        channel = grpc.insecure_channel(obs_addr)
        obs_stub = observability_pb2_grpc.ObservabilityServiceStub(channel)
    return ExperimentationServiceImpl(observability_stub=obs_stub)


def main() -> None:
    """Run the Experimentation Service server (asyncio + grpc.aio)."""
    run_aio_service_main("experiments", _make_servicer)


if __name__ == "__main__":
    main()
