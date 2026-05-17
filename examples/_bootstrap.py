"""
Shared helper for example scripts.

Each example traditionally boots its own in-process Gateway + platform
services so that ``python examples/quickstart_*.py`` is one command.
That collides with ``docker compose up``, which already binds the
gateway's host ports (50051 for gRPC, 8080 for HTTP). The OSError
from the second binder is noisy and confusing for readers who
followed the README's Docker quick-start.

``start_services_unless_running`` probes ``localhost:50051`` first.
If something is already listening there (Docker, a different terminal,
whatever), it skips the in-process startup and lets the example talk
to the existing stack through the gateway.
"""

from __future__ import annotations

import socket
import threading
import time
from typing import Callable, Iterable


def gateway_reachable(host: str = "localhost", port: int = 50051, timeout: float = 0.5) -> bool:
    """Return True when something is accepting TCP connections at ``host:port``."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def start_services_unless_running(
    starters: Iterable[Callable[[], None]],
    *,
    wait_seconds: float = 1.0,
    gateway_host: str = "localhost",
    gateway_port: int = 50051,
) -> bool:
    """Start each callable in a daemon thread unless the gateway is already up.

    Returns True when the example started its own in-process services,
    False when it detected a running gateway and skipped startup.
    """
    if gateway_reachable(host=gateway_host, port=gateway_port):
        print(
            f"Detected an existing Gateway at {gateway_host}:{gateway_port} — "
            f"skipping in-process service startup."
        )
        return False
    for starter in starters:
        threading.Thread(
            target=starter, daemon=True, name=getattr(starter, "__name__", "service")
        ).start()
        time.sleep(wait_seconds)
    return True
