"""
Route notification used by the Workflow Service to tell the gateway about
a freshly-deployed workflow's address.

After ``DeployWorkflow`` verifies a new container is healthy, it pushes
the ``(api_path, endpoint)`` pair to the gateway via this notifier so
external requests can land on the new container. The gateway holds a
local cache populated by these pushes; on startup it also re-hydrates
via ``WorkflowService.ListRoutes``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import httpx

logger = logging.getLogger(__name__)


class RoutePusher(ABC):
    @abstractmethod
    def push(self, api_path: str, endpoint: str) -> None:
        """Notify the gateway of a new (api_path, endpoint) route."""


class HttpRoutePusher(RoutePusher):
    """Pushes route updates to the gateway's `/__platform/register-route` endpoint."""

    def __init__(self, gateway_http_url: str = "http://localhost:8080"):
        self.gateway_http_url = gateway_http_url

    def push(self, api_path: str, endpoint: str) -> None:
        try:
            httpx.post(
                f"{self.gateway_http_url}/__platform/register-route",
                json={"api_path": api_path, "endpoint": endpoint},
                timeout=5.0,
            )
        except httpx.RequestError as e:
            logger.warning(
                "could not push route %s → %s to gateway %s: %s",
                api_path,
                endpoint,
                self.gateway_http_url,
                e,
            )


class FakeRoutePusher(RoutePusher):
    def __init__(self):
        self.pushed: list[tuple[str, str]] = []

    def push(self, api_path: str, endpoint: str) -> None:
        self.pushed.append((api_path, endpoint))
