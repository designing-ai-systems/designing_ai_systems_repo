"""
Base client class for platform service clients.

All service clients follow the same pattern:
- Connect to gateway via gRPC
- Wrap the channel with the RetryInterceptor (Listing 8.14) so every
  outgoing call inherits exponential-backoff retries on transient failures
  (UNAVAILABLE, DEADLINE_EXCEEDED, RESOURCE_EXHAUSTED).
- Use x-target-service metadata for routing
- Handle Protocol Buffer serialization
"""

import os
from typing import Tuple

import grpc

from genai_platform.grpc_retry import RetryInterceptor


def _use_insecure_channel(gateway_url: str) -> bool:
    """Return True if we should connect with a plaintext gRPC channel.

    Plaintext is used when:
    - the URL points at the local loopback (``localhost``, ``127.0.0.1``), or
    - the env var ``GENAI_GATEWAY_INSECURE=1`` is set — this is how
      compose-network services and locally-launched workflow containers
      opt out of TLS, since the in-cluster gateway is plain gRPC.

    Production deployments leave ``GENAI_GATEWAY_INSECURE`` unset and use
    a public hostname, so the secure-channel path is the default.
    """
    if gateway_url.startswith(("localhost", "127.0.0.1")):
        return True
    return os.environ.get("GENAI_GATEWAY_INSECURE", "0") == "1"


class BaseClient:
    """
    Base class for all platform service clients.

    Provides common functionality for:
    - gRPC channel management (with retry interceptor — Listing 8.14)
    - Service metadata for routing
    - Connection to API Gateway
    """

    def __init__(self, platform, service_name: str):
        """
        Initialize a service client.

        Args:
            platform: GenAIPlatform instance with gateway configuration
            service_name: Name of the target service (e.g., "sessions", "models")
        """
        self.platform = platform
        self.service_name = service_name

        if _use_insecure_channel(platform.gateway_url):
            raw_channel = grpc.insecure_channel(platform.gateway_url)
        else:
            credentials = grpc.ssl_channel_credentials()
            raw_channel = grpc.secure_channel(platform.gateway_url, credentials)

        # Listing 8.14: every SDK call routes through the retry interceptor.
        self._channel = grpc.intercept_channel(raw_channel, RetryInterceptor())

        # Service-specific metadata for gateway routing
        self._metadata: Tuple[Tuple[str, str], ...] = (("x-target-service", service_name),)

    @property
    def metadata(self) -> Tuple[Tuple[str, str], ...]:
        """Get service routing metadata."""
        return self._metadata
