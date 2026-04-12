"""
Base client class for platform service clients.

All service clients follow the same pattern:
- Connect to gateway via gRPC
- Use x-target-service metadata for routing
- Handle Protocol Buffer serialization
"""

from typing import Tuple

import grpc


class BaseClient:
    """
    Base class for all platform service clients.

    Provides common functionality for:
    - gRPC channel management
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

        # Create gRPC channel to gateway
        # Use insecure channel for localhost/testing, secure for production
        if platform.gateway_url.startswith("localhost") or platform.gateway_url.startswith(
            "127.0.0.1"
        ):
            self._channel = grpc.insecure_channel(platform.gateway_url)
        else:
            credentials = grpc.ssl_channel_credentials()
            self._channel = grpc.secure_channel(platform.gateway_url, credentials)

        # Service-specific metadata for gateway routing
        self._metadata: Tuple[Tuple[str, str], ...] = (("x-target-service", service_name),)

    @property
    def metadata(self) -> Tuple[Tuple[str, str], ...]:
        """Get service routing metadata."""
        return self._metadata
