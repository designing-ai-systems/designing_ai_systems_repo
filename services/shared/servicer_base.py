"""
Base class for service servicers.

Provides common functionality that all service implementations can use.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import grpc

if TYPE_CHECKING:
    import grpc.aio


class BaseServicer(ABC):
    """
    Base class for platform service servicers.

    All service implementations should inherit from this and implement
    the add_to_server method.
    """

    @abstractmethod
    def add_to_server(self, server: grpc.Server):
        """
        Add this servicer to a gRPC server.

        This method should call the appropriate add_*_Servicer_to_server
        function from the generated proto code.

        Args:
            server: gRPC server instance
        """
        pass


class BaseAioServicer(ABC):
    """
    Base class for async (grpc.aio) platform servicers.

    Matches the book's async patterns for Tool and Guardrails services
    (e.g. await credential_store.retrieve in Listing 6.14).
    """

    @abstractmethod
    def add_to_aio_server(self, server: "grpc.aio.Server") -> None:
        """
        Register this servicer on a grpc.aio.Server.

        Implementations call the generated add_*_Servicer_to_server(self, server).
        """
        pass
