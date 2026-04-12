"""
Base class for service servicers.

Provides common functionality that all service implementations can use.
"""

from abc import ABC, abstractmethod

import grpc


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
