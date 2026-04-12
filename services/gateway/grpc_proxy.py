"""
gRPC Proxy - Routes internal gRPC calls to platform services.

Routes based on x-target-service metadata from request context.
Maintains Protocol Buffer efficiency by forwarding binary data.
"""

from typing import Dict, Optional

import grpc

from proto import models_pb2_grpc, sessions_pb2_grpc
from services.gateway.registry import ServiceRegistry


class GenericProxy:
    """
    Generic proxy that routes gRPC calls to platform services.

    Routes based on x-target-service metadata from request context.
    Maintains Protocol Buffer efficiency by forwarding binary data.
    """

    def __init__(self, registry: ServiceRegistry):
        self.registry = registry
        # Map service names to their stub factories
        self._stub_factories: Dict[str, callable] = {
            "sessions": lambda channel: sessions_pb2_grpc.SessionServiceStub(channel),
            "models": lambda channel: models_pb2_grpc.ModelServiceStub(channel),
        }

    def _extract_target_service(self, context) -> Optional[str]:
        """Extract target service from gRPC metadata."""
        metadata = dict(context.invocation_metadata())
        return metadata.get("x-target-service")

    def _forward_request(self, service_name: str, stub_factory, method_name: str, request, context):
        """Forward request to backend service."""
        try:
            backend_addr = self.registry.get_platform_service_address(service_name)
            channel = grpc.insecure_channel(backend_addr)
            stub = stub_factory(channel)

            # Call the method on the backend stub
            method = getattr(stub, method_name)
            response = method(request)

            # Check if response is streaming (iterator/generator)
            if hasattr(response, "__iter__") and not isinstance(response, (str, bytes)):
                # For streaming responses, yield from the iterator
                # and close channel when done
                def stream_with_cleanup():
                    try:
                        yield from response
                    finally:
                        channel.close()

                return stream_with_cleanup()
            else:
                # For unary responses, close immediately
                channel.close()
                return response

        except grpc.RpcError as e:
            context.set_code(e.code())
            context.set_details(e.details())
            raise
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise


class GenericServiceProxy:
    """
    Generic proxy handler that automatically forwards all methods.

    Uses __getattribute__ to intercept ALL attribute access (including method checks)
    and forward method calls to the backend. This works even when gRPC checks
    for method existence before calling.
    """

    def __init__(self, proxy: GenericProxy):
        self.proxy = proxy
        # Cache for actual attributes to avoid infinite recursion
        self._proxy = proxy

    def __getattribute__(self, name: str):
        """
        Intercept ALL attribute access to handle gRPC method checks.

        When gRPC checks for method existence or calls a method, this intercepts
        it and returns a handler function that forwards to the backend.
        """
        # First, check if it's an actual attribute (to avoid infinite recursion)
        if name.startswith("_") or name in ("proxy", "_proxy"):
            return super().__getattribute__(name)

        # Get the proxy instance (using super to avoid recursion)
        proxy = super().__getattribute__("_proxy")

        # If it's a method call (not a check), return a handler
        def handler(request, context):
            # Extract target service from metadata
            target_service = proxy._extract_target_service(context)
            if not target_service:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Missing x-target-service metadata")
                return None

            # Get stub factory for the target service
            stub_factory = proxy._stub_factories.get(target_service)
            if not stub_factory:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Service '{target_service}' not found")
                return None

            # Forward the request
            return proxy._forward_request(target_service, stub_factory, name, request, context)

        return handler


# Proxy handlers for platform services
#
# Why do we need these? Because gRPC's add_*_Servicer_to_server functions
# expect a servicer instance that inherits from the generated Servicer class.
# We combine GenericServiceProxy (for automatic forwarding) with the specific
# Servicer class (to satisfy gRPC's interface requirements).
#
# All methods are automatically forwarded via GenericServiceProxy.__getattr__


class SessionServiceProxy(GenericServiceProxy, sessions_pb2_grpc.SessionServiceServicer):
    """
    Proxy handler for Session Service.

    Inherits from GenericServiceProxy (automatic forwarding) and
    SessionServiceServicer (gRPC interface requirement).

    All methods automatically forwarded - no manual implementation needed!
    """

    pass


class ModelServiceProxy(GenericServiceProxy, models_pb2_grpc.ModelServiceServicer):
    """
    Proxy handler for Model Service.

    Inherits from GenericServiceProxy (automatic forwarding) and
    ModelServiceServicer (gRPC interface requirement).

    All methods automatically forwarded - no manual implementation needed!
    """

    pass
