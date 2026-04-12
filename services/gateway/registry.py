"""
Service Registry - Tracks available backend services and workflows.

The registry maintains mappings for:
- Platform services (sessions, models, data, etc.) for internal gRPC routing
- Workflow services (user-defined workflows) for external HTTP routing
"""

from typing import Dict, List


class ServiceRegistry:
    """
    Registry of available backend services.

    Tracks both:
    - Platform services (sessions, models, data, etc.) for internal gRPC routing
    - Workflow services (user-defined workflows) for external HTTP routing
    """

    def __init__(self):
        # Platform services: service_name -> [addresses]
        self._platform_services: Dict[str, List[str]] = {}
        # Workflows: api_path -> [addresses]
        self._workflows: Dict[str, List[str]] = {}

    def register_platform_service(self, service_name: str, address: str):
        """Register a platform service (sessions, models, etc.)."""
        if service_name not in self._platform_services:
            self._platform_services[service_name] = []
        if address not in self._platform_services[service_name]:
            self._platform_services[service_name].append(address)
            print(f"Registered platform service '{service_name}' at {address}")

    def register_workflow(self, api_path: str, address: str):
        """Register a workflow service."""
        if api_path not in self._workflows:
            self._workflows[api_path] = []
        if address not in self._workflows[api_path]:
            self._workflows[api_path].append(address)
            print(f"Registered workflow '{api_path}' at {address}")

    def get_platform_service_address(self, service_name: str) -> str:
        """Get address for a platform service."""
        addresses = self._platform_services.get(service_name, [])
        if not addresses:
            raise ValueError(f"Platform service '{service_name}' not registered")
        return addresses[0]  # Simple: use first address

    def get_workflow_address(self, api_path: str) -> str:
        """Get address for a workflow based on API path."""
        addresses = self._workflows.get(api_path, [])
        if not addresses:
            raise ValueError(f"Workflow '{api_path}' not registered")
        return addresses[0]  # Simple: use first address
