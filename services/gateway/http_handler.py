"""
HTTP Handler - Routes external HTTP requests to workflow containers.

Handles POST requests from external clients and forwards them to
the appropriate workflow containers based on API path.
"""

import json
from http.server import BaseHTTPRequestHandler

from services.gateway.registry import ServiceRegistry


class WorkflowHTTPHandler(BaseHTTPRequestHandler):
    """HTTP handler that routes requests to workflow containers."""

    def __init__(self, registry: ServiceRegistry, *args, **kwargs):
        self.registry = registry
        super().__init__(*args, **kwargs)

    def do_POST(self):
        """Handle POST requests to workflow endpoints."""
        try:
            # Extract API path from request
            api_path = self.path

            # Get workflow address from registry
            workflow_addr = self.registry.get_workflow_address(api_path)

            # Forward request to workflow container
            # For now, this is a placeholder - in production, we'd use
            # an HTTP client library to forward the request
            # TODO: Implement actual HTTP forwarding to workflow containers

            # Placeholder response
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            response = {
                "message": f"Request routed to workflow at {workflow_addr}",
                "api_path": api_path,
                "note": "HTTP forwarding to workflows not yet implemented",
            }
            self.wfile.write(json.dumps(response).encode())

        except ValueError as e:
            # Workflow not found
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            error = {"error": str(e)}
            self.wfile.write(json.dumps(error).encode())
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            error = {"error": str(e)}
            self.wfile.write(json.dumps(error).encode())

    def log_message(self, format, *args):
        """Override to reduce log noise."""
        pass
