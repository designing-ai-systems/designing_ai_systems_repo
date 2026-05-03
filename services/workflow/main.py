"""
Workflow Service - Main entry point.

Manages workflow registration, deployment, and async job state for the
platform. The companion data plane (the SDK runtime server in
genai_platform/runtime/server.py, added in commit 2 of the chapter-8 plan)
calls into this service over gRPC for job lifecycle operations.
"""

from services.shared.server import create_grpc_server, get_service_port, run_service
from services.workflow.service import WorkflowServiceImpl


def main():
    """Run the Workflow Service server."""
    service_name = "workflow"
    port = get_service_port(service_name)

    servicer = WorkflowServiceImpl()
    server = create_grpc_server(servicer=servicer, port=port, service_name=service_name)

    run_service(server, service_name, port)


if __name__ == "__main__":
    main()
