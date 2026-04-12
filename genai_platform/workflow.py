"""
Workflow decorator for defining deployable AI workflows.

The @workflow decorator captures deployment configuration and marks
functions as workflows that can be deployed as independent services.
"""

from functools import wraps
from typing import Any, Callable, Dict, Optional


def workflow(
    name: str,
    api_path: str,
    response_mode: str = "sync",
    autoscaling_config: Optional[Dict[str, Any]] = None,
    deployment_config: Optional[Dict[str, Any]] = None,
) -> Callable:
    """
    Decorator for defining AI workflows.

    Workflows are functions that get deployed as independent containerized
    services. The decorator captures deployment configuration that the
    deployment tool uses to package and register workflows.

    Args:
        name: Unique workflow identifier
        api_path: API endpoint path where this workflow is exposed
        response_mode: Response pattern - "sync", "async", or "stream"
        autoscaling_config: Optional dict with autoscaling settings:
            - min_replicas: Minimum number of container replicas (default: 1)
            - max_replicas: Maximum number of container replicas (default: 10)
            - target_ongoing_requests: Target number of ongoing requests per replica
                                      before scaling up (default: 10)
            - target_cpu_percent: Optional CPU threshold (0-100) for scaling.
                                 If not set, uses request-based scaling only.
        deployment_config: Optional dict with advanced deployment settings.
            This dict is flexible and accepts any keys - the deployment tool
            will pass through all configuration. Common options include:
            - memory_limit: Memory limit per container (e.g., "512Mi", "2Gi")
            - cpu_limit: CPU limit per container (e.g., "500m", "2")
            - gpu_count: Number of GPUs per container (e.g., 1, 2)
            - gpu_type: GPU type/model (e.g., "nvidia-t4", "nvidia-a100")
            - request_timeout: Request timeout in seconds
            - execution_timeout: Maximum execution time in seconds
            - env_vars: Dict of environment variables
            - secrets: List of secret names to mount
            - health_check_path: Path for health checks (default: "/health")
            - health_check_interval: Health check interval in seconds
            - max_retries: Maximum retry attempts for failed requests
            - labels: Dict of labels for observability/tagging
            Any additional keys will be passed through to the deployment system.

    Example:
        # Simple workflow with defaults
        @workflow(
            name="patient_intake_assistant",
            api_path="/patient-assistant"
        )
        def handle_patient_question(question, patient_id):
            platform = GenAIPlatform()
            # ... workflow logic ...
            return {"response": "..."}

        # Workflow with custom autoscaling
        @workflow(
            name="high_traffic_service",
            api_path="/api",
            autoscaling_config={
                "min_replicas": 2,
                "max_replicas": 20,
                "target_ongoing_requests": 5  # Scale up when >5 requests per replica
            }
        )
        def handle_request(data):
            # ... handle request ...
            pass

        # GPU-enabled workflow for model inference
        @workflow(
            name="model_inference",
            api_path="/infer",
            deployment_config={
                "gpu_count": 1,
                "gpu_type": "nvidia-t4",
                "memory_limit": "8Gi",
                "cpu_limit": "4"
            }
        )
        def run_inference(input_data):
            # ... GPU-accelerated inference ...
            pass

    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # The wrapper just calls the original function
            # The decorator doesn't change behavior, just adds metadata
            return func(*args, **kwargs)

        # Build deployment metadata
        metadata = {"name": name, "api_path": api_path, "response_mode": response_mode}

        # Set up autoscaling config - merge user config with defaults
        default_autoscaling = {"min_replicas": 1, "max_replicas": 10, "target_ongoing_requests": 10}
        if autoscaling_config:
            # Merge user config over defaults
            metadata["autoscaling"] = {**default_autoscaling, **autoscaling_config}
        else:
            metadata["autoscaling"] = default_autoscaling

        # Add deployment config if provided
        if deployment_config:
            metadata["deployment_config"] = deployment_config

        # Attach deployment metadata to the function
        wrapper._workflow_metadata = metadata

        return wrapper

    return decorator
