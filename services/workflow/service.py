"""
Workflow Service — gRPC servicer (sync, follows the sessions/data pattern).

Thin translation layer: receives proto requests, delegates to
WorkflowRegistry / JobStore / a deployment dict, returns proto responses.

In commit 1 the deployment work is metadata-only — `DeployWorkflow` records
a `WorkflowDeployment` row but does not actually run any container. Commit 3
replaces the body of `DeployWorkflow` with real `docker run` + health-check
polling + a `RegisterRoute` call to the gateway.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 8.21: WorkflowService gRPC contract
  - Listings 8.22–8.23: registry + deployment + job message types
  - Listing 8.10: /jobs/{job_id} polling endpoint (the runtime server in
    commit 2 calls into these CreateJob/Update/Complete RPCs)
"""

import logging
import os
from typing import Dict, Optional

import grpc

from proto import workflow_pb2, workflow_pb2_grpc
from services.shared.servicer_base import BaseServicer
from services.workflow.deployer import (
    Deployer,
    DockerDeployer,
    HttpRoutePusher,
    RoutePusher,
)
from services.workflow.jobs_store import InMemoryJobStore, JobStore
from services.workflow.models import (
    ReliabilityConfig,
    ResourceConfig,
    ScalingConfig,
    WorkflowDeployment,
    WorkflowSpec,
)
from services.workflow.store import (
    InMemoryWorkflowRegistry,
    WorkflowRegistry,
    create_registry,
)

logger = logging.getLogger(__name__)


def _default_route_health_check(endpoint: str) -> bool:
    """Probe `http://<endpoint>/health/ready`. Returns True iff 200."""
    import httpx

    try:
        return httpx.get(f"http://{endpoint}/health/ready", timeout=0.5).status_code == 200
    except httpx.RequestError:
        return False


class WorkflowServiceImpl(workflow_pb2_grpc.WorkflowServiceServicer, BaseServicer):
    def __init__(
        self,
        registry: Optional[WorkflowRegistry] = None,
        jobs: Optional[JobStore] = None,
        deployer: Optional[Deployer] = None,
        route_pusher: Optional[RoutePusher] = None,
        route_health_check=None,
    ) -> None:
        self.registry = registry or InMemoryWorkflowRegistry()
        self.jobs = jobs or InMemoryJobStore()
        self.deployer = deployer or DockerDeployer()
        self.route_pusher = route_pusher or HttpRoutePusher(
            gateway_http_url=os.environ.get("GENAI_GATEWAY_HTTP_URL", "http://localhost:8080")
        )
        # Tests pass `lambda endpoint: True` to disable real network probes;
        # production keeps the default httpx-based check.
        self._route_health_check = route_health_check or _default_route_health_check
        self._deployments: Dict[str, WorkflowDeployment] = {}
        # workflow_id -> running container id (so RollbackWorkflow can stop it).
        self._containers: Dict[str, str] = {}
        # api_path -> endpoint. Workflow Service is the source of truth; the
        # gateway re-hydrates from `ListRoutes` on startup and gets push
        # updates via the gateway's `/__platform/register-route` endpoint.
        self._routes: Dict[str, str] = {}
        # name -> workflow_id. Cached so DeployWorkflow can resolve the spec
        # by id without scanning the registry.
        self._id_to_name: Dict[str, str] = {}

    def add_to_server(self, server: grpc.Server) -> None:
        workflow_pb2_grpc.add_WorkflowServiceServicer_to_server(self, server)

    # ---- proto <-> domain -------------------------------------------------

    def _proto_to_domain(self, proto: workflow_pb2.WorkflowSpec) -> WorkflowSpec:
        scaling = ScalingConfig()
        if proto.HasField("scaling"):
            scaling = ScalingConfig(
                min_replicas=proto.scaling.min_replicas or 1,
                max_replicas=proto.scaling.max_replicas or 10,
                target_cpu_percent=proto.scaling.target_cpu_percent or 70,
            )
        resources = ResourceConfig()
        if proto.HasField("resources"):
            resources = ResourceConfig(
                cpu=proto.resources.cpu or "500m",
                memory=proto.resources.memory or "512Mi",
                gpu_type=proto.resources.gpu_type,
                num_gpus=proto.resources.num_gpus,
            )
        reliability = ReliabilityConfig()
        if proto.HasField("reliability"):
            reliability = ReliabilityConfig(
                timeout_seconds=proto.reliability.timeout_seconds or 30,
                max_retries=proto.reliability.max_retries or 3,
            )
        return WorkflowSpec(
            name=proto.name,
            api_path=proto.api_path,
            container_image=proto.container_image,
            response_mode=proto.response_mode or "sync",
            scaling=scaling,
            resources=resources,
            reliability=reliability,
            version=proto.version or 1,
        )

    def _domain_to_proto(self, spec: WorkflowSpec) -> workflow_pb2.WorkflowSpec:
        return workflow_pb2.WorkflowSpec(
            name=spec.name,
            api_path=spec.api_path,
            container_image=spec.container_image,
            response_mode=spec.response_mode,
            scaling=workflow_pb2.ScalingConfig(
                min_replicas=spec.scaling.min_replicas,
                max_replicas=spec.scaling.max_replicas,
                target_cpu_percent=spec.scaling.target_cpu_percent,
            ),
            resources=workflow_pb2.ResourceConfig(
                cpu=spec.resources.cpu,
                memory=spec.resources.memory,
                gpu_type=spec.resources.gpu_type,
                num_gpus=spec.resources.num_gpus,
            ),
            reliability=workflow_pb2.ReliabilityConfig(
                timeout_seconds=spec.reliability.timeout_seconds,
                max_retries=spec.reliability.max_retries,
            ),
            version=spec.version,
        )

    def _deployment_to_proto(self, d: WorkflowDeployment) -> workflow_pb2.WorkflowDeployment:
        return workflow_pb2.WorkflowDeployment(
            workflow_id=d.workflow_id,
            deployment_id=d.deployment_id,
            version=d.version,
            status=d.status,
            current_replicas=d.current_replicas,
            desired_replicas=d.desired_replicas,
            healthy_endpoints=d.healthy_endpoints,
        )

    # ---- Registry RPCs ----------------------------------------------------

    def RegisterWorkflow(self, request, context):
        spec = self._proto_to_domain(request.spec)
        wf_id, version = self.registry.register(spec)
        self._id_to_name[wf_id] = spec.name
        return workflow_pb2.RegisterWorkflowResponse(workflow_id=wf_id, version=version)

    def GetWorkflow(self, request, context):
        spec = self.registry.get(request.name)
        if spec is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Workflow '{request.name}' not found")
            return workflow_pb2.GetWorkflowResponse()
        return workflow_pb2.GetWorkflowResponse(spec=self._domain_to_proto(spec))

    def ListWorkflows(self, request, context):
        return workflow_pb2.ListWorkflowsResponse(
            specs=[self._domain_to_proto(s) for s in self.registry.list()]
        )

    def UpdateWorkflow(self, request, context):
        try:
            new_version = self.registry.update(self._proto_to_domain(request.spec))
            return workflow_pb2.UpdateWorkflowResponse(version=new_version)
        except KeyError as e:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(e))
            return workflow_pb2.UpdateWorkflowResponse()

    def DeleteWorkflow(self, request, context):
        ok = self.registry.delete(request.name)
        return workflow_pb2.DeleteWorkflowResponse(success=ok)

    # ---- Deployment RPCs --------------------------------------------------

    def _resolve_workflow(self, workflow_id: str) -> Optional[WorkflowSpec]:
        name = self._id_to_name.get(workflow_id)
        if name is None:
            return None
        return self.registry.get(name)

    def DeployWorkflow(self, request, context):
        """Run the workflow's container, wait for it to be ready, register the route.

        Sequence (chapter 8 Deploy walkthrough, phases 5-6):
          1. Resolve the workflow spec by id.
          2. Stop any prior container running this workflow.
          3. Run the new image via the injected ``Deployer`` (Docker by default).
          4. Poll ``/health/ready`` until 200 or timeout.
          5. Update the local deployment + route maps.
          6. Push the (api_path, endpoint) pair to the gateway so external
             HTTP requests start landing on the new container.
        """
        spec = self._resolve_workflow(request.workflow_id)
        if spec is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Workflow id '{request.workflow_id}' not found")
            return workflow_pb2.DeployWorkflowResponse()

        version = request.version or spec.version

        previous_container = self._containers.get(request.workflow_id)
        if previous_container:
            try:
                self.deployer.stop(previous_container)
            except Exception:  # noqa: BLE001 — best-effort cleanup
                logger.exception("failed to stop previous container %s", previous_container)

        try:
            result = self.deployer.deploy(
                image=spec.container_image or f"genai-workflow/{spec.name}:latest",
                name=spec.name,
                gateway_url=os.environ.get(
                    "GENAI_CONTAINER_GATEWAY_URL", "host.docker.internal:50051"
                ),
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("docker deploy for %s failed", spec.name)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"deploy failed: {e}")
            return workflow_pb2.DeployWorkflowResponse()

        ready = self.deployer.wait_until_ready(host_port=result.host_port, timeout_seconds=30.0)
        endpoint = f"localhost:{result.host_port}"

        if not ready:
            # Surface the container's stderr/stdout so the operator can see
            # why /health/ready never came up (most often: import error in
            # the workflow source file). The CLI prints the error details.
            logs = self.deployer.container_logs(result.container_id)
            if logs:
                logger.error(
                    "workflow %s failed health check; container logs:\n%s",
                    spec.name,
                    logs,
                )
            context.set_details(
                f"Workflow '{spec.name}' container failed /health/ready.\n"
                f"--- last container log lines ---\n{logs or '(no logs available)'}"
            )

        deployment = WorkflowDeployment(
            workflow_id=request.workflow_id,
            deployment_id=f"{spec.name}-v{version}",
            version=version,
            status="healthy" if ready else "failed",
            current_replicas=1 if ready else 0,
            desired_replicas=spec.scaling.min_replicas,
            healthy_endpoints=[endpoint] if ready else [],
        )
        self._deployments[request.workflow_id] = deployment
        self._containers[request.workflow_id] = result.container_id

        if ready:
            self._routes[spec.api_path] = endpoint
            self.route_pusher.push(spec.api_path, endpoint)

        return workflow_pb2.DeployWorkflowResponse(deployment=self._deployment_to_proto(deployment))

    def GetDeploymentStatus(self, request, context):
        d = self._deployments.get(request.workflow_id)
        if d is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"No deployment for workflow id '{request.workflow_id}'")
            return workflow_pb2.GetDeploymentStatusResponse()
        return workflow_pb2.GetDeploymentStatusResponse(deployment=self._deployment_to_proto(d))

    def RollbackWorkflow(self, request, context):
        d = self._deployments.get(request.workflow_id)
        if d is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"No deployment for workflow id '{request.workflow_id}'")
            return workflow_pb2.RollbackWorkflowResponse()
        d.version = request.target_version or max(1, d.version - 1)
        d.status = "deploying"
        return workflow_pb2.RollbackWorkflowResponse(deployment=self._deployment_to_proto(d))

    # ---- Routing RPC ------------------------------------------------------

    def RegisterRoute(self, request, context):
        """Workflow Service records the route locally; commit 3 also pushes it
        to the gateway. The gateway re-hydrates via `ListRoutes` on startup.
        """
        self._routes[request.api_path] = request.endpoint
        return workflow_pb2.RegisterRouteResponse(success=True)

    def ListRoutes(self, request, context):
        """Source of truth for the gateway's routing-table re-hydration.

        Probes each registered endpoint's `/health/ready` first and drops
        routes whose container has gone away (e.g., docker daemon
        restarted, container OOM'd, manual `docker rm -f`). Without this,
        a dead container's stale route survives forever in both
        `_routes` and the gateway's local cache.
        """
        self._prune_dead_routes()
        return workflow_pb2.ListRoutesResponse(
            routes=[workflow_pb2.Route(api_path=p, endpoint=e) for p, e in self._routes.items()]
        )

    def _prune_dead_routes(self) -> None:
        dead: list[str] = []
        for path, endpoint in list(self._routes.items()):
            if not self._route_health_check(endpoint):
                dead.append(path)
        for path in dead:
            logger.info("pruning dead route %s → %s", path, self._routes[path])
            del self._routes[path]

    def routes(self) -> Dict[str, str]:
        """Test/inspection accessor for the registered routes."""
        return dict(self._routes)

    # ---- Job RPCs ---------------------------------------------------------

    def CreateJob(self, request, context):
        jid = self.jobs.create_job(
            workflow_id=request.workflow_id,
            input_json=request.input_json,
            assigned_endpoint=request.assigned_endpoint,
        )
        return workflow_pb2.CreateJobResponse(job_id=jid)

    def GetJobStatus(self, request, context):
        job = self.jobs.get_job(request.job_id)
        if job is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Job '{request.job_id}' not found")
            return workflow_pb2.GetJobStatusResponse()
        return workflow_pb2.GetJobStatusResponse(
            job=workflow_pb2.WorkflowJob(
                job_id=job.job_id,
                workflow_id=job.workflow_id,
                status=job.status,
                progress_message=job.progress_message,
                input_json=job.input_json,
                result_json=job.result_json,
                error=job.error,
                checkpoint_json=job.checkpoint_json,
                assigned_endpoint=job.assigned_endpoint,
                created_at=job.created_at,
                updated_at=job.updated_at,
            )
        )

    def UpdateJobProgress(self, request, context):
        ok = self.jobs.update_progress(request.job_id, request.progress_message)
        if not ok:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Job '{request.job_id}' not found")
        return workflow_pb2.UpdateJobProgressResponse(success=ok)

    def SaveJobCheckpoint(self, request, context):
        ok = self.jobs.save_checkpoint(request.job_id, request.checkpoint_json)
        if not ok:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Job '{request.job_id}' not found")
        return workflow_pb2.SaveJobCheckpointResponse(success=ok)

    def CompleteJob(self, request, context):
        ok = self.jobs.complete(request.job_id, request.result_json)
        if not ok:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Job '{request.job_id}' not found")
        return workflow_pb2.CompleteJobResponse(success=ok)

    def FailJob(self, request, context):
        ok = self.jobs.fail(request.job_id, request.error)
        if not ok:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Job '{request.job_id}' not found")
        return workflow_pb2.FailJobResponse(success=ok)

    def CancelJob(self, request, context):
        ok = self.jobs.cancel(request.job_id)
        if not ok:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Job '{request.job_id}' not found")
        return workflow_pb2.CancelJobResponse(success=ok)


# Re-export the env-driven factory for parity with sessions/store.create_storage.
__all__ = ["WorkflowServiceImpl", "create_registry"]
