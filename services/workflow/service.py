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

from typing import Dict, Optional

import grpc

from proto import workflow_pb2, workflow_pb2_grpc
from services.shared.servicer_base import BaseServicer
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


class WorkflowServiceImpl(workflow_pb2_grpc.WorkflowServiceServicer, BaseServicer):
    def __init__(
        self,
        registry: Optional[WorkflowRegistry] = None,
        jobs: Optional[JobStore] = None,
    ) -> None:
        self.registry = registry or InMemoryWorkflowRegistry()
        self.jobs = jobs or InMemoryJobStore()
        # Workflow_id -> latest deployment record. Commit 3 layers real
        # subprocess management on top of this dict.
        self._deployments: Dict[str, WorkflowDeployment] = {}
        # api_path -> endpoint. Workflow Service tracks routes locally so
        # the gateway can re-hydrate via ListRoutes (or, for now, by reading
        # this map after restarts during demos).
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
        """Commit-1 implementation: records deployment metadata only.

        Commit 3 replaces the body with real `docker run` + health polling
        + `RegisterRoute` call to the gateway.
        """
        spec = self._resolve_workflow(request.workflow_id)
        if spec is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Workflow id '{request.workflow_id}' not found")
            return workflow_pb2.DeployWorkflowResponse()

        deployment = WorkflowDeployment(
            workflow_id=request.workflow_id,
            deployment_id=f"{spec.name}-v{request.version or spec.version}",
            version=request.version or spec.version,
            status="deploying",
            current_replicas=0,
            desired_replicas=spec.scaling.min_replicas,
            healthy_endpoints=[],
        )
        self._deployments[request.workflow_id] = deployment
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
        to the gateway. The gateway re-hydrates from `routes()` on startup
        (Routing model and storage section of the chapter-8 plan).
        """
        self._routes[request.api_path] = request.endpoint
        return workflow_pb2.RegisterRouteResponse(success=True)

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
