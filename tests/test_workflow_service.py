"""Tests for WorkflowServiceImpl gRPC servicer (Listings 8.21–8.23)."""

import json

import grpc
import pytest

from proto import workflow_pb2
from services.workflow.deployer import FakeDeployer, FakeRoutePusher
from services.workflow.service import WorkflowServiceImpl


def _svc(deployer=None, route_pusher=None):
    """Construct a WorkflowServiceImpl with fake deploy infrastructure."""
    return WorkflowServiceImpl(
        deployer=deployer or FakeDeployer(),
        route_pusher=route_pusher or FakeRoutePusher(),
    )


class FakeContext:
    """Minimal sync gRPC context for unit tests."""

    def __init__(self):
        self.code = None
        self.details_str = None

    def set_code(self, code):
        self.code = code

    def set_details(self, details):
        self.details_str = details


def _make_spec(name="patient_intake", api_path="/p", response_mode="sync"):
    return workflow_pb2.WorkflowSpec(
        name=name,
        api_path=api_path,
        response_mode=response_mode,
        scaling=workflow_pb2.ScalingConfig(min_replicas=1, max_replicas=5, target_cpu_percent=70),
        resources=workflow_pb2.ResourceConfig(cpu="500m", memory="512Mi"),
        reliability=workflow_pb2.ReliabilityConfig(timeout_seconds=30, max_retries=3),
    )


# ---- Registry RPCs ----------------------------------------------------------


class TestRegisterWorkflow:
    def test_returns_workflow_id_and_initial_version(self):
        svc = _svc()
        resp = svc.RegisterWorkflow(
            workflow_pb2.RegisterWorkflowRequest(spec=_make_spec()), FakeContext()
        )
        assert resp.workflow_id
        assert resp.version == 1

    def test_re_registering_same_name_bumps_version(self):
        svc = _svc()
        svc.RegisterWorkflow(
            workflow_pb2.RegisterWorkflowRequest(spec=_make_spec(name="w")), FakeContext()
        )
        resp2 = svc.RegisterWorkflow(
            workflow_pb2.RegisterWorkflowRequest(spec=_make_spec(name="w")), FakeContext()
        )
        assert resp2.version == 2


class TestGetWorkflow:
    def test_returns_spec_for_known(self):
        svc = _svc()
        svc.RegisterWorkflow(
            workflow_pb2.RegisterWorkflowRequest(spec=_make_spec(name="w")), FakeContext()
        )
        resp = svc.GetWorkflow(workflow_pb2.GetWorkflowRequest(name="w"), FakeContext())
        assert resp.spec.name == "w"

    def test_unknown_returns_not_found(self):
        svc = _svc()
        ctx = FakeContext()
        svc.GetWorkflow(workflow_pb2.GetWorkflowRequest(name="missing"), ctx)
        assert ctx.code == grpc.StatusCode.NOT_FOUND


class TestListWorkflows:
    def test_returns_all_specs(self):
        svc = _svc()
        svc.RegisterWorkflow(
            workflow_pb2.RegisterWorkflowRequest(spec=_make_spec(name="a")), FakeContext()
        )
        svc.RegisterWorkflow(
            workflow_pb2.RegisterWorkflowRequest(spec=_make_spec(name="b")), FakeContext()
        )
        resp = svc.ListWorkflows(workflow_pb2.ListWorkflowsRequest(), FakeContext())
        assert sorted(s.name for s in resp.specs) == ["a", "b"]


class TestUpdateAndDelete:
    def test_update_increments_version(self):
        svc = _svc()
        svc.RegisterWorkflow(
            workflow_pb2.RegisterWorkflowRequest(spec=_make_spec(name="w")), FakeContext()
        )
        resp = svc.UpdateWorkflow(
            workflow_pb2.UpdateWorkflowRequest(spec=_make_spec(name="w", api_path="/v2")),
            FakeContext(),
        )
        assert resp.version == 2

    def test_update_unknown_returns_not_found(self):
        svc = _svc()
        ctx = FakeContext()
        svc.UpdateWorkflow(workflow_pb2.UpdateWorkflowRequest(spec=_make_spec(name="missing")), ctx)
        assert ctx.code == grpc.StatusCode.NOT_FOUND

    def test_delete_existing(self):
        svc = _svc()
        svc.RegisterWorkflow(
            workflow_pb2.RegisterWorkflowRequest(spec=_make_spec(name="w")), FakeContext()
        )
        resp = svc.DeleteWorkflow(workflow_pb2.DeleteWorkflowRequest(name="w"), FakeContext())
        assert resp.success is True

    def test_delete_missing(self):
        svc = _svc()
        resp = svc.DeleteWorkflow(workflow_pb2.DeleteWorkflowRequest(name="x"), FakeContext())
        assert resp.success is False


# ---- Deployment RPCs --------------------------------------------------------


class TestDeployWorkflow:
    def test_healthy_deploy_records_endpoint_and_pushes_route(self):
        deployer = FakeDeployer(host_port=18001)
        pusher = FakeRoutePusher()
        svc = _svc(deployer=deployer, route_pusher=pusher)
        reg = svc.RegisterWorkflow(
            workflow_pb2.RegisterWorkflowRequest(spec=_make_spec(name="w", api_path="/p")),
            FakeContext(),
        )
        resp = svc.DeployWorkflow(
            workflow_pb2.DeployWorkflowRequest(workflow_id=reg.workflow_id, version=1),
            FakeContext(),
        )
        d = resp.deployment
        assert d.status == "healthy"
        assert list(d.healthy_endpoints) == ["localhost:18001"]
        # Deployer was called with the expected name + image.
        assert len(deployer.deployed) == 1
        assert deployer.deployed[0]["name"] == "w"
        # Route was both stored locally and pushed to the gateway.
        assert svc.routes()["/p"] == "localhost:18001"
        assert pusher.pushed == [("/p", "localhost:18001")]

    def test_unhealthy_deploy_does_not_register_route(self):
        deployer = FakeDeployer()
        deployer.ready_returns = False
        pusher = FakeRoutePusher()
        svc = _svc(deployer=deployer, route_pusher=pusher)
        reg = svc.RegisterWorkflow(
            workflow_pb2.RegisterWorkflowRequest(spec=_make_spec(name="w", api_path="/p")),
            FakeContext(),
        )
        resp = svc.DeployWorkflow(
            workflow_pb2.DeployWorkflowRequest(workflow_id=reg.workflow_id, version=1),
            FakeContext(),
        )
        assert resp.deployment.status == "failed"
        assert svc.routes() == {}
        assert pusher.pushed == []

    def test_redeploy_stops_previous_container(self):
        deployer = FakeDeployer()
        svc = _svc(deployer=deployer)
        reg = svc.RegisterWorkflow(
            workflow_pb2.RegisterWorkflowRequest(spec=_make_spec(name="w")),
            FakeContext(),
        )
        svc.DeployWorkflow(
            workflow_pb2.DeployWorkflowRequest(workflow_id=reg.workflow_id, version=1),
            FakeContext(),
        )
        svc.DeployWorkflow(
            workflow_pb2.DeployWorkflowRequest(workflow_id=reg.workflow_id, version=2),
            FakeContext(),
        )
        # Second deploy should have stopped the first container.
        assert deployer.stopped == ["fake-w"]

    def test_unknown_workflow_returns_not_found(self):
        svc = _svc()
        ctx = FakeContext()
        svc.DeployWorkflow(
            workflow_pb2.DeployWorkflowRequest(workflow_id="does-not-exist", version=1), ctx
        )
        assert ctx.code == grpc.StatusCode.NOT_FOUND


class TestGetDeploymentStatus:
    def test_returns_record_after_deploy(self):
        svc = _svc()
        reg = svc.RegisterWorkflow(
            workflow_pb2.RegisterWorkflowRequest(spec=_make_spec(name="w")), FakeContext()
        )
        svc.DeployWorkflow(
            workflow_pb2.DeployWorkflowRequest(workflow_id=reg.workflow_id, version=1),
            FakeContext(),
        )
        resp = svc.GetDeploymentStatus(
            workflow_pb2.GetDeploymentStatusRequest(workflow_id=reg.workflow_id), FakeContext()
        )
        assert resp.deployment.workflow_id == reg.workflow_id

    def test_unknown_returns_not_found(self):
        svc = _svc()
        ctx = FakeContext()
        svc.GetDeploymentStatus(workflow_pb2.GetDeploymentStatusRequest(workflow_id="missing"), ctx)
        assert ctx.code == grpc.StatusCode.NOT_FOUND


# ---- Routing RPC ------------------------------------------------------------


class TestRegisterRoute:
    def test_records_route_in_local_table(self):
        svc = _svc()
        resp = svc.RegisterRoute(
            workflow_pb2.RegisterRouteRequest(
                api_path="/patient-assistant", endpoint="localhost:8001"
            ),
            FakeContext(),
        )
        assert resp.success is True
        # Service exposes the routes for tests / re-hydration
        assert svc.routes()["/patient-assistant"] == "localhost:8001"


# ---- Job RPCs ---------------------------------------------------------------


class TestJobLifecycle:
    def test_create_then_get(self):
        svc = _svc()
        # Register a workflow first so there's something to attach the job to.
        reg = svc.RegisterWorkflow(
            workflow_pb2.RegisterWorkflowRequest(spec=_make_spec(name="researcher")),
            FakeContext(),
        )
        c = svc.CreateJob(
            workflow_pb2.CreateJobRequest(
                workflow_id=reg.workflow_id,
                input_json='{"topic": "x"}',
                assigned_endpoint="localhost:8001",
            ),
            FakeContext(),
        )
        assert c.job_id

        g = svc.GetJobStatus(workflow_pb2.GetJobStatusRequest(job_id=c.job_id), FakeContext())
        assert g.job.job_id == c.job_id
        assert g.job.status == "pending"
        assert g.job.input_json == '{"topic": "x"}'

    def test_progress_then_complete(self):
        svc = _svc()
        c = svc.CreateJob(
            workflow_pb2.CreateJobRequest(workflow_id="w", input_json="{}"),
            FakeContext(),
        )
        svc.UpdateJobProgress(
            workflow_pb2.UpdateJobProgressRequest(
                job_id=c.job_id, progress_message="running step 1"
            ),
            FakeContext(),
        )
        svc.SaveJobCheckpoint(
            workflow_pb2.SaveJobCheckpointRequest(
                job_id=c.job_id, checkpoint_json='{"phase": "retrieved"}'
            ),
            FakeContext(),
        )
        svc.CompleteJob(
            workflow_pb2.CompleteJobRequest(job_id=c.job_id, result_json='{"answer": 42}'),
            FakeContext(),
        )

        g = svc.GetJobStatus(workflow_pb2.GetJobStatusRequest(job_id=c.job_id), FakeContext())
        assert g.job.status == "succeeded"
        assert json.loads(g.job.result_json) == {"answer": 42}
        assert g.job.progress_message == "running step 1"
        assert g.job.checkpoint_json == '{"phase": "retrieved"}'

    def test_fail_records_error(self):
        svc = _svc()
        c = svc.CreateJob(
            workflow_pb2.CreateJobRequest(workflow_id="w", input_json="{}"), FakeContext()
        )
        svc.FailJob(workflow_pb2.FailJobRequest(job_id=c.job_id, error="boom"), FakeContext())
        g = svc.GetJobStatus(workflow_pb2.GetJobStatusRequest(job_id=c.job_id), FakeContext())
        assert g.job.status == "failed"
        assert g.job.error == "boom"

    def test_cancel(self):
        svc = _svc()
        c = svc.CreateJob(
            workflow_pb2.CreateJobRequest(workflow_id="w", input_json="{}"), FakeContext()
        )
        svc.CancelJob(workflow_pb2.CancelJobRequest(job_id=c.job_id), FakeContext())
        g = svc.GetJobStatus(workflow_pb2.GetJobStatusRequest(job_id=c.job_id), FakeContext())
        assert g.job.status == "cancelled"

    def test_get_unknown_job_returns_not_found(self):
        svc = _svc()
        ctx = FakeContext()
        svc.GetJobStatus(workflow_pb2.GetJobStatusRequest(job_id="missing"), ctx)
        assert ctx.code == grpc.StatusCode.NOT_FOUND


@pytest.mark.parametrize(
    "op",
    [
        ("UpdateJobProgress", workflow_pb2.UpdateJobProgressRequest, {"progress_message": "x"}),
        ("SaveJobCheckpoint", workflow_pb2.SaveJobCheckpointRequest, {"checkpoint_json": "{}"}),
        ("CompleteJob", workflow_pb2.CompleteJobRequest, {"result_json": "{}"}),
        ("FailJob", workflow_pb2.FailJobRequest, {"error": "x"}),
        ("CancelJob", workflow_pb2.CancelJobRequest, {}),
    ],
)
class TestJobOpsOnUnknownJob:
    def test_returns_not_found(self, op):
        svc = _svc()
        method_name, request_cls, kwargs = op
        ctx = FakeContext()
        getattr(svc, method_name)(request_cls(job_id="missing", **kwargs), ctx)
        assert ctx.code == grpc.StatusCode.NOT_FOUND
