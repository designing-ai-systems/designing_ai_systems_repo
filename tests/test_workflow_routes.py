"""Tests for the Workflow Service's route table — RegisterRoute + ListRoutes.

These power gateway re-hydration on startup (the gateway calls ListRoutes
and re-registers every workflow's path → endpoint into its local cache).
"""

from proto import workflow_pb2
from services.workflow.service import WorkflowServiceImpl


class _FakeContext:
    def set_code(self, _):
        pass

    def set_details(self, _):
        pass


def test_list_routes_empty():
    svc = WorkflowServiceImpl()
    resp = svc.ListRoutes(workflow_pb2.ListRoutesRequest(), _FakeContext())
    assert list(resp.routes) == []


def test_list_routes_returns_registered_pairs():
    svc = WorkflowServiceImpl()
    svc.RegisterRoute(
        workflow_pb2.RegisterRouteRequest(api_path="/patient-assistant", endpoint="localhost:8001"),
        _FakeContext(),
    )
    svc.RegisterRoute(
        workflow_pb2.RegisterRouteRequest(api_path="/research", endpoint="localhost:8002"),
        _FakeContext(),
    )
    resp = svc.ListRoutes(workflow_pb2.ListRoutesRequest(), _FakeContext())
    routes = sorted((r.api_path, r.endpoint) for r in resp.routes)
    assert routes == [
        ("/patient-assistant", "localhost:8001"),
        ("/research", "localhost:8002"),
    ]


def test_re_registering_overwrites():
    svc = WorkflowServiceImpl()
    svc.RegisterRoute(
        workflow_pb2.RegisterRouteRequest(api_path="/p", endpoint="addr:1"),
        _FakeContext(),
    )
    svc.RegisterRoute(
        workflow_pb2.RegisterRouteRequest(api_path="/p", endpoint="addr:2"),
        _FakeContext(),
    )
    resp = svc.ListRoutes(workflow_pb2.ListRoutesRequest(), _FakeContext())
    assert {(r.api_path, r.endpoint) for r in resp.routes} == {("/p", "addr:2")}
