"""Tests for the Workflow Service's route table — RegisterRoute + ListRoutes.

These power gateway re-hydration on startup (the gateway calls ListRoutes
and re-registers every workflow's path → endpoint into its local cache).
"""

from proto import workflow_pb2
from services.workflow.service import WorkflowServiceImpl


def _svc(*, all_alive: bool = True):
    """Helper: WorkflowServiceImpl with an injectable health check for routes."""
    return WorkflowServiceImpl(route_health_check=lambda _endpoint: all_alive)


class _FakeContext:
    def set_code(self, _):
        pass

    def set_details(self, _):
        pass


def test_list_routes_empty():
    svc = _svc()
    resp = svc.ListRoutes(workflow_pb2.ListRoutesRequest(), _FakeContext())
    assert list(resp.routes) == []


def test_list_routes_returns_registered_pairs():
    svc = _svc()
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
    svc = _svc()
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


def test_list_routes_prunes_dead_endpoints():
    """A registered endpoint whose /health/ready does not respond is dropped.

    Without this, a stale route survives in `_routes` after the container
    dies, and the gateway's cache (re-hydrated from this RPC) keeps
    pointing at the dead address.
    """
    svc = _svc(all_alive=False)
    svc.RegisterRoute(
        workflow_pb2.RegisterRouteRequest(api_path="/dead", endpoint="127.0.0.1:1"),
        _FakeContext(),
    )
    resp = svc.ListRoutes(workflow_pb2.ListRoutesRequest(), _FakeContext())
    assert list(resp.routes) == []
    # Local cache also dropped.
    assert "/dead" not in svc.routes()


def test_list_routes_keeps_live_endpoints():
    svc = _svc(all_alive=True)
    svc.RegisterRoute(
        workflow_pb2.RegisterRouteRequest(api_path="/alive", endpoint="127.0.0.1:8001"),
        _FakeContext(),
    )
    resp = svc.ListRoutes(workflow_pb2.ListRoutesRequest(), _FakeContext())
    assert {(r.api_path, r.endpoint) for r in resp.routes} == {("/alive", "127.0.0.1:8001")}


def test_prune_only_drops_failing_endpoints():
    """Health checker that returns True for some endpoints, False for others."""
    failing = {"127.0.0.1:1"}
    svc = WorkflowServiceImpl(route_health_check=lambda endpoint: endpoint not in failing)
    svc.RegisterRoute(
        workflow_pb2.RegisterRouteRequest(api_path="/dead", endpoint="127.0.0.1:1"),
        _FakeContext(),
    )
    svc.RegisterRoute(
        workflow_pb2.RegisterRouteRequest(api_path="/alive", endpoint="127.0.0.1:8001"),
        _FakeContext(),
    )
    resp = svc.ListRoutes(workflow_pb2.ListRoutesRequest(), _FakeContext())
    assert {(r.api_path, r.endpoint) for r in resp.routes} == {("/alive", "127.0.0.1:8001")}
