"""Tests for async tool execution (Listing 6.16)."""

import asyncio
import json

import httpx

from proto import tools_pb2
from services.tools.service import ToolServiceImpl


class FakeContext:
    def __init__(self):
        self.code = None
        self.details_str = None

    def set_code(self, code):
        self.code = code

    def set_details(self, details):
        self.details_str = details


def _mock_client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def _register_tool(svc, *, name, endpoint, timeout_seconds=0):
    tool = tools_pb2.ToolDefinition(name=name, description=name, endpoint=endpoint)
    if timeout_seconds:
        tool.execution_limits.CopyFrom(tools_pb2.ExecutionLimits(timeout_seconds=timeout_seconds))
    await svc.RegisterTool(tools_pb2.RegisterToolRequest(tool=tool), FakeContext())


async def _wait_until_done(svc, task_id, attempts=50):
    for _ in range(attempts):
        resp = await svc.GetTask(tools_pb2.GetTaskRequest(task_id=task_id), FakeContext())
        if resp.status not in ("pending", "running"):
            return resp
        await asyncio.sleep(0.01)
    return resp


class TestExecuteToolAsync:
    async def test_execute_async_returns_task_id_and_pending_status(self):
        def handler(request):
            return httpx.Response(200, json={"ok": True})

        svc = ToolServiceImpl(http_client=_mock_client(handler))
        await _register_tool(svc, name="t.async", endpoint="https://api.example.com/x")

        resp = await svc.ExecuteToolAsync(
            tools_pb2.ExecuteToolRequest(tool_name="t.async", arguments_json="{}"),
            FakeContext(),
        )

        assert resp.task_id
        assert resp.status in ("pending", "running", "succeeded")

    async def test_get_task_returns_completed_status_and_result(self):
        def handler(request):
            return httpx.Response(200, json={"appointment_id": "apt-1"})

        svc = ToolServiceImpl(http_client=_mock_client(handler))
        await _register_tool(svc, name="t.async", endpoint="https://api.example.com/x")

        start = await svc.ExecuteToolAsync(
            tools_pb2.ExecuteToolRequest(tool_name="t.async", arguments_json="{}"),
            FakeContext(),
        )
        final = await _wait_until_done(svc, start.task_id)

        assert final.status == "succeeded"
        assert json.loads(final.result_json) == {"appointment_id": "apt-1"}
        assert final.error == ""

    async def test_failed_task_captures_error(self):
        def handler(request):
            return httpx.Response(500, json={"err": "boom"})

        svc = ToolServiceImpl(http_client=_mock_client(handler))
        await _register_tool(svc, name="t.async", endpoint="https://api.example.com/x")

        start = await svc.ExecuteToolAsync(
            tools_pb2.ExecuteToolRequest(tool_name="t.async", arguments_json="{}"),
            FakeContext(),
        )
        final = await _wait_until_done(svc, start.task_id)

        assert final.status == "failed"
        assert "500" in final.error

    async def test_timed_out_task_marked_timed_out(self):
        def handler(request):
            raise httpx.ReadTimeout("simulated", request=request)

        svc = ToolServiceImpl(http_client=_mock_client(handler))
        await _register_tool(
            svc, name="t.async", endpoint="https://api.example.com/x", timeout_seconds=1
        )

        start = await svc.ExecuteToolAsync(
            tools_pb2.ExecuteToolRequest(tool_name="t.async", arguments_json="{}"),
            FakeContext(),
        )
        final = await _wait_until_done(svc, start.task_id)

        assert final.status == "timed_out"
        assert "timeout" in final.error.lower()


class TestGetTask:
    async def test_unknown_task_id_returns_not_found(self):
        svc = ToolServiceImpl()
        ctx = FakeContext()
        resp = await svc.GetTask(tools_pb2.GetTaskRequest(task_id="does-not-exist"), ctx)
        import grpc as _grpc

        assert ctx.code == _grpc.StatusCode.NOT_FOUND
        assert resp.status == ""


class TestConcurrentAsyncTasks:
    async def test_many_concurrent_tasks_complete_independently(self):
        # Each request records its own body; handler returns that body echoed.
        def handler(request):
            args = json.loads(request.content.decode() or "{}")
            return httpx.Response(200, json={"echo": args.get("i", -1)})

        svc = ToolServiceImpl(http_client=_mock_client(handler))
        await _register_tool(svc, name="t.async", endpoint="https://api.example.com/x")

        starts = []
        for i in range(5):
            r = await svc.ExecuteToolAsync(
                tools_pb2.ExecuteToolRequest(
                    tool_name="t.async", arguments_json=json.dumps({"i": i})
                ),
                FakeContext(),
            )
            starts.append((i, r.task_id))

        results = []
        for i, tid in starts:
            final = await _wait_until_done(svc, tid)
            assert final.status == "succeeded"
            results.append((i, json.loads(final.result_json)["echo"]))

        assert sorted(results) == [(i, i) for i in range(5)]
