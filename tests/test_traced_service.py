"""Unit tests for TracedService (Listing 7.6)."""

import pytest

from services.shared.observability_client import ObservabilityClient
from services.shared.traced_service import StatusCode, TraceContext, TracedService


class _RecordingClient(ObservabilityClient):
    """Captures end_* calls so tests can assert on them."""

    def __init__(self):
        super().__init__(stub=None, autostart=False)
        self.spans = []
        self.generations = []

    def end_span(self, span_id, *, status=StatusCode.OK, error_message="", **_kwargs):
        self.spans.append((span_id, status, error_message))
        # Drain pending lookup the parent stores so subsequent end_span calls don't error.
        self._pending_spans.pop(span_id, None)

    def end_generation(self, span_id, *, status=StatusCode.OK, error_message="", **_kwargs):
        self.generations.append((span_id, status, error_message))
        self._pending_generations.pop(span_id, None)


class _Svc(TracedService):
    pass


class TestTraceOperation:
    def test_yields_child_with_parent_link(self):
        svc = _Svc("test", _RecordingClient())
        ctx = TraceContext(trace_id="t-1", span_id="parent-1")
        with svc.trace_operation("step", ctx, foo="bar") as child:
            assert child.trace_id == "t-1"
            assert child.parent_span_id == "parent-1"
            assert child.span_id is not None and child.span_id != "parent-1"

    def test_status_ok_on_normal_exit(self):
        rec = _RecordingClient()
        svc = _Svc("test", rec)
        with svc.trace_operation("step", TraceContext(trace_id="t-1")):
            pass
        assert len(rec.spans) == 1
        _, status, msg = rec.spans[0]
        assert status == StatusCode.OK
        assert msg == ""

    def test_status_error_propagates_exception(self):
        rec = _RecordingClient()
        svc = _Svc("test", rec)
        with pytest.raises(ValueError, match="boom"):
            with svc.trace_operation("step", TraceContext(trace_id="t-1")):
                raise ValueError("boom")
        _, status, msg = rec.spans[0]
        assert status == StatusCode.ERROR
        assert "boom" in msg


class TestTraceGeneration:
    def test_yields_generation_handle(self):
        rec = _RecordingClient()
        svc = _Svc("models", rec)
        ctx = TraceContext(trace_id="t-1", span_id="parent-1")
        with svc.trace_generation(ctx, model="gpt-4o", requested_model="gpt-4o-mini") as gen:
            assert gen.model == "gpt-4o"
            assert gen.requested_model == "gpt-4o-mini"
            gen.update(prompt_tokens=10, completion_tokens=5, cost_usd=0.001)
        assert rec.generations
        _, status, _ = rec.generations[0]
        assert status == StatusCode.OK

    def test_error_in_generation_records_status(self):
        rec = _RecordingClient()
        svc = _Svc("models", rec)
        with pytest.raises(RuntimeError):
            with svc.trace_generation(TraceContext(trace_id="t-1"), model="gpt-4o"):
                raise RuntimeError("provider down")
        _, status, msg = rec.generations[0]
        assert status == StatusCode.ERROR
        assert "provider" in msg
