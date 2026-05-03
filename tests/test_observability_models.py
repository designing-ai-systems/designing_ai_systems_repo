"""Unit tests for observability domain dataclasses (Listings 7.2, 7.5, 7.11)."""

from datetime import datetime, timedelta, timezone

from services.observability.models import (
    Generation,
    LogEvent,
    Score,
    Span,
    assemble_trace,
)


class TestSpanBasics:
    def test_default_status_ok(self):
        span = Span(trace_id="t", span_id="s", service="models")
        assert span.status == "OK"
        assert span.error_message == ""
        assert span.attributes == {}
        assert span.events == []

    def test_duration_ms(self):
        start = datetime.now(timezone.utc)
        end = start + timedelta(milliseconds=250)
        span = Span(start_time=start, end_time=end)
        assert span.duration_ms == 250.0

    def test_duration_zero_when_open(self):
        span = Span(start_time=datetime.now(timezone.utc))
        assert span.duration_ms == 0.0


class TestScoreOneOf:
    def test_score_can_be_numeric(self):
        s = Score(name="helpfulness", value=0.85, source="MODEL_JUDGE")
        assert s.value == 0.85
        assert s.source == "MODEL_JUDGE"

    def test_score_can_be_categorical(self):
        s = Score(name="tone", value="warm", source="HUMAN")
        assert s.value == "warm"

    def test_score_can_be_boolean(self):
        s = Score(name="resolved", value=True, source="USER_FEEDBACK")
        assert s.value is True


class TestLogEvent:
    def test_default_severity(self):
        e = LogEvent(event_type="model_call")
        assert e.severity == "INFO"


class TestAssembleTrace:
    def test_aggregates_spans_and_generations(self):
        start = datetime.now(timezone.utc)
        s1 = Span(
            trace_id="t1",
            span_id="s1",
            start_time=start,
            end_time=start + timedelta(milliseconds=100),
            attributes={"workflow_id": "patient-intake"},
        )
        gen_span = Span(
            trace_id="t1",
            span_id="s2",
            start_time=start + timedelta(milliseconds=20),
            end_time=start + timedelta(milliseconds=80),
        )
        gen = Generation(
            span=gen_span, model="gpt-4o", prompt_tokens=10, completion_tokens=5, cost_usd=0.001
        )
        trace = assemble_trace("t1", [s1], [gen], [])
        assert trace.trace_id == "t1"
        assert trace.workflow_id == "patient-intake"
        assert trace.total_duration_ms == 100.0
        assert trace.total_cost_usd == 0.001
        assert trace.total_tokens == 15

    def test_handles_no_spans(self):
        trace = assemble_trace("t-empty", [], [], [])
        assert trace.trace_id == "t-empty"
        assert trace.total_duration_ms == 0.0
