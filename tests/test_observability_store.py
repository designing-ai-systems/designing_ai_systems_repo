"""Unit tests for the in-memory ObservabilityStore."""

from datetime import datetime, timedelta, timezone

from services.observability.models import (
    BudgetAlert,
    Generation,
    LogEvent,
    MetricRecord,
    Score,
    Span,
)
from services.observability.store import InMemoryObservabilityStore


def _make_span(**kwargs):
    base = dict(
        trace_id="t1",
        span_id="s1",
        service="models",
        operation="models.chat",
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc) + timedelta(milliseconds=10),
    )
    base.update(kwargs)
    return Span(**base)


class TestSpansAndTraces:
    def test_record_spans_and_get_trace(self):
        store = InMemoryObservabilityStore()
        store.record_spans([_make_span(span_id="s1"), _make_span(span_id="s2")])
        trace = store.get_trace("t1")
        assert trace is not None
        assert len(trace.spans) == 2

    def test_get_trace_returns_none_for_unknown(self):
        store = InMemoryObservabilityStore()
        assert store.get_trace("missing") is None

    def test_query_traces_filter_by_workflow(self):
        store = InMemoryObservabilityStore()
        store.record_spans([_make_span(trace_id="t-a", attributes={"workflow_id": "w-a"})])
        store.record_spans([_make_span(trace_id="t-b", attributes={"workflow_id": "w-b"})])
        traces = store.query_traces(workflow_id="w-a")
        assert len(traces) == 1
        assert traces[0].trace_id == "t-a"

    def test_query_traces_filter_by_min_duration(self):
        store = InMemoryObservabilityStore()
        now = datetime.now(timezone.utc)
        store.record_spans(
            [
                _make_span(
                    trace_id="t-fast", start_time=now, end_time=now + timedelta(milliseconds=10)
                )
            ]
        )
        store.record_spans(
            [
                _make_span(
                    trace_id="t-slow", start_time=now, end_time=now + timedelta(milliseconds=500)
                )
            ]
        )
        slow_traces = store.query_traces(min_duration_ms=100)
        assert {t.trace_id for t in slow_traces} == {"t-slow"}


class TestLogs:
    def test_ingest_and_query_logs_filter(self):
        store = InMemoryObservabilityStore()
        store.ingest_logs(
            [
                LogEvent(trace_id="t1", event_type="model_call", severity="INFO", message="ok"),
                LogEvent(
                    trace_id="t1", event_type="model_fallback", severity="WARNING", message="fb"
                ),
                LogEvent(trace_id="t2", event_type="model_call", severity="INFO", message="ok2"),
            ]
        )
        events, cursor, total = store.query_logs(trace_id="t1", limit=10)
        assert total == 2
        assert all(e.trace_id == "t1" for e in events)

    def test_query_logs_min_severity(self):
        store = InMemoryObservabilityStore()
        store.ingest_logs(
            [
                LogEvent(severity="INFO", message="a"),
                LogEvent(severity="WARNING", message="b"),
                LogEvent(severity="ERROR", message="c"),
            ]
        )
        events, _, total = store.query_logs(min_severity="WARNING")
        assert total == 2

    def test_query_logs_cursor_pagination(self):
        store = InMemoryObservabilityStore()
        store.ingest_logs([LogEvent(message=str(i)) for i in range(7)])
        page1, cursor1, total = store.query_logs(limit=3)
        assert len(page1) == 3
        assert cursor1 != ""
        assert total == 7
        page2, cursor2, total2 = store.query_logs(limit=3, cursor=cursor1)
        assert len(page2) == 3
        assert cursor2 != ""
        page3, cursor3, _ = store.query_logs(limit=3, cursor=cursor2)
        assert len(page3) == 1
        assert cursor3 == ""


class TestMetrics:
    def test_record_and_query_metrics_percentiles(self):
        store = InMemoryObservabilityStore()
        records = [
            MetricRecord(name="latency_ms", type="HISTOGRAM", value=v) for v in range(1, 101)
        ]
        store.record_metrics(records)
        result = store.query_metrics(name="latency_ms", aggregation="p95")
        # p95 of 1..100 should be 96 (linear interpolation between idx 94 and 95).
        assert 94.5 <= result["value"] <= 96.5
        assert result["sample_count"] == 100
        assert "p50" in result["percentiles"]

    def test_query_metrics_with_label_filter(self):
        store = InMemoryObservabilityStore()
        store.record_metrics(
            [
                MetricRecord(name="rpm", value=1.0, labels={"workflow": "a"}),
                MetricRecord(name="rpm", value=2.0, labels={"workflow": "a"}),
                MetricRecord(name="rpm", value=3.0, labels={"workflow": "b"}),
            ]
        )
        a = store.query_metrics(name="rpm", label_filters={"workflow": "a"}, aggregation="sum")
        assert a["value"] == 3.0


class TestScores:
    def test_record_and_query_score(self):
        store = InMemoryObservabilityStore()
        score_id = store.record_score(
            Score(trace_id="t1", name="helpfulness", value=0.8, source="MODEL_JUDGE")
        )
        results = store.query_scores(trace_id="t1", name="helpfulness")
        assert len(results) == 1
        assert results[0].score_id == score_id


class TestCostReportAndBudget:
    def test_cost_report_grouping(self):
        store = InMemoryObservabilityStore()
        now = datetime.now(timezone.utc)
        store.record_generations(
            [
                Generation(
                    span=Span(trace_id="t1", start_time=now, attributes={"team": "eng"}),
                    model="gpt-4o",
                    prompt_tokens=100,
                    completion_tokens=50,
                    cost_usd=0.5,
                ),
                Generation(
                    span=Span(trace_id="t2", start_time=now, attributes={"team": "eng"}),
                    model="gpt-4o-mini",
                    prompt_tokens=10,
                    completion_tokens=10,
                    cost_usd=0.01,
                ),
                Generation(
                    span=Span(trace_id="t3", start_time=now, attributes={"team": "ds"}),
                    model="claude-haiku-4-5",
                    prompt_tokens=200,
                    completion_tokens=100,
                    cost_usd=0.2,
                ),
            ]
        )
        report = store.get_cost_report(
            start_time=now - timedelta(hours=1),
            end_time=now + timedelta(hours=1),
            group_by=["team"],
        )
        assert report.total_cost_usd == 0.71
        team_totals = {b.dimensions["team"]: b.cost_usd for b in report.buckets}
        assert team_totals["eng"] == 0.51
        assert team_totals["ds"] == 0.20

    def test_set_and_get_budget_status(self):
        store = InMemoryObservabilityStore()
        store.set_budget_alert(
            BudgetAlert(
                name="eng-monthly",
                scope_type="team",
                scope_value="eng",
                limit_usd=1.0,
                period="monthly",
            )
        )
        store.record_generations(
            [
                Generation(
                    span=Span(start_time=datetime.now(timezone.utc), attributes={"team": "eng"}),
                    model="gpt-4o",
                    cost_usd=0.7,
                )
            ]
        )
        status = store.get_budget_status("eng-monthly")
        assert status is not None
        assert status.alert.name == "eng-monthly"
        assert status.current_spend_usd >= 0.69
        assert status.percent_used >= 0.69
        assert any(t == 0.7 for t in status.thresholds_crossed)


class TestServiceHealth:
    def test_health_returns_unknown_without_spans(self):
        store = InMemoryObservabilityStore()
        health = store.get_service_health("models")
        assert health.status == "unknown"

    def test_health_reports_error_rate(self):
        store = InMemoryObservabilityStore()
        store.record_spans(
            [
                _make_span(span_id="ok-1", status="OK"),
                _make_span(span_id="ok-2", status="OK"),
                _make_span(span_id="err-1", status="ERROR"),
            ]
        )
        health = store.get_service_health("models")
        assert health.span_count == 3
        assert 0.3 <= health.error_rate <= 0.4
