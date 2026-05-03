"""In-process tests for ObservabilityServiceImpl (Listing 7.1)."""

from datetime import datetime, timedelta, timezone

from proto import observability_pb2
from services.observability.service import ObservabilityServiceImpl


class FakeContext:
    def __init__(self):
        self.code = None
        self.details_str = None

    def set_code(self, code):
        self.code = code

    def set_details(self, details):
        self.details_str = details


def _make_span(trace_id="t1", span_id="s1", *, status=observability_pb2.OK, attrs=None):
    span = observability_pb2.Span(
        trace_id=trace_id,
        span_id=span_id,
        service="models",
        operation="models.chat",
        status=status,
    )
    span.start_time.GetCurrentTime()
    span.end_time.GetCurrentTime()
    if attrs:
        for k, v in attrs.items():
            span.attributes[k] = v
    return span


class TestSpansAndTrace:
    async def test_record_then_get_trace(self):
        svc = ObservabilityServiceImpl()
        await svc.RecordSpan(
            observability_pb2.RecordSpanRequest(
                spans=[_make_span(span_id="s1"), _make_span(span_id="s2")]
            ),
            FakeContext(),
        )
        trace = await svc.GetTrace(observability_pb2.GetTraceRequest(trace_id="t1"), FakeContext())
        assert trace.trace_id == "t1"
        assert len(trace.spans) == 2

    async def test_record_generation_populates_trace_cost(self):
        svc = ObservabilityServiceImpl()
        gen = observability_pb2.Generation(
            model="gpt-4o", prompt_tokens=100, completion_tokens=20, cost_usd=0.005
        )
        gen.span.CopyFrom(_make_span(trace_id="t-gen", span_id="g1"))
        await svc.RecordGeneration(
            observability_pb2.RecordGenerationRequest(generations=[gen]), FakeContext()
        )
        trace = await svc.GetTrace(
            observability_pb2.GetTraceRequest(trace_id="t-gen"), FakeContext()
        )
        assert trace.total_cost_usd == 0.005
        assert trace.total_tokens == 120


class TestLogs:
    async def test_ingest_then_query_by_trace(self):
        svc = ObservabilityServiceImpl()
        log = observability_pb2.LogEvent(
            trace_id="t1", event_type="model_call", severity=observability_pb2.INFO, message="ok"
        )
        log.timestamp.GetCurrentTime()
        await svc.IngestLogs(observability_pb2.IngestLogsRequest(events=[log]), FakeContext())
        resp = await svc.QueryLogs(observability_pb2.QueryLogsRequest(trace_id="t1"), FakeContext())
        assert resp.total_matched == 1
        assert resp.events[0].message == "ok"


class TestMetrics:
    async def test_record_and_query_p95(self):
        svc = ObservabilityServiceImpl()
        records = []
        for v in range(1, 101):
            mr = observability_pb2.MetricRecord(
                name="latency", type=observability_pb2.HISTOGRAM, value=float(v)
            )
            mr.timestamp.GetCurrentTime()
            records.append(mr)
        await svc.RecordMetrics(
            observability_pb2.RecordMetricsRequest(records=records), FakeContext()
        )
        resp = await svc.QueryMetrics(
            observability_pb2.QueryMetricsRequest(name="latency", aggregation="p95"),
            FakeContext(),
        )
        assert resp.sample_count == 100
        assert 94 <= resp.value <= 96


class TestScores:
    async def test_record_and_query_score_round_trip(self):
        svc = ObservabilityServiceImpl()
        score = observability_pb2.Score(
            trace_id="t1",
            name="helpfulness",
            source=observability_pb2.MODEL_JUDGE,
            numeric_value=0.93,
        )
        score.timestamp.GetCurrentTime()
        recorded = await svc.RecordScore(
            observability_pb2.RecordScoreRequest(score=score), FakeContext()
        )
        assert recorded.score_id

        resp = await svc.QueryScores(
            observability_pb2.QueryScoresRequest(trace_id="t1", name="helpfulness"),
            FakeContext(),
        )
        assert len(resp.scores) == 1
        assert resp.scores[0].numeric_value == 0.93
        assert resp.scores[0].source == observability_pb2.MODEL_JUDGE


class TestCostReport:
    async def test_cost_report_grouped_by_team(self):
        svc = ObservabilityServiceImpl()
        gens = []
        for team, cost in [("eng", 0.5), ("eng", 0.2), ("ds", 0.1)]:
            gen = observability_pb2.Generation(model="gpt-4o", cost_usd=cost)
            gen.span.CopyFrom(_make_span(trace_id=f"t-{team}-{cost}", attrs={"team": team}))
            gens.append(gen)
        await svc.RecordGeneration(
            observability_pb2.RecordGenerationRequest(generations=gens), FakeContext()
        )

        request = observability_pb2.CostReportRequest(group_by=["team"])
        request.start_time.FromDatetime(datetime.now(timezone.utc) - timedelta(hours=1))
        request.end_time.FromDatetime(datetime.now(timezone.utc) + timedelta(hours=1))
        report = await svc.GetCostReport(request, FakeContext())
        team_totals = {b.dimensions["team"]: b.cost_usd for b in report.buckets}
        assert team_totals["eng"] == 0.7
        assert team_totals["ds"] == 0.1


class TestServiceHealth:
    async def test_health_unknown_no_data(self):
        svc = ObservabilityServiceImpl()
        resp = await svc.GetServiceHealth(
            observability_pb2.ServiceHealthRequest(service="models"), FakeContext()
        )
        assert resp.status == "unknown"
