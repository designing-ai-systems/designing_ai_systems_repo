"""
Observability Service — gRPC service implementation (grpc.aio).

Thin proto<->domain translation layer. The store does the work; the
servicer just unwraps requests, calls into the store, and packs the
results back into proto messages.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 7.1: ObservabilityService gRPC contract
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import grpc
import grpc.aio
from google.protobuf.timestamp_pb2 import Timestamp

from proto import observability_pb2, observability_pb2_grpc
from services.observability.models import (
    BudgetAlert,
    Generation,
    LogEvent,
    MetricRecord,
    Score,
    Span,
    SpanEvent,
)
from services.observability.store import (
    InMemoryObservabilityStore,
    ObservabilityStore,
)
from services.shared.servicer_base import BaseAioServicer

logger = logging.getLogger(__name__)


_SEVERITY_TO_PROTO = {
    "DEBUG": observability_pb2.DEBUG,
    "INFO": observability_pb2.INFO,
    "WARNING": observability_pb2.WARNING,
    "ERROR": observability_pb2.ERROR,
    "CRITICAL": observability_pb2.CRITICAL,
}
_SEVERITY_FROM_PROTO = {v: k for k, v in _SEVERITY_TO_PROTO.items()}

_STATUS_TO_PROTO = {
    "OK": observability_pb2.OK,
    "ERROR": observability_pb2.SPAN_ERROR,
}
_STATUS_FROM_PROTO = {
    observability_pb2.OK: "OK",
    observability_pb2.SPAN_ERROR: "ERROR",
}

_METRIC_TO_PROTO = {
    "COUNTER": observability_pb2.COUNTER,
    "HISTOGRAM": observability_pb2.HISTOGRAM,
}
_METRIC_FROM_PROTO = {
    observability_pb2.COUNTER: "COUNTER",
    observability_pb2.HISTOGRAM: "HISTOGRAM",
}

_SCORE_SOURCE_TO_PROTO = {
    "AUTOMATED": observability_pb2.AUTOMATED,
    "MODEL_JUDGE": observability_pb2.MODEL_JUDGE,
    "HUMAN": observability_pb2.HUMAN,
    "USER_FEEDBACK": observability_pb2.USER_FEEDBACK,
}
_SCORE_SOURCE_FROM_PROTO = {v: k for k, v in _SCORE_SOURCE_TO_PROTO.items()}


def _ts_to_dt(ts: Optional[Timestamp]) -> Optional[datetime]:
    if ts is None or (ts.seconds == 0 and ts.nanos == 0):
        return None
    return datetime.fromtimestamp(ts.seconds + ts.nanos / 1e9, tz=timezone.utc)


def _dt_to_ts(dt: Optional[datetime]) -> Optional[Timestamp]:
    if dt is None:
        return None
    ts = Timestamp()
    ts.FromDatetime(dt)
    return ts


class ObservabilityServiceImpl(
    observability_pb2_grpc.ObservabilityServiceServicer, BaseAioServicer
):
    """Observability gRPC servicer (Listing 7.1)."""

    def __init__(self, store: Optional[ObservabilityStore] = None) -> None:
        self.store: ObservabilityStore = store or InMemoryObservabilityStore()

    def add_to_aio_server(self, server: grpc.aio.Server) -> None:
        observability_pb2_grpc.add_ObservabilityServiceServicer_to_server(self, server)

    # ------------------------------------------------------------------
    # Logs
    # ------------------------------------------------------------------
    async def IngestLogs(self, request, context):
        events = [self._proto_log_to_domain(e) for e in request.events]
        ingested = self.store.ingest_logs(events)
        return observability_pb2.IngestLogsResponse(ingested=ingested)

    async def QueryLogs(self, request, context):
        events, next_cursor, total = self.store.query_logs(
            trace_id=request.trace_id,
            span_id=request.span_id,
            service=request.service,
            event_type=request.event_type,
            min_severity=_SEVERITY_FROM_PROTO.get(request.min_severity, ""),
            attribute_filters=dict(request.attribute_filters),
            start_time=_ts_to_dt(request.start_time),
            end_time=_ts_to_dt(request.end_time),
            limit=request.limit or 100,
            cursor=request.cursor,
        )
        return observability_pb2.QueryLogsResponse(
            events=[self._domain_log_to_proto(e) for e in events],
            next_cursor=next_cursor,
            total_matched=total,
        )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    async def RecordMetrics(self, request, context):
        records = [self._proto_metric_to_domain(r) for r in request.records]
        recorded = self.store.record_metrics(records)
        return observability_pb2.RecordMetricsResponse(recorded=recorded)

    async def QueryMetrics(self, request, context):
        result = self.store.query_metrics(
            name=request.name,
            label_filters=dict(request.label_filters),
            start_time=_ts_to_dt(request.start_time),
            end_time=_ts_to_dt(request.end_time),
            aggregation=request.aggregation or "sum",
        )
        return observability_pb2.QueryMetricsResponse(
            name=result["name"],
            aggregation=result["aggregation"],
            value=result["value"],
            sample_count=result["sample_count"],
            percentiles=result["percentiles"],
        )

    # ------------------------------------------------------------------
    # Spans / generations / traces
    # ------------------------------------------------------------------
    async def RecordSpan(self, request, context):
        spans = [self._proto_span_to_domain(s) for s in request.spans]
        recorded = self.store.record_spans(spans)
        return observability_pb2.RecordSpanResponse(recorded=recorded)

    async def RecordGeneration(self, request, context):
        generations = [self._proto_generation_to_domain(g) for g in request.generations]
        recorded = self.store.record_generations(generations)
        return observability_pb2.RecordGenerationResponse(recorded=recorded)

    async def GetTrace(self, request, context):
        trace = self.store.get_trace(request.trace_id)
        if trace is None:
            return observability_pb2.Trace(trace_id=request.trace_id)
        return self._domain_trace_to_proto(trace)

    async def QueryTraces(self, request, context):
        traces = self.store.query_traces(
            workflow_id=request.workflow_id or None,
            user_id=request.user_id or None,
            session_id=request.session_id or None,
            tags=list(request.tags) or None,
            start_time=_ts_to_dt(request.start_time),
            end_time=_ts_to_dt(request.end_time),
            min_duration_ms=request.min_duration_ms or None,
            max_duration_ms=request.max_duration_ms or None,
            min_score=dict(request.min_score) or None,
            max_score=dict(request.max_score) or None,
            limit=request.limit or 100,
        )
        return observability_pb2.QueryTracesResponse(
            traces=[self._domain_trace_to_proto(t) for t in traces],
            total_matched=len(traces),
        )

    # ------------------------------------------------------------------
    # Scores
    # ------------------------------------------------------------------
    async def RecordScore(self, request, context):
        score = self._proto_score_to_domain(request.score)
        score_id = self.store.record_score(score)
        return observability_pb2.RecordScoreResponse(score_id=score_id)

    async def QueryScores(self, request, context):
        scores = self.store.query_scores(
            trace_id=request.trace_id,
            name=request.name,
            source=_SCORE_SOURCE_FROM_PROTO.get(request.source, ""),
            start_time=_ts_to_dt(request.start_time),
            end_time=_ts_to_dt(request.end_time),
            limit=request.limit or 100,
        )
        return observability_pb2.QueryScoresResponse(
            scores=[self._domain_score_to_proto(s) for s in scores]
        )

    # ------------------------------------------------------------------
    # Cost & budgets
    # ------------------------------------------------------------------
    async def GetCostReport(self, request, context):
        start = _ts_to_dt(request.start_time) or datetime.now(timezone.utc)
        end = _ts_to_dt(request.end_time) or datetime.now(timezone.utc)
        report = self.store.get_cost_report(
            start_time=start,
            end_time=end,
            group_by=list(request.group_by),
            filters=dict(request.filters),
        )
        proto_report = observability_pb2.CostReport(
            group_by=list(report.group_by),
            buckets=[
                observability_pb2.CostBucket(
                    dimensions=b.dimensions,
                    cost_usd=b.cost_usd,
                    prompt_tokens=b.prompt_tokens,
                    completion_tokens=b.completion_tokens,
                    request_count=b.request_count,
                )
                for b in report.buckets
            ],
            total_cost_usd=report.total_cost_usd,
        )
        if report.start_time:
            proto_report.start_time.CopyFrom(_dt_to_ts(report.start_time))
        if report.end_time:
            proto_report.end_time.CopyFrom(_dt_to_ts(report.end_time))
        return proto_report

    async def SetBudgetAlert(self, request, context):
        alert = BudgetAlert(
            name=request.alert.name,
            scope_type=request.alert.scope_type or "team",
            scope_value=request.alert.scope_value,
            limit_usd=request.alert.limit_usd,
            period=request.alert.period or "monthly",
            thresholds=list(request.alert.thresholds) or [0.7, 0.9, 1.0],
            notification_channels=list(request.alert.notification_channels),
        )
        name = self.store.set_budget_alert(alert)
        return observability_pb2.SetBudgetAlertResponse(name=name)

    async def GetBudgetStatus(self, request, context):
        status = self.store.get_budget_status(request.name)
        if status is None:
            return observability_pb2.BudgetStatus()
        return observability_pb2.BudgetStatus(
            alert=observability_pb2.BudgetAlert(
                name=status.alert.name,
                scope_type=status.alert.scope_type,
                scope_value=status.alert.scope_value,
                limit_usd=status.alert.limit_usd,
                period=status.alert.period,
                thresholds=status.alert.thresholds,
                notification_channels=status.alert.notification_channels,
            ),
            current_spend_usd=status.current_spend_usd,
            projected_spend_usd=status.projected_spend_usd,
            percent_used=status.percent_used,
            thresholds_crossed=status.thresholds_crossed,
        )

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------
    async def GetServiceHealth(self, request, context):
        health = self.store.get_service_health(
            service=request.service,
            lookback_seconds=request.lookback_seconds or 600,
        )
        resp = observability_pb2.ServiceHealthResponse(
            service=health.service,
            status=health.status,
            span_count=health.span_count,
            error_rate=health.error_rate,
            detail=health.detail,
        )
        if health.last_span_at is not None:
            resp.last_span_at.CopyFrom(_dt_to_ts(health.last_span_at))
        return resp

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------
    def _proto_log_to_domain(self, event) -> LogEvent:
        return LogEvent(
            event_id=event.event_id,
            trace_id=event.trace_id,
            span_id=event.span_id,
            timestamp=_ts_to_dt(event.timestamp) or datetime.now(timezone.utc),
            service=event.service,
            severity=_SEVERITY_FROM_PROTO.get(event.severity, "INFO"),
            event_type=event.event_type,
            message=event.message,
            attributes=dict(event.attributes),
            numeric_attributes=dict(event.numeric_attributes),
            workflow_id=event.workflow_id,
            user_id=event.user_id,
        )

    def _domain_log_to_proto(self, event: LogEvent):
        proto = observability_pb2.LogEvent(
            event_id=event.event_id,
            trace_id=event.trace_id,
            span_id=event.span_id,
            service=event.service,
            severity=_SEVERITY_TO_PROTO.get(event.severity.upper(), observability_pb2.INFO),
            event_type=event.event_type,
            message=event.message,
            attributes=event.attributes,
            numeric_attributes=event.numeric_attributes,
            workflow_id=event.workflow_id,
            user_id=event.user_id,
        )
        if event.timestamp:
            proto.timestamp.CopyFrom(_dt_to_ts(event.timestamp))
        return proto

    def _proto_metric_to_domain(self, record) -> MetricRecord:
        return MetricRecord(
            name=record.name,
            type=_METRIC_FROM_PROTO.get(record.type, "COUNTER"),
            value=record.value,
            labels=dict(record.labels),
            timestamp=_ts_to_dt(record.timestamp) or datetime.now(timezone.utc),
        )

    def _proto_span_to_domain(self, span) -> Span:
        events = [
            SpanEvent(
                name=ev.name,
                timestamp=_ts_to_dt(ev.timestamp) or datetime.now(timezone.utc),
                attributes=dict(ev.attributes),
            )
            for ev in span.events
        ]
        return Span(
            trace_id=span.trace_id,
            span_id=span.span_id,
            parent_span_id=span.parent_span_id,
            service=span.service,
            operation=span.operation,
            start_time=_ts_to_dt(span.start_time) or datetime.now(timezone.utc),
            end_time=_ts_to_dt(span.end_time),
            status=_STATUS_FROM_PROTO.get(span.status, "OK"),
            error_message=span.error_message,
            attributes=dict(span.attributes),
            numeric_attributes=dict(span.numeric_attributes),
            events=events,
        )

    def _domain_span_to_proto(self, span: Span):
        proto = observability_pb2.Span(
            trace_id=span.trace_id,
            span_id=span.span_id,
            parent_span_id=span.parent_span_id,
            service=span.service,
            operation=span.operation,
            status=_STATUS_TO_PROTO.get(span.status, observability_pb2.OK),
            error_message=span.error_message,
            attributes=span.attributes,
            numeric_attributes=span.numeric_attributes,
        )
        if span.start_time:
            proto.start_time.CopyFrom(_dt_to_ts(span.start_time))
        if span.end_time:
            proto.end_time.CopyFrom(_dt_to_ts(span.end_time))
        for ev in span.events:
            proto_ev = proto.events.add()
            proto_ev.name = ev.name
            if ev.timestamp:
                proto_ev.timestamp.CopyFrom(_dt_to_ts(ev.timestamp))
            for k, v in ev.attributes.items():
                proto_ev.attributes[k] = v
        return proto

    def _proto_generation_to_domain(self, gen) -> Generation:
        return Generation(
            span=self._proto_span_to_domain(gen.span),
            model=gen.model,
            provider=gen.provider,
            requested_model=gen.requested_model,
            prompt_tokens=gen.prompt_tokens,
            completion_tokens=gen.completion_tokens,
            cost_usd=gen.cost_usd,
            cache_hit=gen.cache_hit,
            fallback_used=gen.fallback_used,
            time_to_first_token_ms=gen.time_to_first_token_ms,
        )

    def _domain_generation_to_proto(self, gen: Generation):
        proto = observability_pb2.Generation(
            model=gen.model,
            provider=gen.provider,
            requested_model=gen.requested_model,
            prompt_tokens=gen.prompt_tokens,
            completion_tokens=gen.completion_tokens,
            cost_usd=gen.cost_usd,
            cache_hit=gen.cache_hit,
            fallback_used=gen.fallback_used,
            time_to_first_token_ms=gen.time_to_first_token_ms,
        )
        proto.span.CopyFrom(self._domain_span_to_proto(gen.span))
        return proto

    def _proto_score_to_domain(self, score) -> Score:
        kind = score.WhichOneof("value")
        if kind == "numeric_value":
            value = score.numeric_value
        elif kind == "categorical_value":
            value = score.categorical_value
        elif kind == "boolean_value":
            value = score.boolean_value
        else:
            value = None
        return Score(
            score_id=score.score_id,
            trace_id=score.trace_id,
            span_id=score.span_id,
            generation_id=score.generation_id,
            name=score.name,
            value=value,
            source=_SCORE_SOURCE_FROM_PROTO.get(score.source, "AUTOMATED"),
            comment=score.comment,
            metadata=dict(score.metadata),
            timestamp=_ts_to_dt(score.timestamp) or datetime.now(timezone.utc),
        )

    def _domain_score_to_proto(self, score: Score):
        proto = observability_pb2.Score(
            score_id=score.score_id,
            trace_id=score.trace_id,
            span_id=score.span_id,
            generation_id=score.generation_id,
            name=score.name,
            source=_SCORE_SOURCE_TO_PROTO.get(score.source, observability_pb2.AUTOMATED),
            comment=score.comment,
            metadata=score.metadata,
        )
        if score.timestamp:
            proto.timestamp.CopyFrom(_dt_to_ts(score.timestamp))
        if isinstance(score.value, bool):
            proto.boolean_value = score.value
        elif isinstance(score.value, (int, float)):
            proto.numeric_value = float(score.value)
        elif isinstance(score.value, str):
            proto.categorical_value = score.value
        return proto

    def _domain_trace_to_proto(self, trace):
        proto = observability_pb2.Trace(
            trace_id=trace.trace_id,
            session_id=trace.session_id,
            workflow_id=trace.workflow_id,
            user_id=trace.user_id,
            input=trace.input,
            output=trace.output,
            total_duration_ms=trace.total_duration_ms,
            total_cost_usd=trace.total_cost_usd,
            total_tokens=trace.total_tokens,
            tags=trace.tags,
        )
        for span in trace.spans:
            proto.spans.add().CopyFrom(self._domain_span_to_proto(span))
        for gen in trace.generations:
            proto.generations.add().CopyFrom(self._domain_generation_to_proto(gen))
        for score in trace.scores:
            proto.scores.add().CopyFrom(self._domain_score_to_proto(score))
        return proto
