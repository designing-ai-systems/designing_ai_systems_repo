"""
Observability Service client.

Surfaces the Observability Service through ``platform.observability``.
The SDK client buffers spans, generations, logs, and metrics locally
and flushes them to the Observability Service via the gateway, so
telemetry collection never adds a network round-trip to the user's
request path.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 7.1: ObservabilityService gRPC contract
  - Listing 7.9: client batching/buffering
  - Listing 7.10: custom span via SDK (trace_operation)
  - Listing 7.13: cost drill-down (get_cost_report)
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Union

from google.protobuf.timestamp_pb2 import Timestamp

from proto import observability_pb2, observability_pb2_grpc
from services.observability.models import (
    BudgetAlert,
    BudgetStatus,
    CostBucket,
    CostReport,
    Generation,
    LogEvent,
    Score,
    ServiceHealth,
    Span,
    SpanEvent,
    Trace,
)
from services.shared.observability_client import ObservabilityClient as InternalObservabilityClient
from services.shared.traced_service import TraceContext

from .base import BaseClient

_SCORE_SOURCE_TO_PROTO = {
    "AUTOMATED": observability_pb2.AUTOMATED,
    "MODEL_JUDGE": observability_pb2.MODEL_JUDGE,
    "HUMAN": observability_pb2.HUMAN,
    "USER_FEEDBACK": observability_pb2.USER_FEEDBACK,
}
_SCORE_SOURCE_FROM_PROTO = {v: k for k, v in _SCORE_SOURCE_TO_PROTO.items()}

_STATUS_FROM_PROTO = {
    observability_pb2.OK: "OK",
    observability_pb2.SPAN_ERROR: "ERROR",
}

_METRIC_FROM_PROTO = {
    observability_pb2.COUNTER: "COUNTER",
    observability_pb2.HISTOGRAM: "HISTOGRAM",
}
_METRIC_TO_PROTO = {v: k for k, v in _METRIC_FROM_PROTO.items()}

_SEVERITY_TO_PROTO = {
    "DEBUG": observability_pb2.DEBUG,
    "INFO": observability_pb2.INFO,
    "WARNING": observability_pb2.WARNING,
    "ERROR": observability_pb2.ERROR,
    "CRITICAL": observability_pb2.CRITICAL,
}
_SEVERITY_FROM_PROTO = {v: k for k, v in _SEVERITY_TO_PROTO.items()}


def _now_ts() -> Timestamp:
    ts = Timestamp()
    ts.GetCurrentTime()
    return ts


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


class _MetadataInjectingStub:
    """Wraps an Observability stub so every method call carries gateway metadata.

    The internal buffered client calls ``stub.RecordSpan(request)`` without
    any keyword arguments; the gateway, however, needs ``x-target-service``
    metadata to know which backend to route to. This thin proxy injects the
    metadata so the buffered client doesn't have to know about routing.
    """

    def __init__(self, stub, metadata):
        self._stub = stub
        self._metadata = metadata

    def __getattr__(self, name):
        method = getattr(self._stub, name)

        def wrapped(request, *, metadata=None, **kwargs):
            return method(request, metadata=self._metadata, **kwargs)

        return wrapped


class ObservabilityClient(BaseClient):
    """SDK client for the Observability Service."""

    def __init__(self, platform):
        super().__init__(platform, "observability")
        self._stub = observability_pb2_grpc.ObservabilityServiceStub(self._channel)
        # Internal buffered client for span/generation/log/metric publication.
        # The SDK's wrapper just routes to this and exposes query helpers.
        self._buffered = InternalObservabilityClient(
            stub=_MetadataInjectingStub(self._stub, self.metadata),
            service_name="sdk",
            autostart=True,
        )

    # ------------------------------------------------------------------
    # Tracing — Listing 7.6 / 7.10
    # ------------------------------------------------------------------
    @contextmanager
    def trace_operation(
        self,
        operation: str,
        trace_context: Optional[TraceContext] = None,
        **attributes: Any,
    ) -> Iterator[TraceContext]:
        """Open a custom span (Listing 7.10).

        Workflow developers use this to instrument application-specific
        steps that platform services don't cover (custom reranking,
        business rules, external API calls).
        """
        ctx = trace_context or TraceContext(trace_id=uuid.uuid4().hex)
        span = self._buffered.start_span(
            trace_id=ctx.trace_id,
            parent_span_id=ctx.span_id,
            service="sdk",
            operation=operation,
            attributes=attributes,
        )
        child = TraceContext(
            trace_id=ctx.trace_id,
            span_id=span.span_id,
            parent_span_id=ctx.span_id,
            workflow_id=ctx.workflow_id,
            user_id=ctx.user_id,
            session_id=ctx.session_id,
            tags=list(ctx.tags),
        )
        try:
            yield child
        except Exception as exc:
            self._buffered.end_span(span.span_id, status="ERROR", error_message=str(exc))  # type: ignore[arg-type]
            raise
        self._buffered.end_span(span.span_id)

    def flush(self) -> None:
        """Force a flush of all buffered telemetry."""
        self._buffered.flush_all()

    def shutdown(self) -> None:
        """Stop the background flush thread and ship anything left."""
        self._buffered.stop(flush=True)

    # ------------------------------------------------------------------
    # Direct span/generation recording (used by tests + advanced users)
    # ------------------------------------------------------------------
    def record_span(self, span: Span) -> None:
        proto = self._domain_span_to_proto(span)
        self._stub.RecordSpan(
            observability_pb2.RecordSpanRequest(spans=[proto]), metadata=self.metadata
        )

    def record_generation(self, generation: Generation) -> None:
        proto = self._domain_generation_to_proto(generation)
        self._stub.RecordGeneration(
            observability_pb2.RecordGenerationRequest(generations=[proto]),
            metadata=self.metadata,
        )

    def log(
        self,
        event_type: str,
        message: str,
        *,
        severity: str = "INFO",
        trace_id: str = "",
        span_id: str = "",
        attributes: Optional[Dict[str, str]] = None,
        workflow_id: str = "",
        user_id: str = "",
    ) -> None:
        self._buffered.log(
            event_type=event_type,
            severity=severity,
            message=message,
            trace_id=trace_id,
            span_id=span_id,
            attributes=attributes,
            workflow_id=workflow_id,
            user_id=user_id,
        )

    def record_counter(
        self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None
    ) -> None:
        self._buffered.record_counter(name, value, labels)

    def record_histogram(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        self._buffered.record_histogram(name, value, labels)

    def record_score(
        self,
        *,
        trace_id: str,
        name: str,
        value: Union[float, str, bool],
        source: str = "AUTOMATED",
        span_id: str = "",
        generation_id: str = "",
        comment: str = "",
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        proto = observability_pb2.Score(
            trace_id=trace_id,
            span_id=span_id,
            generation_id=generation_id,
            name=name,
            comment=comment,
            metadata=metadata or {},
        )
        proto.source = _SCORE_SOURCE_TO_PROTO.get(source, observability_pb2.AUTOMATED)
        if isinstance(value, bool):
            proto.boolean_value = value
        elif isinstance(value, (int, float)):
            proto.numeric_value = float(value)
        elif isinstance(value, str):
            proto.categorical_value = value
        proto.timestamp.CopyFrom(_now_ts())
        response = self._stub.RecordScore(
            observability_pb2.RecordScoreRequest(score=proto), metadata=self.metadata
        )
        return response.score_id

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def get_trace(self, trace_id: str) -> Optional[Trace]:
        # Make sure any pending writes are visible to the read.
        self._buffered.flush_all()
        proto = self._stub.GetTrace(
            observability_pb2.GetTraceRequest(trace_id=trace_id), metadata=self.metadata
        )
        if not proto.trace_id:
            return None
        return self._proto_trace_to_domain(proto)

    def query_traces(
        self,
        *,
        workflow_id: str = "",
        user_id: str = "",
        session_id: str = "",
        tags: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        min_duration_ms: float = 0.0,
        max_duration_ms: float = 0.0,
        min_score: Optional[Dict[str, float]] = None,
        max_score: Optional[Dict[str, float]] = None,
        limit: int = 100,
    ) -> List[Trace]:
        self._buffered.flush_all()
        request = observability_pb2.QueryTracesRequest(
            workflow_id=workflow_id,
            user_id=user_id,
            session_id=session_id,
            tags=tags or [],
            min_duration_ms=min_duration_ms,
            max_duration_ms=max_duration_ms,
            min_score=min_score or {},
            max_score=max_score or {},
            limit=limit,
        )
        if start_time:
            request.start_time.CopyFrom(_dt_to_ts(start_time))
        if end_time:
            request.end_time.CopyFrom(_dt_to_ts(end_time))
        response = self._stub.QueryTraces(request, metadata=self.metadata)
        return [self._proto_trace_to_domain(t) for t in response.traces]

    def query_logs(
        self,
        *,
        trace_id: str = "",
        span_id: str = "",
        service: str = "",
        event_type: str = "",
        min_severity: str = "",
        attribute_filters: Optional[Dict[str, str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        cursor: str = "",
    ) -> Dict[str, Any]:
        self._buffered.flush_all()
        request = observability_pb2.QueryLogsRequest(
            trace_id=trace_id,
            span_id=span_id,
            service=service,
            event_type=event_type,
            attribute_filters=attribute_filters or {},
            limit=limit,
            cursor=cursor,
        )
        if min_severity:
            request.min_severity = _SEVERITY_TO_PROTO.get(
                min_severity.upper(), observability_pb2.INFO
            )
        if start_time:
            request.start_time.CopyFrom(_dt_to_ts(start_time))
        if end_time:
            request.end_time.CopyFrom(_dt_to_ts(end_time))
        response = self._stub.QueryLogs(request, metadata=self.metadata)
        return {
            "events": [self._proto_log_to_domain(e) for e in response.events],
            "next_cursor": response.next_cursor,
            "total_matched": response.total_matched,
        }

    def query_metrics(
        self,
        *,
        name: str,
        label_filters: Optional[Dict[str, str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        aggregation: str = "sum",
    ) -> Dict[str, Any]:
        self._buffered.flush_all()
        request = observability_pb2.QueryMetricsRequest(
            name=name,
            label_filters=label_filters or {},
            aggregation=aggregation,
        )
        if start_time:
            request.start_time.CopyFrom(_dt_to_ts(start_time))
        if end_time:
            request.end_time.CopyFrom(_dt_to_ts(end_time))
        response = self._stub.QueryMetrics(request, metadata=self.metadata)
        return {
            "name": response.name,
            "aggregation": response.aggregation,
            "value": response.value,
            "sample_count": response.sample_count,
            "percentiles": dict(response.percentiles),
        }

    def query_scores(
        self,
        *,
        trace_id: str = "",
        name: str = "",
        source: str = "",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Score]:
        self._buffered.flush_all()
        request = observability_pb2.QueryScoresRequest(
            trace_id=trace_id,
            name=name,
            limit=limit,
        )
        if source:
            request.source = _SCORE_SOURCE_TO_PROTO.get(source.upper(), observability_pb2.AUTOMATED)
        if start_time:
            request.start_time.CopyFrom(_dt_to_ts(start_time))
        if end_time:
            request.end_time.CopyFrom(_dt_to_ts(end_time))
        response = self._stub.QueryScores(request, metadata=self.metadata)
        return [self._proto_score_to_domain(s) for s in response.scores]

    # ------------------------------------------------------------------
    # Cost / budgets / health
    # ------------------------------------------------------------------
    def get_cost_report(
        self,
        *,
        start_time: datetime,
        end_time: datetime,
        group_by: Optional[List[str]] = None,
        filters: Optional[Dict[str, str]] = None,
        granularity: str = "monthly",
    ) -> CostReport:
        self._buffered.flush_all()
        request = observability_pb2.CostReportRequest(
            group_by=group_by or [],
            filters=filters or {},
            granularity=granularity,
        )
        request.start_time.CopyFrom(_dt_to_ts(start_time))
        request.end_time.CopyFrom(_dt_to_ts(end_time))
        response = self._stub.GetCostReport(request, metadata=self.metadata)
        buckets = [
            CostBucket(
                dimensions=dict(b.dimensions),
                cost_usd=b.cost_usd,
                prompt_tokens=b.prompt_tokens,
                completion_tokens=b.completion_tokens,
                request_count=b.request_count,
            )
            for b in response.buckets
        ]
        return CostReport(
            start_time=_ts_to_dt(response.start_time) or start_time,
            end_time=_ts_to_dt(response.end_time) or end_time,
            group_by=list(response.group_by),
            buckets=buckets,
            total_cost_usd=response.total_cost_usd,
        )

    def set_budget_alert(
        self,
        *,
        name: str,
        scope_type: str,
        scope_value: str,
        limit_usd: float,
        period: str = "monthly",
        thresholds: Optional[List[float]] = None,
        notification_channels: Optional[List[str]] = None,
    ) -> str:
        proto_alert = observability_pb2.BudgetAlert(
            name=name,
            scope_type=scope_type,
            scope_value=scope_value,
            limit_usd=limit_usd,
            period=period,
            thresholds=thresholds or [0.7, 0.9, 1.0],
            notification_channels=notification_channels or [],
        )
        response = self._stub.SetBudgetAlert(
            observability_pb2.SetBudgetAlertRequest(alert=proto_alert),
            metadata=self.metadata,
        )
        return response.name

    def get_budget_status(self, name: str) -> BudgetStatus:
        response = self._stub.GetBudgetStatus(
            observability_pb2.GetBudgetStatusRequest(name=name), metadata=self.metadata
        )
        alert = BudgetAlert(
            name=response.alert.name,
            scope_type=response.alert.scope_type,
            scope_value=response.alert.scope_value,
            limit_usd=response.alert.limit_usd,
            period=response.alert.period,
            thresholds=list(response.alert.thresholds),
            notification_channels=list(response.alert.notification_channels),
        )
        return BudgetStatus(
            alert=alert,
            current_spend_usd=response.current_spend_usd,
            projected_spend_usd=response.projected_spend_usd,
            percent_used=response.percent_used,
            thresholds_crossed=list(response.thresholds_crossed),
        )

    def get_service_health(self, service: str, lookback_seconds: int = 600) -> ServiceHealth:
        response = self._stub.GetServiceHealth(
            observability_pb2.ServiceHealthRequest(
                service=service, lookback_seconds=lookback_seconds
            ),
            metadata=self.metadata,
        )
        return ServiceHealth(
            service=response.service,
            status=response.status,
            last_span_at=_ts_to_dt(response.last_span_at),
            span_count=response.span_count,
            error_rate=response.error_rate,
            detail=response.detail,
        )

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------
    def _domain_span_to_proto(self, span: Span):
        proto = observability_pb2.Span(
            trace_id=span.trace_id,
            span_id=span.span_id or uuid.uuid4().hex,
            parent_span_id=span.parent_span_id,
            service=span.service,
            operation=span.operation,
            error_message=span.error_message,
            attributes=span.attributes,
            numeric_attributes=span.numeric_attributes,
        )
        proto.status = observability_pb2.OK if span.status == "OK" else observability_pb2.SPAN_ERROR
        if span.start_time:
            proto.start_time.CopyFrom(_dt_to_ts(span.start_time))
        if span.end_time:
            proto.end_time.CopyFrom(_dt_to_ts(span.end_time))
        return proto

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

    def _proto_trace_to_domain(self, proto_trace) -> Trace:
        return Trace(
            trace_id=proto_trace.trace_id,
            session_id=proto_trace.session_id,
            workflow_id=proto_trace.workflow_id,
            user_id=proto_trace.user_id,
            spans=[self._proto_span_to_domain(s) for s in proto_trace.spans],
            generations=[self._proto_generation_to_domain(g) for g in proto_trace.generations],
            input=proto_trace.input,
            output=proto_trace.output,
            total_duration_ms=proto_trace.total_duration_ms,
            total_cost_usd=proto_trace.total_cost_usd,
            total_tokens=proto_trace.total_tokens,
            scores=[self._proto_score_to_domain(s) for s in proto_trace.scores],
            tags=list(proto_trace.tags),
        )
