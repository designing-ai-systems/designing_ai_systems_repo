"""
Observability storage abstraction.

This service stores five primitives in five separate in-memory tables
(spans, generations, logs, metrics, scores) plus a sixth table for
budget alerts. The book describes a production deployment that backs
each table with a different storage engine; this implementation keeps
everything in process so the chapter-7 listings work without external
infrastructure.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 7.1: ObservabilityService gRPC contract
  - Listing 7.5: Span / Generation / Trace
"""

from __future__ import annotations

import math
import threading
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from services.observability.models import (
    BudgetAlert,
    BudgetStatus,
    CostBucket,
    CostReport,
    Generation,
    LogEvent,
    MetricRecord,
    Score,
    ServiceHealth,
    Span,
    Trace,
    assemble_trace,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------


class ObservabilityStore(ABC):
    """Abstract storage interface for observability data."""

    # spans / generations
    @abstractmethod
    def record_spans(self, spans: List[Span]) -> int: ...

    @abstractmethod
    def record_generations(self, generations: List[Generation]) -> int: ...

    @abstractmethod
    def get_trace(self, trace_id: str) -> Optional[Trace]: ...

    @abstractmethod
    def query_traces(
        self,
        *,
        workflow_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        min_duration_ms: Optional[float] = None,
        max_duration_ms: Optional[float] = None,
        min_score: Optional[Dict[str, float]] = None,
        max_score: Optional[Dict[str, float]] = None,
        limit: int = 100,
    ) -> List[Trace]: ...

    # logs
    @abstractmethod
    def ingest_logs(self, events: List[LogEvent]) -> int: ...

    @abstractmethod
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
    ) -> Tuple[List[LogEvent], str, int]: ...

    # metrics
    @abstractmethod
    def record_metrics(self, records: List[MetricRecord]) -> int: ...

    @abstractmethod
    def query_metrics(
        self,
        *,
        name: str,
        label_filters: Optional[Dict[str, str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        aggregation: str = "sum",
    ) -> Dict[str, Any]: ...

    # scores
    @abstractmethod
    def record_score(self, score: Score) -> str: ...

    @abstractmethod
    def query_scores(
        self,
        *,
        trace_id: str = "",
        name: str = "",
        source: str = "",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Score]: ...

    # cost & budgets
    @abstractmethod
    def get_cost_report(
        self,
        *,
        start_time: datetime,
        end_time: datetime,
        group_by: List[str],
        filters: Optional[Dict[str, str]] = None,
    ) -> CostReport: ...

    @abstractmethod
    def set_budget_alert(self, alert: BudgetAlert) -> str: ...

    @abstractmethod
    def get_budget_status(self, name: str) -> Optional[BudgetStatus]: ...

    # health
    @abstractmethod
    def get_service_health(self, service: str, lookback_seconds: int = 600) -> ServiceHealth: ...


# ---------------------------------------------------------------------------
# In-memory implementation
# ---------------------------------------------------------------------------


_SEVERITY_ORDER = {
    "DEBUG": 1,
    "INFO": 2,
    "WARNING": 3,
    "ERROR": 4,
    "CRITICAL": 5,
}


def _percentile(values: List[float], p: float) -> float:
    """Return the p-th percentile from a sorted (ascending) list of values."""
    if not values:
        return 0.0
    if p <= 0:
        return values[0]
    if p >= 100:
        return values[-1]
    k = (len(values) - 1) * (p / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return values[int(k)]
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


class InMemoryObservabilityStore(ObservabilityStore):
    """In-memory observability store. Thread-safe, lossy on restart."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._spans: List[Span] = []
        self._generations: List[Generation] = []
        self._logs: List[LogEvent] = []
        self._metrics: List[MetricRecord] = []
        self._scores: List[Score] = []
        self._budgets: Dict[str, BudgetAlert] = {}

    # ------------------------------------------------------------------
    # Spans / generations / traces
    # ------------------------------------------------------------------
    def record_spans(self, spans: List[Span]) -> int:
        with self._lock:
            self._spans.extend(spans)
        return len(spans)

    def record_generations(self, generations: List[Generation]) -> int:
        with self._lock:
            self._generations.extend(generations)
        return len(generations)

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        with self._lock:
            spans = [s for s in self._spans if s.trace_id == trace_id]
            gens = [g for g in self._generations if g.span.trace_id == trace_id]
            scores = [s for s in self._scores if s.trace_id == trace_id]
        if not spans and not gens and not scores:
            return None
        return assemble_trace(trace_id, spans, gens, scores)

    def query_traces(
        self,
        *,
        workflow_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        min_duration_ms: Optional[float] = None,
        max_duration_ms: Optional[float] = None,
        min_score: Optional[Dict[str, float]] = None,
        max_score: Optional[Dict[str, float]] = None,
        limit: int = 100,
    ) -> List[Trace]:
        with self._lock:
            trace_ids = sorted(
                {s.trace_id for s in self._spans} | {g.span.trace_id for g in self._generations}
            )
        traces: List[Trace] = []
        for tid in trace_ids:
            trace = self.get_trace(tid)
            if trace is None:
                continue

            if workflow_id and trace.workflow_id != workflow_id:
                # also accept traces whose spans carry workflow_id as attribute
                if not any(s.attributes.get("workflow_id") == workflow_id for s in trace.spans):
                    continue
            if user_id and trace.user_id != user_id:
                if not any(s.attributes.get("user_id") == user_id for s in trace.spans):
                    continue
            if session_id and trace.session_id != session_id:
                if not any(s.attributes.get("session_id") == session_id for s in trace.spans):
                    continue
            if tags:
                trace_tag_set = set(trace.tags)
                for s in trace.spans:
                    raw = s.attributes.get("tags", "")
                    if raw:
                        trace_tag_set.update(t.strip() for t in raw.split(",") if t.strip())
                if not all(t in trace_tag_set for t in tags):
                    continue

            span_starts = [s.start_time for s in trace.spans if s.start_time]
            span_starts.extend(g.span.start_time for g in trace.generations if g.span.start_time)
            span_ends = [s.end_time for s in trace.spans if s.end_time]
            span_ends.extend(g.span.end_time for g in trace.generations if g.span.end_time)
            if start_time and span_starts and min(span_starts) < start_time:
                continue
            if end_time and span_ends and max(span_ends) > end_time:
                continue

            if min_duration_ms is not None and trace.total_duration_ms < min_duration_ms:
                continue
            if max_duration_ms is not None and trace.total_duration_ms > max_duration_ms:
                continue

            if min_score:
                fail = False
                for name, threshold in min_score.items():
                    matching = [
                        s
                        for s in trace.scores
                        if s.name == name and isinstance(s.value, (int, float))
                    ]
                    if not matching:
                        fail = True
                        break
                    if max(float(s.value) for s in matching) < threshold:
                        fail = True
                        break
                if fail:
                    continue
            if max_score:
                fail = False
                for name, threshold in max_score.items():
                    matching = [
                        s
                        for s in trace.scores
                        if s.name == name and isinstance(s.value, (int, float))
                    ]
                    if not matching:
                        continue
                    if min(float(s.value) for s in matching) > threshold:
                        fail = True
                        break
                if fail:
                    continue

            traces.append(trace)
            if len(traces) >= limit:
                break
        return traces

    # ------------------------------------------------------------------
    # Logs
    # ------------------------------------------------------------------
    def ingest_logs(self, events: List[LogEvent]) -> int:
        for evt in events:
            if not evt.event_id:
                evt.event_id = uuid.uuid4().hex
        with self._lock:
            self._logs.extend(events)
        return len(events)

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
    ) -> Tuple[List[LogEvent], str, int]:
        with self._lock:
            events = list(self._logs)

        threshold = _SEVERITY_ORDER.get(min_severity.upper(), 0) if min_severity else 0
        attribute_filters = attribute_filters or {}

        def keep(e: LogEvent) -> bool:
            if trace_id and e.trace_id != trace_id:
                return False
            if span_id and e.span_id != span_id:
                return False
            if service and e.service != service:
                return False
            if event_type and e.event_type != event_type:
                return False
            if threshold and _SEVERITY_ORDER.get(e.severity.upper(), 0) < threshold:
                return False
            if start_time and e.timestamp < start_time:
                return False
            if end_time and e.timestamp > end_time:
                return False
            for k, v in attribute_filters.items():
                if e.attributes.get(k) != v:
                    return False
            return True

        filtered = [e for e in events if keep(e)]
        offset = 0
        if cursor:
            try:
                offset = int(cursor)
            except ValueError:
                offset = 0
        page = filtered[offset : offset + max(0, limit)]
        next_cursor = str(offset + len(page)) if (offset + len(page)) < len(filtered) else ""
        return page, next_cursor, len(filtered)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def record_metrics(self, records: List[MetricRecord]) -> int:
        with self._lock:
            self._metrics.extend(records)
        return len(records)

    def query_metrics(
        self,
        *,
        name: str,
        label_filters: Optional[Dict[str, str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        aggregation: str = "sum",
    ) -> Dict[str, Any]:
        label_filters = label_filters or {}
        with self._lock:
            records = [m for m in self._metrics if m.name == name]
        records = [
            m
            for m in records
            if all(m.labels.get(k) == v for k, v in label_filters.items())
            and (start_time is None or m.timestamp >= start_time)
            and (end_time is None or m.timestamp <= end_time)
        ]
        values = sorted(m.value for m in records)
        agg = (aggregation or "sum").lower()
        result_value = 0.0
        if values:
            if agg == "sum":
                result_value = sum(values)
            elif agg == "count":
                result_value = float(len(values))
            elif agg == "avg":
                result_value = sum(values) / len(values)
            elif agg == "p50":
                result_value = _percentile(values, 50.0)
            elif agg == "p95":
                result_value = _percentile(values, 95.0)
            elif agg == "p99":
                result_value = _percentile(values, 99.0)
        percentiles: Dict[str, float] = {}
        if values:
            percentiles = {
                "p50": _percentile(values, 50.0),
                "p95": _percentile(values, 95.0),
                "p99": _percentile(values, 99.0),
            }
        return {
            "name": name,
            "aggregation": agg,
            "value": result_value,
            "sample_count": len(values),
            "percentiles": percentiles,
        }

    # ------------------------------------------------------------------
    # Scores
    # ------------------------------------------------------------------
    def record_score(self, score: Score) -> str:
        if not score.score_id:
            score.score_id = uuid.uuid4().hex
        with self._lock:
            self._scores.append(score)
        return score.score_id

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
        with self._lock:
            scores = list(self._scores)
        if trace_id:
            scores = [s for s in scores if s.trace_id == trace_id]
        if name:
            scores = [s for s in scores if s.name == name]
        if source:
            scores = [s for s in scores if s.source == source]
        if start_time:
            scores = [s for s in scores if s.timestamp >= start_time]
        if end_time:
            scores = [s for s in scores if s.timestamp <= end_time]
        return scores[:limit]

    # ------------------------------------------------------------------
    # Cost & budgets
    # ------------------------------------------------------------------
    def get_cost_report(
        self,
        *,
        start_time: datetime,
        end_time: datetime,
        group_by: List[str],
        filters: Optional[Dict[str, str]] = None,
    ) -> CostReport:
        filters = filters or {}
        with self._lock:
            gens = list(self._generations)

        in_range = [
            g
            for g in gens
            if (g.span.start_time is None or start_time <= g.span.start_time <= end_time)
        ]

        def matches_filters(g: Generation) -> bool:
            for k, v in filters.items():
                if k == "model" and g.model != v:
                    return False
                if k == "provider" and g.provider != v:
                    return False
                if g.span.attributes.get(k) != v and k not in ("model", "provider"):
                    return False
            return True

        in_range = [g for g in in_range if matches_filters(g)]

        buckets: Dict[Tuple[str, ...], CostBucket] = {}
        for g in in_range:
            key_parts: List[str] = []
            dimensions: Dict[str, str] = {}
            for dim in group_by:
                if dim == "model":
                    val = g.model
                elif dim == "provider":
                    val = g.provider
                else:
                    val = g.span.attributes.get(dim, "")
                dimensions[dim] = val
                key_parts.append(val)
            key = tuple(key_parts)
            bucket = buckets.setdefault(key, CostBucket(dimensions=dict(dimensions)))
            bucket.cost_usd += g.cost_usd
            bucket.prompt_tokens += g.prompt_tokens
            bucket.completion_tokens += g.completion_tokens
            bucket.request_count += 1

        report = CostReport(
            start_time=start_time,
            end_time=end_time,
            group_by=list(group_by),
            buckets=sorted(buckets.values(), key=lambda b: -b.cost_usd),
            total_cost_usd=sum(b.cost_usd for b in buckets.values()),
        )
        return report

    def set_budget_alert(self, alert: BudgetAlert) -> str:
        with self._lock:
            self._budgets[alert.name] = alert
        return alert.name

    def get_budget_status(self, name: str) -> Optional[BudgetStatus]:
        with self._lock:
            alert = self._budgets.get(name)
        if alert is None:
            return None
        period_start, period_end = _budget_period_window(alert.period)

        filters: Dict[str, str] = {}
        if alert.scope_type and alert.scope_value:
            if alert.scope_type == "team":
                filters["team"] = alert.scope_value
            elif alert.scope_type == "workflow":
                filters["workflow_id"] = alert.scope_value
            elif alert.scope_type == "application":
                filters["application_id"] = alert.scope_value

        report = self.get_cost_report(
            start_time=period_start,
            end_time=period_end,
            group_by=[],
            filters=filters,
        )
        current = report.total_cost_usd
        elapsed = max(1e-9, (datetime.now(timezone.utc) - period_start).total_seconds())
        total = max(elapsed, (period_end - period_start).total_seconds())
        projected = current * (total / elapsed) if elapsed < total else current
        percent_used = (current / alert.limit_usd) if alert.limit_usd > 0 else 0.0
        crossed = [t for t in alert.thresholds if percent_used >= t]
        return BudgetStatus(
            alert=alert,
            current_spend_usd=current,
            projected_spend_usd=projected,
            percent_used=percent_used,
            thresholds_crossed=crossed,
        )

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------
    def get_service_health(self, service: str, lookback_seconds: int = 600) -> ServiceHealth:
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=max(1, lookback_seconds))
        with self._lock:
            spans = [s for s in self._spans if s.service == service]
            gen_spans = [g.span for g in self._generations if g.span.service == service]
        all_spans = spans + gen_spans
        recent = [s for s in all_spans if s.start_time and s.start_time >= cutoff]
        last_span_at = max((s.start_time for s in all_spans), default=None)
        if not recent:
            return ServiceHealth(
                service=service,
                status="unknown",
                last_span_at=last_span_at,
                span_count=0,
                error_rate=0.0,
                detail="no spans in lookback window",
            )
        errors = sum(1 for s in recent if s.status == "ERROR")
        rate = errors / max(1, len(recent))
        status = "healthy"
        if rate >= 0.5:
            status = "degraded"
        elif rate >= 0.1:
            status = "degraded"
        return ServiceHealth(
            service=service,
            status=status,
            last_span_at=last_span_at,
            span_count=len(recent),
            error_rate=rate,
            detail=f"{errors} error span(s) of {len(recent)} in {lookback_seconds}s window",
        )


def _budget_period_window(period: str) -> Tuple[datetime, datetime]:
    """Return the (start, end) of the current budget period."""
    now = datetime.now(timezone.utc)
    period = (period or "monthly").lower()
    if period == "daily":
        start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
        end = start + timedelta(days=1)
    elif period == "weekly":
        start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc) - timedelta(
            days=now.weekday()
        )
        end = start + timedelta(days=7)
    else:  # monthly
        start = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
        if now.month == 12:
            end = datetime(now.year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            end = datetime(now.year, now.month + 1, 1, tzinfo=timezone.utc)
    return start, end
