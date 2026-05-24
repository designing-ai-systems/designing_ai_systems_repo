"""
PostgreSQL-backed ObservabilityStore.

Mirrors the in-memory store's interface so the servicer doesn't care
which backend is configured. Each table maps onto one observability
primitive (spans, generations, logs, metrics, scores, budgets).

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 7.1: ObservabilityService gRPC contract (read-side queries)
  - Listing 7.5: Span / Generation / Trace persistence
"""

from __future__ import annotations

import json
import math
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
import psycopg2.extras

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
    SpanEvent,
    Trace,
    assemble_trace,
)
from services.observability.store import ObservabilityStore, _budget_period_window

_SEVERITY_ORDER = {
    "DEBUG": 1,
    "INFO": 2,
    "WARNING": 3,
    "ERROR": 4,
    "CRITICAL": 5,
}


def _percentile(values: List[float], p: float) -> float:
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


class PostgresObservabilityStore(ObservabilityStore):
    """PostgreSQL implementation of ObservabilityStore.

    Same interface as ``InMemoryObservabilityStore``; selection is
    driven by the ``OBSERVABILITY_POSTGRES_DSN`` env var in
    ``services.observability.main``. Tests skip when no DSN is set so
    the unit-test suite stays Postgres-free.
    """

    def __init__(self, connection_string: Optional[str] = None) -> None:
        if not connection_string:
            connection_string = os.getenv(
                "OBSERVABILITY_POSTGRES_DSN",
                "postgresql://localhost/genai_platform",
            )
        self.conn = psycopg2.connect(
            connection_string,
            cursor_factory=psycopg2.extras.RealDictCursor,
        )
        # Autocommit so every statement is its own transaction. Without
        # this, read-only queries (query_traces, get_trace, query_logs,
        # query_metrics, …) leave the connection "idle in transaction"
        # holding row locks. Subsequent attempts to TRUNCATE or run any
        # other AccessExclusiveLock-requiring statement then block for
        # the lifetime of the next write — which can be forever in a
        # quiet system. Explicit ``self.conn.commit()`` calls below are
        # now no-ops but harmless.
        self.conn.autocommit = True
        self._create_tables()

    def _create_tables(self) -> None:
        schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
        with open(schema_path) as f:
            sql = f.read()
        with self.conn.cursor() as cur:
            cur.execute(sql)
        self.conn.commit()

    # ------------------------------------------------------------------
    # Spans / generations / traces
    # ------------------------------------------------------------------
    def record_spans(self, spans: List[Span]) -> int:
        if not spans:
            return 0
        with self.conn.cursor() as cur:
            for span in spans:
                cur.execute(
                    """INSERT INTO spans (
                           span_id, trace_id, parent_span_id, service, operation,
                           start_time, end_time, status, error_message,
                           attributes, numeric_attributes, events)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb)
                       ON CONFLICT (span_id) DO UPDATE SET
                           end_time = EXCLUDED.end_time,
                           status = EXCLUDED.status,
                           error_message = EXCLUDED.error_message,
                           attributes = EXCLUDED.attributes,
                           numeric_attributes = EXCLUDED.numeric_attributes""",
                    (
                        span.span_id,
                        span.trace_id,
                        span.parent_span_id,
                        span.service,
                        span.operation,
                        span.start_time,
                        span.end_time,
                        span.status,
                        span.error_message,
                        json.dumps(span.attributes),
                        json.dumps(span.numeric_attributes),
                        json.dumps(
                            [
                                {
                                    "name": e.name,
                                    "timestamp": e.timestamp.isoformat(),
                                    "attributes": e.attributes,
                                }
                                for e in span.events
                            ]
                        ),
                    ),
                )
            self.conn.commit()
        return len(spans)

    def record_generations(self, generations: List[Generation]) -> int:
        if not generations:
            return 0
        with self.conn.cursor() as cur:
            for gen in generations:
                span = gen.span
                cur.execute(
                    """INSERT INTO generations (
                           span_id, trace_id, parent_span_id, service, operation,
                           start_time, end_time, status, error_message,
                           attributes, numeric_attributes,
                           model, provider, requested_model, prompt_tokens,
                           completion_tokens, cost_usd, cache_hit, fallback_used,
                           time_to_first_token_ms)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb,
                               %s, %s, %s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (span_id) DO UPDATE SET
                           end_time = EXCLUDED.end_time,
                           status = EXCLUDED.status,
                           prompt_tokens = EXCLUDED.prompt_tokens,
                           completion_tokens = EXCLUDED.completion_tokens,
                           cost_usd = EXCLUDED.cost_usd,
                           cache_hit = EXCLUDED.cache_hit,
                           fallback_used = EXCLUDED.fallback_used""",
                    (
                        span.span_id,
                        span.trace_id,
                        span.parent_span_id,
                        span.service,
                        span.operation,
                        span.start_time,
                        span.end_time,
                        span.status,
                        span.error_message,
                        json.dumps(span.attributes),
                        json.dumps(span.numeric_attributes),
                        gen.model,
                        gen.provider,
                        gen.requested_model,
                        gen.prompt_tokens,
                        gen.completion_tokens,
                        gen.cost_usd,
                        gen.cache_hit,
                        gen.fallback_used,
                        gen.time_to_first_token_ms,
                    ),
                )
            self.conn.commit()
        return len(generations)

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM spans WHERE trace_id = %s", (trace_id,))
            spans = [self._row_to_span(r) for r in cur.fetchall()]
            cur.execute("SELECT * FROM generations WHERE trace_id = %s", (trace_id,))
            gens = [self._row_to_generation(r) for r in cur.fetchall()]
            cur.execute("SELECT * FROM scores WHERE trace_id = %s", (trace_id,))
            scores = [self._row_to_score(r) for r in cur.fetchall()]
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
        # Build a single SQL query that finds candidate trace_ids whose
        # span window overlaps the requested time window, then assemble
        # each one. This keeps the SQL simple at the cost of N+1 reads.
        sql = """
            SELECT trace_id, MIN(start_time) AS s, MAX(end_time) AS e
              FROM (
                SELECT trace_id, start_time, end_time FROM spans
                UNION ALL
                SELECT trace_id, start_time, end_time FROM generations
              ) all_spans
             GROUP BY trace_id
        """
        with self.conn.cursor() as cur:
            cur.execute(sql)
            candidates = cur.fetchall()

        traces: List[Trace] = []
        for row in candidates:
            tid = row["trace_id"]
            t_start = row["s"]
            t_end = row["e"] or t_start
            # Overlap semantics: drop only if the trace is entirely outside
            # the window. Matches the in-memory store's filter.
            if start_time and t_end is not None and t_end < start_time:
                continue
            if end_time and t_start is not None and t_start > end_time:
                continue

            trace = self.get_trace(tid)
            if trace is None:
                continue

            if workflow_id:
                if trace.workflow_id != workflow_id and not any(
                    s.attributes.get("workflow_id") == workflow_id for s in trace.spans
                ):
                    continue
            if user_id:
                if trace.user_id != user_id and not any(
                    s.attributes.get("user_id") == user_id for s in trace.spans
                ):
                    continue
            if session_id:
                if trace.session_id != session_id and not any(
                    s.attributes.get("session_id") == session_id for s in trace.spans
                ):
                    continue
            if tags:
                trace_tag_set = set(trace.tags)
                for s in trace.spans:
                    raw = s.attributes.get("tags", "")
                    if raw:
                        trace_tag_set.update(t.strip() for t in raw.split(",") if t.strip())
                if not all(t in trace_tag_set for t in tags):
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
                    if not matching or max(float(s.value) for s in matching) < threshold:
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
        if not events:
            return 0
        with self.conn.cursor() as cur:
            for evt in events:
                # Same defensive UUID as record_score — an empty event_id
                # would collide via ON CONFLICT and silently drop the event.
                if not evt.event_id:
                    evt.event_id = uuid.uuid4().hex
                cur.execute(
                    """INSERT INTO logs (
                           event_id, trace_id, span_id, timestamp, service,
                           severity, event_type, message,
                           attributes, numeric_attributes,
                           workflow_id, user_id)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s,
                               %s::jsonb, %s::jsonb, %s, %s)
                       ON CONFLICT (event_id) DO NOTHING""",
                    (
                        evt.event_id,
                        evt.trace_id,
                        evt.span_id,
                        evt.timestamp,
                        evt.service,
                        evt.severity,
                        evt.event_type,
                        evt.message,
                        json.dumps(evt.attributes),
                        json.dumps(evt.numeric_attributes),
                        evt.workflow_id,
                        evt.user_id,
                    ),
                )
            self.conn.commit()
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
        conditions = ["TRUE"]
        params: list = []
        if trace_id:
            conditions.append("trace_id = %s")
            params.append(trace_id)
        if span_id:
            conditions.append("span_id = %s")
            params.append(span_id)
        if service:
            conditions.append("service = %s")
            params.append(service)
        if event_type:
            conditions.append("event_type = %s")
            params.append(event_type)
        if start_time:
            conditions.append("timestamp >= %s")
            params.append(start_time)
        if end_time:
            conditions.append("timestamp <= %s")
            params.append(end_time)
        threshold = _SEVERITY_ORDER.get(min_severity.upper(), 0) if min_severity else 0

        where = " AND ".join(conditions)
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) AS c FROM logs WHERE {where}", params)
            total = int(cur.fetchone()["c"])

            offset = 0
            if cursor:
                try:
                    offset = int(cursor)
                except ValueError:
                    offset = 0
            cur.execute(
                f"SELECT * FROM logs WHERE {where} ORDER BY timestamp ASC, event_id ASC "
                f"OFFSET %s LIMIT %s",
                params + [offset, max(0, limit)],
            )
            rows = cur.fetchall()

        events: List[LogEvent] = []
        for row in rows:
            evt = self._row_to_log(row)
            if threshold and _SEVERITY_ORDER.get(evt.severity.upper(), 0) < threshold:
                continue
            if attribute_filters and any(
                evt.attributes.get(k) != v for k, v in attribute_filters.items()
            ):
                continue
            events.append(evt)
        next_cursor = str(offset + len(rows)) if (offset + len(rows)) < total else ""
        return events, next_cursor, total

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def record_metrics(self, records: List[MetricRecord]) -> int:
        if not records:
            return 0
        with self.conn.cursor() as cur:
            for rec in records:
                cur.execute(
                    """INSERT INTO metrics (name, type, value, labels, timestamp)
                       VALUES (%s, %s, %s, %s::jsonb, %s)""",
                    (rec.name, rec.type, rec.value, json.dumps(rec.labels), rec.timestamp),
                )
            self.conn.commit()
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
        conditions = ["name = %s"]
        params: list = [name]
        if start_time:
            conditions.append("timestamp >= %s")
            params.append(start_time)
        if end_time:
            conditions.append("timestamp <= %s")
            params.append(end_time)
        for k, v in (label_filters or {}).items():
            conditions.append("labels ->> %s = %s")
            params.extend([k, v])
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT value FROM metrics WHERE {' AND '.join(conditions)} ORDER BY value ASC",
                params,
            )
            values = [row["value"] for row in cur.fetchall()]
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
        percentiles = (
            {
                "p50": _percentile(values, 50.0),
                "p95": _percentile(values, 95.0),
                "p99": _percentile(values, 99.0),
            }
            if values
            else {}
        )
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
        # Without a fresh UUID, every score the SDK records (which never
        # sets score_id) would collide on the empty-string PK via
        # ON CONFLICT and overwrite the previous row.
        if not score.score_id:
            score.score_id = uuid.uuid4().hex
        kind = "numeric"
        numeric_value: Optional[float] = None
        string_value: Optional[str] = None
        boolean_value: Optional[bool] = None
        if isinstance(score.value, bool):
            kind = "boolean"
            boolean_value = score.value
        elif isinstance(score.value, (int, float)):
            kind = "numeric"
            numeric_value = float(score.value)
        elif isinstance(score.value, str):
            kind = "categorical"
            string_value = score.value
        with self.conn.cursor() as cur:
            cur.execute(
                """INSERT INTO scores (
                       score_id, trace_id, span_id, generation_id, name,
                       value_kind, numeric_value, string_value, boolean_value,
                       source, comment, metadata, timestamp)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                   ON CONFLICT (score_id) DO UPDATE SET
                       numeric_value = EXCLUDED.numeric_value,
                       string_value  = EXCLUDED.string_value,
                       boolean_value = EXCLUDED.boolean_value,
                       metadata      = EXCLUDED.metadata""",
                (
                    score.score_id,
                    score.trace_id,
                    score.span_id,
                    score.generation_id,
                    score.name,
                    kind,
                    numeric_value,
                    string_value,
                    boolean_value,
                    score.source,
                    score.comment,
                    json.dumps(score.metadata),
                    score.timestamp,
                ),
            )
            self.conn.commit()
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
        conditions = ["TRUE"]
        params: list = []
        if trace_id:
            conditions.append("trace_id = %s")
            params.append(trace_id)
        if name:
            conditions.append("name = %s")
            params.append(name)
        if source:
            conditions.append("source = %s")
            params.append(source)
        if start_time:
            conditions.append("timestamp >= %s")
            params.append(start_time)
        if end_time:
            conditions.append("timestamp <= %s")
            params.append(end_time)
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT * FROM scores WHERE {' AND '.join(conditions)} "
                f"ORDER BY timestamp DESC LIMIT %s",
                params + [limit],
            )
            return [self._row_to_score(row) for row in cur.fetchall()]

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
        conditions = ["start_time >= %s", "start_time <= %s"]
        params: list = [start_time, end_time]
        for key, value in (filters or {}).items():
            if key == "model":
                conditions.append("model = %s")
                params.append(value)
            elif key == "provider":
                conditions.append("provider = %s")
                params.append(value)
            else:
                conditions.append("attributes ->> %s = %s")
                params.extend([key, value])
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT * FROM generations WHERE {' AND '.join(conditions)}",
                params,
            )
            rows = cur.fetchall()

        buckets: Dict[Tuple[str, ...], CostBucket] = {}
        for row in rows:
            key_parts: List[str] = []
            dimensions: Dict[str, str] = {}
            attrs = row["attributes"] or {}
            for dim in group_by:
                if dim == "model":
                    val = row["model"]
                elif dim == "provider":
                    val = row["provider"]
                else:
                    val = attrs.get(dim, "")
                dimensions[dim] = val
                key_parts.append(val)
            key = tuple(key_parts)
            bucket = buckets.setdefault(key, CostBucket(dimensions=dict(dimensions)))
            bucket.cost_usd += float(row["cost_usd"] or 0.0)
            bucket.prompt_tokens += int(row["prompt_tokens"] or 0)
            bucket.completion_tokens += int(row["completion_tokens"] or 0)
            bucket.request_count += 1

        return CostReport(
            start_time=start_time,
            end_time=end_time,
            group_by=list(group_by),
            buckets=sorted(buckets.values(), key=lambda b: -b.cost_usd),
            total_cost_usd=sum(b.cost_usd for b in buckets.values()),
        )

    def set_budget_alert(self, alert: BudgetAlert) -> str:
        with self.conn.cursor() as cur:
            cur.execute(
                """INSERT INTO budget_alerts (
                       name, scope_type, scope_value, limit_usd, period,
                       thresholds, notification_channels, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s)
                   ON CONFLICT (name) DO UPDATE SET
                       scope_type = EXCLUDED.scope_type,
                       scope_value = EXCLUDED.scope_value,
                       limit_usd = EXCLUDED.limit_usd,
                       period = EXCLUDED.period,
                       thresholds = EXCLUDED.thresholds,
                       notification_channels = EXCLUDED.notification_channels""",
                (
                    alert.name,
                    alert.scope_type,
                    alert.scope_value,
                    alert.limit_usd,
                    alert.period,
                    json.dumps(alert.thresholds),
                    json.dumps(alert.notification_channels),
                    alert.created_at,
                ),
            )
            self.conn.commit()
        return alert.name

    def get_budget_status(self, name: str) -> Optional[BudgetStatus]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM budget_alerts WHERE name = %s", (name,))
            row = cur.fetchone()
        if row is None:
            return None
        alert = BudgetAlert(
            name=row["name"],
            scope_type=row["scope_type"],
            scope_value=row["scope_value"],
            limit_usd=float(row["limit_usd"]),
            period=row["period"],
            thresholds=list(row["thresholds"] or []),
            notification_channels=list(row["notification_channels"] or []),
            created_at=row["created_at"],
        )
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
        with self.conn.cursor() as cur:
            cur.execute(
                """SELECT * FROM (
                       SELECT start_time, status FROM spans WHERE service = %s
                       UNION ALL
                       SELECT start_time, status FROM generations WHERE service = %s
                   ) s ORDER BY start_time DESC""",
                (service, service),
            )
            rows = cur.fetchall()
        if not rows:
            return ServiceHealth(
                service=service,
                status="unknown",
                last_span_at=None,
                span_count=0,
                error_rate=0.0,
                detail="no spans in lookback window",
            )
        last_span_at = rows[0]["start_time"]
        recent = [r for r in rows if r["start_time"] and r["start_time"] >= cutoff]
        if not recent:
            return ServiceHealth(
                service=service,
                status="unknown",
                last_span_at=last_span_at,
                span_count=0,
                error_rate=0.0,
                detail="no spans in lookback window",
            )
        errors = sum(1 for r in recent if r["status"] == "ERROR")
        rate = errors / max(1, len(recent))
        if rate >= 0.5:
            status = "critical"
        elif rate >= 0.1:
            status = "degraded"
        else:
            status = "healthy"
        return ServiceHealth(
            service=service,
            status=status,
            last_span_at=last_span_at,
            span_count=len(recent),
            error_rate=rate,
            detail=f"{errors} error span(s) of {len(recent)} in {lookback_seconds}s window",
        )

    # ------------------------------------------------------------------
    # Row → domain helpers
    # ------------------------------------------------------------------
    def _row_to_span(self, row) -> Span:
        events_raw = row.get("events") or []
        if isinstance(events_raw, str):
            events_raw = json.loads(events_raw)
        events = [
            SpanEvent(
                name=e.get("name", ""),
                timestamp=datetime.fromisoformat(e["timestamp"])
                if e.get("timestamp")
                else datetime.now(timezone.utc),
                attributes=e.get("attributes", {}),
            )
            for e in events_raw
        ]
        return Span(
            trace_id=row["trace_id"],
            span_id=row["span_id"],
            parent_span_id=row["parent_span_id"] or "",
            service=row["service"],
            operation=row["operation"],
            start_time=row["start_time"],
            end_time=row["end_time"],
            status=row["status"],
            error_message=row["error_message"] or "",
            attributes=dict(row["attributes"] or {}),
            numeric_attributes=dict(row["numeric_attributes"] or {}),
            events=events,
        )

    def _row_to_generation(self, row) -> Generation:
        span = Span(
            trace_id=row["trace_id"],
            span_id=row["span_id"],
            parent_span_id=row["parent_span_id"] or "",
            service=row["service"],
            operation=row["operation"],
            start_time=row["start_time"],
            end_time=row["end_time"],
            status=row["status"],
            error_message=row["error_message"] or "",
            attributes=dict(row["attributes"] or {}),
            numeric_attributes=dict(row["numeric_attributes"] or {}),
        )
        return Generation(
            span=span,
            model=row["model"],
            provider=row["provider"] or "",
            requested_model=row["requested_model"] or "",
            prompt_tokens=int(row["prompt_tokens"] or 0),
            completion_tokens=int(row["completion_tokens"] or 0),
            cost_usd=float(row["cost_usd"] or 0.0),
            cache_hit=bool(row["cache_hit"]),
            fallback_used=bool(row["fallback_used"]),
            time_to_first_token_ms=float(row["time_to_first_token_ms"] or 0.0),
        )

    def _row_to_log(self, row) -> LogEvent:
        return LogEvent(
            event_id=row["event_id"],
            trace_id=row["trace_id"] or "",
            span_id=row["span_id"] or "",
            timestamp=row["timestamp"],
            service=row["service"] or "",
            severity=row["severity"],
            event_type=row["event_type"] or "",
            message=row["message"] or "",
            attributes=dict(row["attributes"] or {}),
            numeric_attributes=dict(row["numeric_attributes"] or {}),
            workflow_id=row["workflow_id"] or "",
            user_id=row["user_id"] or "",
        )

    def _row_to_score(self, row) -> Score:
        kind = row["value_kind"]
        value: Any
        if kind == "boolean":
            value = bool(row["boolean_value"]) if row["boolean_value"] is not None else None
        elif kind == "categorical":
            value = row["string_value"]
        else:
            value = float(row["numeric_value"]) if row["numeric_value"] is not None else None
        return Score(
            score_id=row["score_id"],
            trace_id=row["trace_id"] or "",
            span_id=row["span_id"] or "",
            generation_id=row["generation_id"] or "",
            name=row["name"],
            value=value,
            source=row["source"],
            comment=row["comment"] or "",
            metadata=dict(row["metadata"] or {}),
            timestamp=row["timestamp"],
        )
