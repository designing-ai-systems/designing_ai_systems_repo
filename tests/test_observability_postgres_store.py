"""Integration smoke test for the Postgres-backed observability store.

Skipped automatically unless a Postgres DSN is reachable. Two
environment variables are honoured:

- ``OBSERVABILITY_POSTGRES_DSN`` — local development. Point at a
  compose Postgres exposed on a host port.
- ``DB_TEST_URL`` — the env var the GitHub Actions ``test`` job sets
  when it spins up a pgvector service container.

If neither is set or neither resolves to a reachable database, every
test in this file is skipped so the default local + CI flow stays
clean.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from services.observability.models import Generation, LogEvent, MetricRecord, Score, Span

psycopg2 = pytest.importorskip("psycopg2", reason="psycopg2 not installed")


def _dsn() -> str | None:
    """Resolve a Postgres DSN.

    Priority:
      1. ``OBSERVABILITY_POSTGRES_DSN`` — set this locally when pointing at
         a compose Postgres on a host port mapping.
      2. ``DB_TEST_URL`` — the env var the GitHub Actions CI workflow
         sets when it spins up the pgvector service container. Falling
         back to it keeps the same tests gating both environments.
    """
    return os.environ.get("OBSERVABILITY_POSTGRES_DSN") or os.environ.get("DB_TEST_URL")


def _can_connect(dsn: str) -> bool:
    try:
        conn = psycopg2.connect(dsn)
        conn.close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _dsn() or not _can_connect(_dsn() or ""),
    reason="OBSERVABILITY_POSTGRES_DSN / DB_TEST_URL unset or Postgres unreachable",
)


@pytest.fixture
def store():
    from services.observability.postgres_store import PostgresObservabilityStore

    s = PostgresObservabilityStore(connection_string=_dsn())
    # Clean slate per test run.
    with s.conn.cursor() as cur:
        cur.execute("TRUNCATE spans, generations, logs, metrics, scores RESTART IDENTITY")
        s.conn.commit()
    yield s
    s.conn.close()


def _make_span(**overrides) -> Span:
    base = dict(
        trace_id=f"t-{uuid.uuid4().hex[:8]}",
        span_id=uuid.uuid4().hex,
        service="models",
        operation="models.chat",
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc) + timedelta(milliseconds=20),
    )
    base.update(overrides)
    return Span(**base)


class TestPostgresObservabilityStore:
    def test_record_and_get_trace(self, store):
        trace_id = f"trace-{uuid.uuid4().hex[:6]}"
        span_a = _make_span(trace_id=trace_id)
        span_b = _make_span(trace_id=trace_id, parent_span_id=span_a.span_id)
        store.record_spans([span_a, span_b])

        trace = store.get_trace(trace_id)
        assert trace is not None
        assert len(trace.spans) == 2

    def test_record_generation_and_cost_report(self, store):
        trace_id = f"trace-{uuid.uuid4().hex[:6]}"
        gen = Generation(
            span=Span(
                trace_id=trace_id,
                span_id=uuid.uuid4().hex,
                service="models",
                operation="models.generation",
                start_time=datetime.now(timezone.utc),
                end_time=datetime.now(timezone.utc) + timedelta(milliseconds=100),
            ),
            model="gpt-4o",
            provider="openai",
            prompt_tokens=100,
            completion_tokens=50,
            cost_usd=0.0125,
        )
        store.record_generations([gen])

        report = store.get_cost_report(
            start_time=datetime.now(timezone.utc) - timedelta(minutes=1),
            end_time=datetime.now(timezone.utc) + timedelta(minutes=1),
            group_by=["model"],
        )
        assert report.total_cost_usd > 0
        assert any(bucket.dimensions.get("model") == "gpt-4o" for bucket in report.buckets)

    def test_logs_and_query(self, store):
        evt = LogEvent(
            event_id=uuid.uuid4().hex,
            trace_id="t-1",
            timestamp=datetime.now(timezone.utc),
            severity="WARNING",
            event_type="model_fallback",
            message="Fallback",
            service="models",
        )
        store.ingest_logs([evt])
        events, _cursor, total = store.query_logs(trace_id="t-1")
        assert total == 1
        assert events[0].event_id == evt.event_id

    def test_metrics_and_percentiles(self, store):
        now = datetime.now(timezone.utc)
        store.record_metrics(
            [
                MetricRecord(name="latency_ms", type="HISTOGRAM", value=float(v), timestamp=now)
                for v in range(1, 101)
            ]
        )
        result = store.query_metrics(name="latency_ms", aggregation="p95")
        assert result["sample_count"] == 100
        assert 94 <= result["value"] <= 96

    def test_score_round_trip(self, store):
        score = Score(
            score_id=uuid.uuid4().hex,
            trace_id="t-score",
            name="helpfulness",
            value=0.82,
            source="MODEL_JUDGE",
            timestamp=datetime.now(timezone.utc),
        )
        store.record_score(score)
        results = store.query_scores(trace_id="t-score")
        assert len(results) == 1
        assert results[0].value == pytest.approx(0.82)

    def test_reads_do_not_leave_connection_idle_in_transaction(self, store):
        """Regression: the store used to leave reads in an open
        transaction (psycopg2 starts an implicit one with autocommit off).
        That meant subsequent writes — including TRUNCATE — blocked on
        a row-level lock indefinitely. autocommit=True fixes it; this
        test gates the fix.

        Strategy: do a read through the store's connection, then open a
        *second* connection and try to acquire AccessExclusiveLock on
        ``spans`` with a short statement timeout. With the bug present
        the LOCK statement would block waiting on the read's idle
        transaction and trip the timeout. With the fix it returns
        immediately.
        """
        # Trigger a read that would (under the old code) leave an
        # idle-in-transaction connection holding locks.
        store.query_traces(limit=5)

        dsn = _dsn()
        assert dsn, "test should have been skipped if no DSN is reachable"
        probe = psycopg2.connect(dsn)
        probe.autocommit = True
        try:
            with probe.cursor() as cur:
                # 2-second statement timeout — well above the few-ms a
                # clean lock acquisition takes, well under "forever".
                cur.execute("SET LOCAL statement_timeout = '2s'")
                cur.execute("BEGIN")
                cur.execute("LOCK TABLE spans IN ACCESS EXCLUSIVE MODE")
                cur.execute("ROLLBACK")
        finally:
            probe.close()
