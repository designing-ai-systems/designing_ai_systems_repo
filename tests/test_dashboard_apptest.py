"""End-to-end render checks for the observability dashboard.

Uses streamlit.testing.v1.AppTest to actually execute the Streamlit
script in-process with a mocked GenAIPlatform. Catches rendering
errors that a unit test on the helpers misses — exactly the bugs that
slipped through earlier (st.switch_page, plotly None color list,
mixed-dtype scores column).

Tests are skipped automatically if streamlit isn't installed so the
default test environment (without the ``dashboards`` extra) stays
green.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import List
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("GENAI_GATEWAY_URL", "localhost:50051")

streamlit = pytest.importorskip("streamlit", reason="streamlit extra not installed")
pytest.importorskip("plotly", reason="plotly extra not installed")
pytest.importorskip("pandas", reason="pandas extra not installed")

from streamlit.testing.v1 import AppTest  # noqa: E402

DASHBOARD_APP = Path(__file__).parent.parent / "dashboards" / "observability" / "app.py"


# ---------------------------------------------------------------------------
# Fake trace data
# ---------------------------------------------------------------------------


def _span(
    *,
    span_id: str,
    parent_span_id: str = "",
    service: str = "models",
    operation: str = "models.chat",
    start: datetime,
    duration_ms: float = 50.0,
    status: str = "OK",
    attributes=None,
):
    return SimpleNamespace(
        trace_id="trace-1",
        span_id=span_id,
        parent_span_id=parent_span_id,
        service=service,
        operation=operation,
        start_time=start,
        end_time=start + timedelta(milliseconds=duration_ms),
        duration_ms=duration_ms,
        status=status,
        attributes=attributes or {},
        numeric_attributes={},
    )


def _generation(*, span_id, parent_span_id, start, duration_ms=1100.0):
    return SimpleNamespace(
        span=_span(
            span_id=span_id,
            parent_span_id=parent_span_id,
            service="models",
            operation="models.generation",
            start=start,
            duration_ms=duration_ms,
        ),
        model="gpt-4o",
        provider="openai",
        prompt_tokens=3100,
        completion_tokens=180,
        cost_usd=0.018,
        cache_hit=False,
        fallback_used=False,
        time_to_first_token_ms=340.0,
    )


def _score(*, name, value, source, comment="from-test"):
    return SimpleNamespace(
        score_id=f"score-{name}",
        trace_id="trace-1",
        span_id="",
        generation_id="",
        name=name,
        value=value,
        source=source,
        comment=comment,
        metadata={},
        timestamp=datetime(2026, 1, 1, 12, 30, tzinfo=timezone.utc),
    )


def _fixture_trace():
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    root = _span(
        span_id="root",
        service="gateway",
        operation="gateway.handle_request",
        start=base,
        duration_ms=1500.0,
        attributes={
            "input": "What documents do I need?",
            "output": "Please bring your insurance card and a photo ID.",
            "session_id": "session-fixture",
            "workflow_id": "patient-intake",
            "user_id": "patient-12345",
        },
    )
    spans = [
        root,
        _span(
            span_id="s-sessions",
            parent_span_id="root",
            service="sessions",
            operation="sessions.get_messages",
            start=base + timedelta(milliseconds=20),
            duration_ms=45.0,
        ),
        _span(
            span_id="s-data",
            parent_span_id="root",
            service="data",
            operation="data.search",
            start=base + timedelta(milliseconds=70),
            duration_ms=120.0,
        ),
        _span(
            span_id="s-guard-in",
            parent_span_id="root",
            service="guardrails",
            operation="guardrails.validate_input",
            start=base + timedelta(milliseconds=190),
            duration_ms=35.0,
        ),
        _span(
            span_id="s-guard-out",
            parent_span_id="root",
            service="guardrails",
            operation="guardrails.filter_output",
            start=base + timedelta(milliseconds=1325),
            duration_ms=30.0,
        ),
        _span(
            span_id="s-sessions-add",
            parent_span_id="root",
            service="sessions",
            operation="sessions.add_messages",
            start=base + timedelta(milliseconds=1355),
            duration_ms=80.0,
        ),
    ]
    generations = [
        _generation(
            span_id="g-gen",
            parent_span_id="root",
            start=base + timedelta(milliseconds=225),
            duration_ms=1100.0,
        )
    ]
    scores = [
        # Mixes float, str, and float on purpose — caught a real dataframe
        # dtype crash earlier when the value column ended up with object dtype.
        _score(name="helpfulness", value=0.85, source="MODEL_JUDGE"),
        _score(name="correctness", value="correct", source="HUMAN"),
        _score(name="retrieval_relevance", value=0.82, source="AUTOMATED"),
    ]
    return SimpleNamespace(
        trace_id="trace-1",
        session_id="session-fixture",
        workflow_id="patient-intake",
        user_id="patient-12345",
        spans=spans,
        generations=generations,
        scores=scores,
        input="What documents do I need?",
        output="Please bring your insurance card and a photo ID.",
        total_duration_ms=1500.0,
        total_cost_usd=0.018,
        total_tokens=3280,
        tags=[],
    )


def _make_fake_platform(traces: List = None):
    """Build a MagicMock platform whose observability surface returns the
    given traces (or one default fixture trace)."""
    traces = traces if traces is not None else [_fixture_trace()]
    platform = MagicMock()
    platform.observability.query_traces.return_value = traces
    platform.observability.get_trace.side_effect = lambda trace_id: next(
        (t for t in traces if t.trace_id == trace_id), None
    )
    # Cost / metrics / logs return empty so those pages don't crash either.
    platform.observability.get_cost_report.return_value = SimpleNamespace(
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc),
        group_by=["model"],
        buckets=[],
        total_cost_usd=0.0,
    )
    platform.observability.query_metrics.return_value = {
        "name": "metric",
        "aggregation": "p95",
        "value": 0.0,
        "sample_count": 0,
        "percentiles": {},
    }
    platform.observability.query_logs.return_value = {"events": [], "total_matched": 0}
    platform.observability.get_service_health.side_effect = lambda *, service, lookback_seconds: (
        SimpleNamespace(
            service=service,
            status="unknown",
            last_span_at=None,
            span_count=0,
            error_rate=0.0,
            detail="no data",
        )
    )
    return platform


def _prime_session_state(at: AppTest, platform) -> None:
    """Stash a pre-built fake platform under the same cache key the
    dashboard uses, so ``get_platform`` returns it instead of constructing
    a real ``GenAIPlatform`` and failing to connect."""
    gateway_url = os.environ.get("GENAI_GATEWAY_URL", "localhost:50051")
    at.session_state["_genai_platform_cache"] = {"url": gateway_url, "platform": platform}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSessionsPageRenders:
    def test_default_sessions_page_no_exceptions(self):
        at = AppTest.from_file(str(DASHBOARD_APP), default_timeout=30)
        _prime_session_state(at, _make_fake_platform())
        at.run()
        assert not at.exception, [str(e.value)[:200] for e in at.exception]

    def test_drill_into_session_renders_per_trace_expanders(self):
        """The fixture has one session with one trace. Drilling in should
        produce at least one ``Trace #N`` expander and zero exceptions —
        exactly the regression that broke when ``st.switch_page`` rejected
        title strings or plotly choked on a None colour."""
        at = AppTest.from_file(str(DASHBOARD_APP), default_timeout=30)
        _prime_session_state(at, _make_fake_platform())
        at.run()
        # Pick the only session in the dropdown.
        session_select = next(sb for sb in at.selectbox if "session" in (sb.label or "").lower())
        session_select.select("session-fixture").run()
        assert not at.exception, [str(e.value)[:200] for e in at.exception]
        # Must produce at least one Trace expander.
        trace_expander_labels = [e.label for e in at.expander if e.label.startswith("Trace")]
        assert trace_expander_labels, "expected at least one 'Trace #N' expander"
        # And a glossary expander up top.
        assert any("How to read" in e.label for e in at.expander)

    def test_mixed_dtype_scores_dont_crash_dataframe(self):
        """The fixture's scores column carries floats AND strings. If the
        dashboard ever stops coercing values to strings before rendering,
        st.dataframe trips a pyarrow conversion error and the page goes
        blank — caught only via AppTest, not via unit tests on helpers."""
        at = AppTest.from_file(str(DASHBOARD_APP), default_timeout=30)
        _prime_session_state(at, _make_fake_platform())
        at.run()
        at.selectbox[0].select("session-fixture").run()
        assert not at.exception, [str(e.value)[:200] for e in at.exception]


class TestPageGlossaryAlwaysRenders:
    """Every page has a 📖 expander at the top. Cheap smoke test that the
    expander gets created at the default page render."""

    def test_default_page_has_glossary_expander(self):
        at = AppTest.from_file(str(DASHBOARD_APP), default_timeout=30)
        _prime_session_state(at, _make_fake_platform())
        at.run()
        glossary = [e for e in at.expander if "How to read" in e.label]
        assert glossary, "default page missing the '📖 How to read this page' expander"
