"""Unit tests for the pure helpers in dashboards/observability/app.py.

The dashboard's plotly + Streamlit surface is awkward to test, but the
helper functions underneath it are pure data transforms. This module
covers them so a regression to span nesting, input/output extraction,
or the glossary content is caught before the dashboard renders.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import pytest

# Keep the import below this comment: streamlit checks for env at import.
os.environ.setdefault("GENAI_GATEWAY_URL", "localhost:50051")
sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboards.observability import app as dash  # noqa: E402

# ---------------------------------------------------------------------------
# Span / Generation factories
# ---------------------------------------------------------------------------


def _span(
    *,
    span_id: str,
    parent_span_id: str = "",
    service: str = "models",
    operation: str = "models.chat",
    start: Optional[datetime] = None,
    duration_ms: float = 50.0,
    status: str = "OK",
    attributes: Optional[dict] = None,
):
    start = start or datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(milliseconds=duration_ms)
    return SimpleNamespace(
        trace_id="t-fixture",
        span_id=span_id,
        parent_span_id=parent_span_id,
        service=service,
        operation=operation,
        start_time=start,
        end_time=end,
        duration_ms=duration_ms,
        status=status,
        attributes=attributes or {},
        numeric_attributes={},
    )


def _generation(*, span_id: str, parent_span_id: str = "", **span_kwargs):
    span = _span(span_id=span_id, parent_span_id=parent_span_id, **span_kwargs)
    return SimpleNamespace(
        span=span,
        model="gpt-4o",
        provider="openai",
        prompt_tokens=100,
        completion_tokens=50,
        cost_usd=0.01,
        cache_hit=False,
        fallback_used=False,
        time_to_first_token_ms=200.0,
    )


# ---------------------------------------------------------------------------
# _build_span_tree
# ---------------------------------------------------------------------------


class TestBuildSpanTree:
    def test_empty_input_returns_empty_list(self):
        assert dash._build_span_tree([], []) == []

    def test_single_root_span_has_depth_zero(self):
        ordered = dash._build_span_tree([_span(span_id="root")], [])
        assert len(ordered) == 1
        assert ordered[0]["depth"] == 0
        assert ordered[0]["kind"] == "span"

    def test_parent_child_nesting_assigns_increasing_depth(self):
        root = _span(span_id="root", service="gateway")
        child = _span(span_id="child", parent_span_id="root", service="data")
        grandchild = _span(span_id="gc", parent_span_id="child", service="models")
        ordered = dash._build_span_tree([root, child, grandchild], [])
        depths = [(item["span"].span_id, item["depth"]) for item in ordered]
        assert depths == [("root", 0), ("child", 1), ("gc", 2)]

    def test_siblings_sorted_by_start_time(self):
        base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        root = _span(span_id="root", start=base)
        # Insert children in reverse start-time order; expect ordered output.
        late = _span(
            span_id="late",
            parent_span_id="root",
            start=base + timedelta(milliseconds=200),
        )
        early = _span(
            span_id="early",
            parent_span_id="root",
            start=base + timedelta(milliseconds=50),
        )
        ordered = dash._build_span_tree([root, late, early], [])
        ids = [item["span"].span_id for item in ordered]
        assert ids == ["root", "early", "late"]

    def test_generation_nested_under_span(self):
        root = _span(span_id="root", service="gateway")
        gen = _generation(span_id="gen", parent_span_id="root", service="models")
        ordered = dash._build_span_tree([root], [gen])
        assert [(item["span"].span_id, item["kind"], item["depth"]) for item in ordered] == [
            ("root", "span", 0),
            ("gen", "generation", 1),
        ]

    def test_orphan_with_unknown_parent_treated_as_root(self):
        orphan = _span(span_id="orphan", parent_span_id="does-not-exist")
        ordered = dash._build_span_tree([orphan], [])
        assert len(ordered) == 1
        assert ordered[0]["depth"] == 0

    def test_full_figure_7_5_shape(self):
        """Smoke-test the exact shape the walkthrough produces: gateway
        root + five service spans as children + one generation as a
        child. Confirms each direct child gets depth 1."""
        base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        root = _span(span_id="root", service="gateway", start=base)
        children: List = [
            _span(
                span_id="s1",
                parent_span_id="root",
                service="sessions",
                operation="sessions.get_messages",
                start=base + timedelta(milliseconds=20),
            ),
            _span(
                span_id="s2",
                parent_span_id="root",
                service="data",
                operation="data.search",
                start=base + timedelta(milliseconds=70),
            ),
            _span(
                span_id="s3",
                parent_span_id="root",
                service="guardrails",
                operation="guardrails.validate_input",
                start=base + timedelta(milliseconds=190),
            ),
            _span(
                span_id="s4",
                parent_span_id="root",
                service="guardrails",
                operation="guardrails.filter_output",
                start=base + timedelta(milliseconds=1325),
            ),
            _span(
                span_id="s5",
                parent_span_id="root",
                service="sessions",
                operation="sessions.add_messages",
                start=base + timedelta(milliseconds=1355),
            ),
        ]
        gen = _generation(
            span_id="gen",
            parent_span_id="root",
            service="models",
            operation="models.generation",
            start=base + timedelta(milliseconds=225),
        )
        ordered = dash._build_span_tree([root] + children, [gen])
        assert len(ordered) == 7
        assert ordered[0]["span"].span_id == "root"
        assert ordered[0]["depth"] == 0
        # Every other item is a direct child of root → depth 1.
        for item in ordered[1:]:
            assert item["depth"] == 1
        # Generation should appear in the right time-sorted position
        # (between guardrails.validate_input and guardrails.filter_output).
        ops = [item["span"].operation for item in ordered[1:]]
        assert ops == [
            "sessions.get_messages",
            "data.search",
            "guardrails.validate_input",
            "models.generation",
            "guardrails.filter_output",
            "sessions.add_messages",
        ]


# ---------------------------------------------------------------------------
# _extract_io
# ---------------------------------------------------------------------------


def _trace(*, input="", output="", spans=None, generations=None):
    return SimpleNamespace(
        input=input, output=output, spans=spans or [], generations=generations or []
    )


class TestExtractIo:
    def test_trace_input_wins_when_set(self):
        trace = _trace(
            input="user question",
            output="assistant answer",
            spans=[_span(span_id="s", attributes={"input": "different", "output": "different"})],
        )
        assert dash._extract_io(trace) == ("user question", "assistant answer")

    def test_span_input_attribute_picked_up(self):
        trace = _trace(
            spans=[_span(span_id="root", attributes={"input": "from span", "output": "out"})]
        )
        assert dash._extract_io(trace) == ("from span", "out")

    def test_alternative_keys_recognised(self):
        trace = _trace(
            spans=[
                _span(
                    span_id="s",
                    attributes={"user_message": "hi", "assistant_message": "hello"},
                )
            ]
        )
        assert dash._extract_io(trace) == ("hi", "hello")

    def test_generation_attributes_used_as_fallback(self):
        gen = _generation(span_id="g")
        gen.span.attributes = {"query": "from gen", "response": "answer"}
        trace = _trace(generations=[gen])
        assert dash._extract_io(trace) == ("from gen", "answer")

    def test_returns_empty_when_nothing_found(self):
        trace = _trace(spans=[_span(span_id="s")])
        assert dash._extract_io(trace) == ("", "")


# ---------------------------------------------------------------------------
# Glossary / score descriptions / service colors
# ---------------------------------------------------------------------------


class TestGlossaryContent:
    def test_every_navigation_page_has_a_glossary_entry(self):
        expected_pages = {"Sessions", "Traces", "Cost", "Metrics", "Service Health", "Logs"}
        assert set(dash._PAGE_GLOSSARY.keys()) == expected_pages

    def test_each_glossary_paragraph_mentions_the_relevant_primitive(self):
        # Sessions paragraph should reference session_id; Traces should
        # reference spans + generations; etc. Catches a future copy/paste
        # mistake that swaps two pages' explainers.
        assert "session_id" in dash._PAGE_GLOSSARY["Sessions"]
        assert "span" in dash._PAGE_GLOSSARY["Traces"]
        assert "generation" in dash._PAGE_GLOSSARY["Traces"]
        assert "Cost" not in dash._PAGE_GLOSSARY["Traces"]  # no obvious overlap
        assert "Listing 7.13" in dash._PAGE_GLOSSARY["Cost"]
        assert "Counters" in dash._PAGE_GLOSSARY["Metrics"]
        assert "trace_id" in dash._PAGE_GLOSSARY["Logs"]


class TestScoreDescriptions:
    def test_walkthrough_score_names_are_documented(self):
        # quickstart_observability_walkthrough writes these three score names.
        # Every one of them must have a fallback rubric so the dashboard's
        # Scores subtable can populate the `description` column even if a
        # producer forgets to set the score's `comment`.
        for name in ("helpfulness", "correctness", "retrieval_relevance"):
            assert name in dash._SCORE_DESCRIPTIONS
            assert dash._SCORE_DESCRIPTIONS[name].strip()

    def test_four_source_types_are_covered(self):
        assert set(dash._SOURCE_DESCRIPTIONS.keys()) == {
            "AUTOMATED",
            "MODEL_JUDGE",
            "HUMAN",
            "USER_FEEDBACK",
        }


class TestServiceColors:
    def test_every_platform_service_has_a_color(self):
        # Mirror the service list from services/shared/server.py SERVICE_PORTS.
        for service in (
            "gateway",
            "sessions",
            "models",
            "data",
            "guardrails",
            "tools",
            "workflow",
            "observability",
            "experiments",
        ):
            assert service in dash._SERVICE_COLORS, f"missing color for {service}"

    def test_fallback_color_is_a_valid_hex(self):
        # Hex strings start with '#' and have 6 hex digits or are rgba.
        color = dash._SERVICE_COLOR_FALLBACK
        assert color.startswith("#")
        assert len(color) == 7


pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")
