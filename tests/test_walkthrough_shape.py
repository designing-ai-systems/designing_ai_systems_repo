"""Shape assertions for examples/quickstart_observability_walkthrough.py.

The walkthrough is meant to produce the chapter's Figure 7.5 trace
shape exactly: six spans + one generation + three scores per turn, all
under one session_id. If a future edit accidentally drops a service
span or reintroduces the wrapping models.chat span, this test catches
it without anyone having to open the dashboard.
"""

from __future__ import annotations

import datetime
import sys
import uuid
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

# Importing the walkthrough imports genai_platform too; we don't need a
# real platform — we'll synthesize through a mock.
from examples import quickstart_observability_walkthrough as walk  # noqa: E402


class _RecordingObservability:
    """Stand-in for ``platform.observability`` that records every call."""

    def __init__(self) -> None:
        self.spans: List[Any] = []
        self.generations: List[Any] = []
        self.scores: List[Dict[str, Any]] = []
        self.flushes = 0

    def record_span(self, span) -> None:
        self.spans.append(span)

    def record_generation(self, generation) -> None:
        self.generations.append(generation)

    def record_score(self, **kwargs) -> str:
        self.scores.append(kwargs)
        return uuid.uuid4().hex

    def flush(self) -> None:
        self.flushes += 1


def _run_one_turn(turn_index: int = 1) -> _RecordingObservability:
    rec = _RecordingObservability()
    platform = SimpleNamespace(observability=rec)
    walk.synthesize_turn(
        platform,
        session_id="session-test",
        trace_id="trace-test",
        turn_index=turn_index,
        started_at=datetime.datetime(2026, 1, 1, 12, 0, tzinfo=datetime.timezone.utc),
        user_input="hi",
        assistant_output="hello",
    )
    return rec


class TestWalkthroughTraceShape:
    def test_one_turn_writes_six_spans_one_generation_three_scores(self):
        rec = _run_one_turn()
        assert len(rec.spans) == 6, f"expected 6 spans, got {len(rec.spans)}"
        assert len(rec.generations) == 1, (
            f"expected 1 generation; got {len(rec.generations)}. The walkthrough should NOT "
            f"emit a wrapping models.chat span around the generation (Listing 7.7)."
        )
        assert len(rec.scores) == 3, f"expected 3 scores, got {len(rec.scores)}"

    def test_no_models_chat_span(self):
        """Regression: the walkthrough used to write a models.chat parent
        span containing the generation. Per Listing 7.7 the chat call IS
        the generation, no wrapper span needed."""
        rec = _run_one_turn()
        operations = [s.operation for s in rec.spans]
        assert "models.chat" not in operations, (
            "models.chat span should not exist; the generation lives directly "
            "under the gateway root."
        )

    def test_full_pipeline_services_covered(self):
        """The chapter walkthrough touches gateway, sessions, data,
        guardrails, models. All five should appear at least once."""
        rec = _run_one_turn()
        span_services = {s.service for s in rec.spans}
        gen_services = {g.span.service for g in rec.generations}
        all_services = span_services | gen_services
        for required in ("gateway", "sessions", "data", "guardrails", "models"):
            assert required in all_services, f"missing {required} in waterfall"

    def test_span_operations_match_figure_7_5(self):
        rec = _run_one_turn()
        ops = sorted(s.operation for s in rec.spans)
        assert ops == sorted(
            [
                "gateway.handle_request",
                "sessions.get_messages",
                "data.search",
                "guardrails.validate_input",
                "guardrails.filter_output",
                "sessions.add_messages",
            ]
        )

    def test_generation_parents_to_gateway_root(self):
        """The generation must be a direct child of the gateway root span
        — not orphaned, not nested under some intermediate span."""
        rec = _run_one_turn()
        gateway = next(s for s in rec.spans if s.operation == "gateway.handle_request")
        gen = rec.generations[0]
        assert gen.span.parent_span_id == gateway.span_id, (
            f"generation's parent_span_id should be gateway's span_id "
            f"({gateway.span_id}); got {gen.span.parent_span_id}"
        )


class TestWalkthroughScores:
    def test_all_three_source_types_demonstrated(self):
        """Figure 7.8 calls out three score sources — AUTOMATED,
        MODEL_JUDGE, HUMAN. A single turn of the walkthrough should
        produce at least one score from each so a reader sees all
        three sources in the dashboard."""
        rec = _run_one_turn()
        sources = {s["source"] for s in rec.scores}
        assert sources == {"AUTOMATED", "MODEL_JUDGE", "HUMAN"}

    def test_score_names_match_documented_rubrics(self):
        rec = _run_one_turn()
        names = sorted(s["name"] for s in rec.scores)
        assert names == sorted(["helpfulness", "correctness", "retrieval_relevance"])

    def test_every_score_carries_explanatory_comment(self):
        """The walkthrough used to record scores with one-word comments
        ("claude-3.5 judge") that taught the reader nothing. Each score
        should now carry a multi-word rubric-style comment."""
        rec = _run_one_turn()
        for s in rec.scores:
            assert s.get("comment"), f"score {s['name']} has no comment"
            # Cheap heuristic: a real rubric description has several words.
            assert len(s["comment"].split()) >= 5, (
                f"score {s['name']} comment too terse: {s['comment']!r}"
            )

    def test_value_types_match_source_conventions(self):
        rec = _run_one_turn()
        by_name = {s["name"]: s for s in rec.scores}
        # helpfulness is a continuous LLM-judge score; numeric.
        assert isinstance(by_name["helpfulness"]["value"], float)
        # correctness is a categorical human label.
        assert isinstance(by_name["correctness"]["value"], str)
        # retrieval_relevance is a derived numeric.
        assert isinstance(by_name["retrieval_relevance"]["value"], float)


class TestWalkthroughTrace:
    def test_input_and_output_recorded_on_root_span(self):
        """The Trace primitive carries user input + assistant output. The
        walkthrough sets them as attributes on the gateway root span so
        the dashboard's _extract_io helper finds them."""
        rec = _run_one_turn()
        root = next(s for s in rec.spans if s.operation == "gateway.handle_request")
        assert "input" in root.attributes
        assert "output" in root.attributes

    def test_session_id_lifted_via_attribute(self):
        """The Trace's session_id field is populated by assemble_trace
        scanning span attributes. Every span the walkthrough writes must
        carry session_id so the Sessions page can group correctly."""
        rec = _run_one_turn()
        for span in rec.spans:
            assert span.attributes.get("session_id"), (
                f"span {span.operation} missing session_id attribute"
            )

    def test_per_turn_cost_grows_with_turn_index(self):
        """The walkthrough scales prompt tokens / cost slightly per turn
        so a multi-turn run has visually distinguishable rows on the cost
        chart. Sanity check that the parameter does propagate."""
        rec_a = _run_one_turn(turn_index=1)
        rec_b = _run_one_turn(turn_index=3)
        assert rec_b.generations[0].cost_usd > rec_a.generations[0].cost_usd


class TestWalkthroughParenting:
    def test_every_child_span_parents_to_a_real_span(self):
        """No dangling parent_span_id values — every non-root span's
        parent should be present in the same trace's span set. Prevents
        orphans from masquerading as roots in the waterfall."""
        rec = _run_one_turn()
        ids = {s.span_id for s in rec.spans} | {g.span.span_id for g in rec.generations}
        for s in rec.spans + [g.span for g in rec.generations]:
            if s.parent_span_id:
                assert s.parent_span_id in ids, f"orphan span {s.span_id}: parent not present"

    def test_each_service_appears_at_least_as_a_child_of_gateway(self):
        rec = _run_one_turn()
        gateway = next(s for s in rec.spans if s.operation == "gateway.handle_request")
        children_services = Counter(
            s.service for s in rec.spans if s.parent_span_id == gateway.span_id
        )
        gen_children_services = Counter(
            g.span.service for g in rec.generations if g.span.parent_span_id == gateway.span_id
        )
        all_children = children_services + gen_children_services
        # Every service besides gateway itself should appear at least once as a child.
        for required in ("sessions", "data", "guardrails", "models"):
            assert all_children[required] >= 1, f"no {required} span parented to gateway"
