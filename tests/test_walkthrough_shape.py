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
        self.logs: List[Dict[str, Any]] = []
        self.counters: List[Dict[str, Any]] = []
        self.histograms: List[Dict[str, Any]] = []
        self.flushes = 0

    def record_span(self, span) -> None:
        self.spans.append(span)

    def record_generation(self, generation) -> None:
        self.generations.append(generation)

    def record_score(self, **kwargs) -> str:
        self.scores.append(kwargs)
        return uuid.uuid4().hex

    def log(self, **kwargs) -> None:
        self.logs.append(kwargs)

    def record_counter(self, name, value=1.0, labels=None) -> None:
        self.counters.append({"name": name, "value": value, "labels": dict(labels or {})})

    def record_histogram(self, name, value, labels=None) -> None:
        self.histograms.append({"name": name, "value": value, "labels": dict(labels or {})})

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

    def test_per_turn_total_duration_grows_with_turn_index(self):
        """As a conversation grows, real services slow down: bigger
        session history takes longer to fetch, prompts grow, generations
        run longer. The walkthrough should reflect that so a reader
        scanning the Sessions page sees three distinguishable rows
        instead of three identical 1500ms entries."""
        rec_a = _run_one_turn(turn_index=1)
        rec_c = _run_one_turn(turn_index=3)
        root_a = next(s for s in rec_a.spans if s.operation == "gateway.handle_request")
        root_c = next(s for s in rec_c.spans if s.operation == "gateway.handle_request")
        assert root_c.duration_ms > root_a.duration_ms + 100, (
            f"turn 3 should take noticeably longer than turn 1; got "
            f"turn 1 = {root_a.duration_ms} ms, turn 3 = {root_c.duration_ms} ms"
        )

    def test_per_turn_generation_duration_grows_with_turn_index(self):
        """The generation step in particular should slow down across turns
        — larger prompts take longer to process. Otherwise the latency
        story the chapter tells (latency tracks context growth) doesn't
        show up in the demo data."""
        rec_a = _run_one_turn(turn_index=1)
        rec_c = _run_one_turn(turn_index=3)
        assert rec_c.generations[0].span.duration_ms > rec_a.generations[0].span.duration_ms


class TestWalkthroughLogs:
    def test_each_turn_emits_log_events_tied_to_the_trace(self):
        """Listing 7.2 / 7.3 / section 7.4.1: structured logs are how
        services record detailed events that go beyond a span's attributes.
        Without any logs the Logs page filtered by trace_id returns
        nothing, which makes the chapter's trace → logs workflow
        (Figure 7.7) impossible to demonstrate."""
        rec = _run_one_turn()
        assert rec.logs, "walkthrough emits no log events; trace → logs flow is broken"
        # Every log must carry the trace_id so the dashboard's filter works.
        for entry in rec.logs:
            assert entry.get("trace_id") == "trace-test", (
                f"log {entry.get('event_type')!r} missing/wrong trace_id: {entry.get('trace_id')!r}"
            )

    def test_log_severities_cover_at_least_info(self):
        rec = _run_one_turn()
        severities = {entry.get("severity") for entry in rec.logs}
        assert "INFO" in severities, "expected at least one INFO log per turn"

    def test_logs_reference_real_event_types(self):
        rec = _run_one_turn()
        # The chapter calls out event_type as the field readers filter on.
        # Each log must have a non-empty event_type so the Logs page's
        # filter-by-event-type works.
        for entry in rec.logs:
            assert entry.get("event_type"), f"log entry missing event_type: {entry}"

    def test_turn_two_demonstrates_a_warning_log(self):
        """Listing 7.3 specifically shows a provider-fallback WARNING log.
        Including it in one of the turns lets a reader filter by
        severity=WARNING on the Logs page and see exactly the example
        from the book."""
        rec = _run_one_turn(turn_index=2)
        warning_logs = [e for e in rec.logs if e.get("severity") == "WARNING"]
        assert warning_logs, "turn 2 should emit at least one WARNING log (Listing 7.3 fallback)"


class TestWalkthroughMetrics:
    def test_each_turn_emits_listing_7_8_counters_and_histograms(self):
        """Listing 7.8's headline metrics — request total, latency histogram,
        and cost counter — must land per turn or the Metrics page has no
        time series to plot."""
        rec = _run_one_turn()
        counter_names = {c["name"] for c in rec.counters}
        histogram_names = {h["name"] for h in rec.histograms}
        assert "ai.platform.models.requests_total" in counter_names
        assert "ai.platform.models.cost_usd" in counter_names
        assert "ai.platform.models.request_duration_ms" in histogram_names

    def test_data_service_metrics_emitted(self):
        """Listing 7.4 calls out data.search_duration_ms and data.relevance_score
        as Data Service histograms. Without them the Metrics page can't show
        the retrieval-side latency or relevance distribution."""
        rec = _run_one_turn()
        histogram_names = {h["name"] for h in rec.histograms}
        assert "ai.platform.data.search_duration_ms" in histogram_names
        assert "ai.platform.data.relevance_score" in histogram_names

    def test_guardrails_latency_emitted(self):
        rec = _run_one_turn()
        histogram_names = {h["name"] for h in rec.histograms}
        assert "ai.platform.guardrails.evaluation_duration_ms" in histogram_names

    def test_metric_labels_carry_dimensions_for_filtering(self):
        """The chapter (section 7.4.2) is explicit: metrics must carry
        dimensional labels (provider / model / workflow_id) so a query
        like 'p95 latency for gpt-4o in patient-intake' can resolve.
        Every model metric should have these labels."""
        rec = _run_one_turn()
        model_metrics = [
            m for m in rec.counters + rec.histograms if m["name"].startswith("ai.platform.models.")
        ]
        assert model_metrics, "no model-service metrics emitted"
        for m in model_metrics:
            labels = m["labels"]
            for required in ("provider", "model", "workflow_id"):
                assert required in labels, f"metric {m['name']} missing label {required}"

    def test_request_duration_histogram_reflects_chat_duration(self):
        """The sample submitted to the latency histogram should equal the
        chat duration. If someone hard-codes 0.0 here (the bug from
        commit 1 of the chapter-7 follow-up), this catches it."""
        rec = _run_one_turn(turn_index=2)
        samples = [
            h["value"]
            for h in rec.histograms
            if h["name"] == "ai.platform.models.request_duration_ms"
        ]
        assert samples, "no MODEL_REQUEST_DURATION samples"
        # Turn 2 generation runs ~1250ms; the metric should match that.
        assert samples[0] > 1000, f"unrealistic latency sample: {samples[0]}"

    def test_fallback_counter_only_fires_on_turn_two(self):
        """The fallback log appears only on turn 2 (Listing 7.3 example).
        The matching fallback counter should mirror that — incrementing
        once on turn 2 and zero times on turns 1 and 3."""

        def count_in(rec):
            return sum(1 for c in rec.counters if c["name"] == "ai.platform.models.fallbacks_total")

        assert count_in(_run_one_turn(turn_index=1)) == 0
        assert count_in(_run_one_turn(turn_index=2)) == 1
        assert count_in(_run_one_turn(turn_index=3)) == 0


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
