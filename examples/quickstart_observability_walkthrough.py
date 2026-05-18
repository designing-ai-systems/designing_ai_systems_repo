"""
Quickstart: walkthrough of section 7.2.1's worked example — a single
multi-turn conversation rendered as three full Figure 7.5 traces.

This is a Chapter 7 (Observability) demo. The word "session" appears
because the chapter's data model has Session as a **grouping primitive
above traces** (section 7.2): when a patient asks three questions in a
row, all three traces share one ``session_id``. This is distinct from
the Session Service in Chapter 4, which manages session lifecycle —
this demo only borrows the ``session_id`` for trace grouping.

The companion file ``quickstart_observability.py`` is the headlines
demo (Listing 7.10 custom span + Listing 7.13 cost drill-down). This
walkthrough is the deeper tour: three patient-intake turns sharing
one ``session_id``, each turn producing the nested trace from
Figure 7.5:

    gateway.handle_request
      ├── sessions.get_messages
      ├── data.search
      ├── guardrails.validate_input
      ├── models.generation  (LLM call: token counts, cost, TTFT)
      ├── guardrails.filter_output
      └── sessions.add_messages

Six spans + one generation per trace. Per Listing 7.7, the Model
Service's Chat call IS the generation — no wrapping ``models.chat``
span.

Three quality scores per trace (Listing 7.11) cover all the source
types from Figure 7.8: ``helpfulness`` from a MODEL_JUDGE,
``correctness`` from a HUMAN reviewer, and ``retrieval_relevance``
from an AUTOMATED heuristic. Each carries a ``comment`` describing
its rubric and any relevant metadata (judge model, reviewer email).

The example writes spans / generations / scores directly through the
SDK; no live model calls, no API keys needed. After it finishes,
visit http://localhost:8501 and click through:

  - **Sessions**:   one row, 3 turns rolled up
  - **Traces**:     three traces, each showing the indented waterfall
                    and the conversation panel
  - **Cost**:       drill-down by model and workflow
  - **Metrics**:    request-duration percentiles over the lookback
  - **Service Health**: span counts per service

Run:  uv run python examples/quickstart_observability_walkthrough.py
"""

from __future__ import annotations

import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from examples._bootstrap import start_services_unless_running
from genai_platform import GenAIPlatform
from services.gateway.main import main as start_gateway
from services.observability.models import Generation, Span
from services.observability.service import ObservabilityServiceImpl
from services.shared.server import run_aio_service_main


def start_observability() -> None:
    run_aio_service_main("observability", ObservabilityServiceImpl)


# ---------------------------------------------------------------------------
# Trace synthesis
# ---------------------------------------------------------------------------

WORKFLOW_ID = "patient-intake"
USER_ID = "patient-12345"


def _span(
    trace_id: str,
    span_id: str,
    *,
    service: str,
    operation: str,
    start: datetime,
    duration_ms: float,
    parent_span_id: str = "",
    attributes: dict | None = None,
    status: str = "OK",
) -> Span:
    return Span(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        service=service,
        operation=operation,
        start_time=start,
        end_time=start + timedelta(milliseconds=duration_ms),
        status=status,
        attributes=attributes or {},
    )


def synthesize_turn(
    platform: GenAIPlatform,
    *,
    session_id: str,
    trace_id: str,
    turn_index: int,
    started_at: datetime,
    user_input: str,
    assistant_output: str,
) -> None:
    """Write one full trace — six spans + one generation, plus three scores
    — for a single turn in the session.

    The Model Service produces ONE generation per Chat call (Listing 7.7),
    so the LLM step is recorded as a generation directly under the gateway
    root, not as a `models.chat` span containing a child generation.
    """
    # Common attributes that get lifted onto the Trace at assembly time.
    base_attrs = {
        "session_id": session_id,
        "workflow_id": WORKFLOW_ID,
        "user_id": USER_ID,
    }

    # Per-step durations scale with turn_index so each turn looks
    # distinguishable on the dashboard. The story this tells: as the
    # conversation grows, session history retrieval slows (more turns to
    # load), the prompt gets bigger so the generation runs longer, and
    # data.search drifts a bit between calls. Guardrail + session-write
    # spans stay constant — they don't see the growing context.
    growth = turn_index - 1  # 0 for turn 1, 1 for turn 2, 2 for turn 3...
    sessions_get_ms = 45.0 + 20.0 * growth
    data_search_ms = 120.0 + (5.0 if growth % 2 == 0 else -10.0) * growth
    guardrails_in_ms = 35.0
    chat_duration_ms = 1100.0 + 150.0 * growth  # bigger prompt → slower
    ttft_ms = 340.0 + 30.0 * growth
    guardrails_out_ms = 30.0
    sessions_add_ms = 80.0
    # Gateway overhead beyond the sum of its children (network hops,
    # routing, response serialisation). Small constant.
    gateway_overhead_ms = 70.0
    total_duration_ms = (
        sessions_get_ms
        + data_search_ms
        + guardrails_in_ms
        + chat_duration_ms
        + guardrails_out_ms
        + sessions_add_ms
        + gateway_overhead_ms
    )

    # Root span: gateway.handle_request (the trace root).
    root_id = uuid.uuid4().hex
    platform.observability.record_span(
        _span(
            trace_id,
            root_id,
            service="gateway",
            operation="gateway.handle_request",
            start=started_at,
            duration_ms=total_duration_ms,
            attributes={
                **base_attrs,
                "input": user_input,
                "output": assistant_output,
                "tags": "patient-intake,turn-" + str(turn_index),
            },
        )
    )

    # Child spans nested under the root, sequenced realistically.
    cursor = started_at + timedelta(milliseconds=20)  # 20ms before first child

    def step(service: str, operation: str, duration_ms: float, extra: dict | None = None) -> None:
        nonlocal cursor
        platform.observability.record_span(
            _span(
                trace_id,
                uuid.uuid4().hex,
                service=service,
                operation=operation,
                start=cursor,
                duration_ms=duration_ms,
                parent_span_id=root_id,
                attributes={**base_attrs, **(extra or {})},
            )
        )
        cursor += timedelta(milliseconds=duration_ms)

    step(
        "sessions",
        "sessions.get_messages",
        sessions_get_ms,
        {"messages_retrieved": str(turn_index * 2)},
    )
    step(
        "data",
        "data.search",
        data_search_ms,
        {
            "index_name": "patient_procedures",
            "num_results": "4",
            "top_relevance_score": "0.82",
        },
    )
    step(
        "guardrails",
        "guardrails.validate_input",
        guardrails_in_ms,
        {"policies": "no_medical_advice,pii_detection", "result": "passed"},
    )

    # models.generation — the chapter's "where most cost + latency lives."
    # No wrapping `models.chat` span: Listing 7.7 has the Model Service use
    # `trace_generation` directly, so the Chat call IS the generation.
    chat_start = cursor
    gen_span = _span(
        trace_id,
        uuid.uuid4().hex,
        service="models",
        operation="models.generation",
        start=chat_start,
        duration_ms=chat_duration_ms,
        parent_span_id=root_id,
        attributes={**base_attrs, "requested_model": "gpt-4o"},
    )
    platform.observability.record_generation(
        Generation(
            span=gen_span,
            model="gpt-4o",
            provider="openai",
            requested_model="gpt-4o",
            prompt_tokens=3100 + turn_index * 200,
            completion_tokens=180 + turn_index * 20,
            cost_usd=0.018 + turn_index * 0.002,
            cache_hit=False,
            fallback_used=False,
            time_to_first_token_ms=ttft_ms,
        )
    )
    cursor = chat_start + timedelta(milliseconds=chat_duration_ms)

    step(
        "guardrails",
        "guardrails.filter_output",
        guardrails_out_ms,
        {"policies": "pii_redaction", "result": "passed"},
    )
    step(
        "sessions",
        "sessions.add_messages",
        sessions_add_ms,
        {"messages_added": "2"},
    )

    # Three quality scores (Listing 7.11) covering all the source types
    # the chapter calls out. Attached *after* the response — in production
    # they'd flow in asynchronously through a scoring rule (Listing 7.18).
    platform.observability.record_score(
        trace_id=trace_id,
        name="helpfulness",
        value=0.85 + 0.03 * turn_index,
        source="MODEL_JUDGE",
        comment=(
            "0.0–1.0 score from claude-haiku-4-5 rating how directly the "
            "assistant's response answers the user's question. Sampled at "
            "10% of production traffic via an online scoring rule."
        ),
        metadata={
            "judge_model": "claude-haiku-4-5",
            "rubric": "helpfulness",
        },
    )
    platform.observability.record_score(
        trace_id=trace_id,
        name="correctness",
        value="correct",
        source="HUMAN",
        comment=(
            "Reviewer label: correct / partially_correct / incorrect. "
            "Sampled at ~1% for ground-truth calibration of automated and "
            "model-judge scorers (chapter 7, Figure 7.8)."
        ),
        metadata={
            "reviewer": "sarah@healthfirst.com",
            "rubric": "correctness",
        },
    )
    platform.observability.record_score(
        trace_id=trace_id,
        name="retrieval_relevance",
        value=0.82,
        source="AUTOMATED",
        comment=(
            "Top relevance score across documents returned by data.search "
            "on this trace. Pulled from the data.search span's "
            "`top_relevance_score` attribute — no model call required."
        ),
    )


def main() -> None:
    print("=" * 60)
    print("  Quickstart: multi-turn Session with Figure 7.5 traces")
    print("=" * 60)
    started = start_services_unless_running([start_observability, start_gateway])
    if started:
        print("Services ready.\n")

    platform = GenAIPlatform()
    session_id = f"session-{uuid.uuid4().hex[:8]}"
    print(f"\nSynthesizing 3 turns under session_id = {session_id}")
    print(f"workflow_id = {WORKFLOW_ID!r}   user_id = {USER_ID!r}\n")

    # Three turns of a patient intake conversation, spaced ~30s apart.
    base_time = datetime.now(timezone.utc) - timedelta(minutes=10)
    turns = [
        (
            "Can I bring my medical records digitally or do I need paper copies?",
            "You can upload digital records through our patient portal — PDF or "
            "image formats. Paper copies aren't required if your digital records "
            "are legible.",
        ),
        (
            "What documents do I need for my first appointment?",
            "Please bring your insurance card, a photo ID, and any specialist "
            "referral forms. If you've uploaded medical records to the portal, "
            "no paper copies are needed.",
        ),
        (
            "When should I arrive for my appointment?",
            "Please arrive 15 minutes before your scheduled time so we can "
            "complete check-in. Your appointment with Dr. Patel is at 10:30 AM "
            "this Tuesday.",
        ),
    ]
    trace_ids = []
    for i, (user_msg, asst_msg) in enumerate(turns, start=1):
        trace_id = uuid.uuid4().hex
        started_at = base_time + timedelta(seconds=30 * (i - 1))
        synthesize_turn(
            platform,
            session_id=session_id,
            trace_id=trace_id,
            turn_index=i,
            started_at=started_at,
            user_input=user_msg,
            assistant_output=asst_msg,
        )
        trace_ids.append(trace_id)
        print(f"  turn {i}: trace_id={trace_id}")

    # Flush so the dashboard sees the data immediately on next refresh.
    platform.observability.flush()

    print("\n" + "=" * 60)
    print("  Done. Open the dashboard:")
    print(f"    Sessions page → look for {session_id}")
    print("    Traces page   → drill into any of the three trace IDs above")
    print("=" * 60)


if __name__ == "__main__":
    main()
