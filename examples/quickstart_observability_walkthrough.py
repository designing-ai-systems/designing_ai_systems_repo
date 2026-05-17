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
      ├── models.chat
      │     └── models.generation  (token counts, cost, TTFT)
      ├── guardrails.filter_output
      └── sessions.add_messages

Plus two quality scores per trace (Listing 7.11): a model-judge
helpfulness score and a human correctness score.

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
    """Write one full trace — seven spans (incl. a nested generation)
    plus two scores — for a single turn in the session."""
    # Common attributes that get lifted onto the Trace at assembly time.
    base_attrs = {
        "session_id": session_id,
        "workflow_id": WORKFLOW_ID,
        "user_id": USER_ID,
    }

    # Root span: gateway.handle_request (the trace root).
    root_id = uuid.uuid4().hex
    duration_total_ms = 1500.0
    platform.observability.record_span(
        _span(
            trace_id,
            root_id,
            service="gateway",
            operation="gateway.handle_request",
            start=started_at,
            duration_ms=duration_total_ms,
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

    step("sessions", "sessions.get_messages", 45.0, {"messages_retrieved": str(turn_index * 2)})
    step(
        "data",
        "data.search",
        120.0,
        {
            "index_name": "patient_procedures",
            "num_results": "4",
            "top_relevance_score": "0.82",
        },
    )
    step(
        "guardrails",
        "guardrails.validate_input",
        35.0,
        {"policies": "no_medical_advice,pii_detection", "result": "passed"},
    )

    # models.chat as a parent span; the generation nests under it.
    chat_id = uuid.uuid4().hex
    chat_start = cursor
    chat_duration_ms = 1100.0
    platform.observability.record_span(
        _span(
            trace_id,
            chat_id,
            service="models",
            operation="models.chat",
            start=chat_start,
            duration_ms=chat_duration_ms,
            parent_span_id=root_id,
            attributes={**base_attrs, "requested_model": "gpt-4o"},
        )
    )

    # The Generation — the chapter's "where most cost + latency lives."
    gen_span = _span(
        trace_id,
        uuid.uuid4().hex,
        service="models",
        operation="models.generation",
        start=chat_start + timedelta(milliseconds=30),  # provider call after model selection
        duration_ms=chat_duration_ms - 60,
        parent_span_id=chat_id,
        attributes=base_attrs,
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
            time_to_first_token_ms=340.0,
        )
    )
    cursor = chat_start + timedelta(milliseconds=chat_duration_ms)

    step(
        "guardrails",
        "guardrails.filter_output",
        30.0,
        {"policies": "pii_redaction", "result": "passed"},
    )
    step(
        "sessions",
        "sessions.add_messages",
        80.0,
        {"messages_added": "2"},
    )

    # Two quality scores (Listing 7.11). Attached *after* the response —
    # in production they'd flow in asynchronously through a scoring rule.
    platform.observability.record_score(
        trace_id=trace_id,
        name="helpfulness",
        value=0.85 + 0.03 * turn_index,
        source="MODEL_JUDGE",
        comment="claude-3.5 judge",
        metadata={"judge_model": "claude-haiku-4-5"},
    )
    platform.observability.record_score(
        trace_id=trace_id,
        name="correctness",
        value="correct",
        source="HUMAN",
        comment="reviewer agreed with assistant response",
        metadata={"reviewer": "sarah@healthfirst.com"},
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
