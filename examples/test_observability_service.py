"""
Comprehensive Observability Service test.

Boots the Gateway and Observability Service in-process and exercises
every RPC end to end. No Postgres, no API keys, no Model Service.

Covers (Listings 7.1, 7.2, 7.4, 7.5, 7.10, 7.11, 7.13):
  - RecordSpan + RecordGeneration / GetTrace round-trip
  - RecordScore + QueryScores
  - IngestLogs + QueryLogs
  - RecordMetrics (counter + histogram) + QueryMetrics (p50/p95/p99)
  - GetCostReport grouped by team and by model
  - SetBudgetAlert + GetBudgetStatus
  - GetServiceHealth

Run:  python examples/test_observability_service.py
"""

from __future__ import annotations

import sys
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from genai_platform import GenAIPlatform
from services.gateway.main import main as start_gateway
from services.observability.models import Generation, Span
from services.observability.service import ObservabilityServiceImpl
from services.shared.server import run_aio_service_main


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


def start_observability():
    run_aio_service_main("observability", ObservabilityServiceImpl)


def start_in_thread(target, name):
    threading.Thread(target=target, daemon=True, name=name).start()


def _make_span(
    trace_id, span_id, *, service, operation, parent_span_id="", attrs=None, status="OK"
):
    now = datetime.now(timezone.utc)
    return Span(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        service=service,
        operation=operation,
        start_time=now - timedelta(milliseconds=10),
        end_time=now,
        status=status,
        attributes=attrs or {},
    )


def test_traces_and_generations(platform: GenAIPlatform):
    section("TEST 1: traces / spans / generations (Listings 7.5, 7.6)")
    trace_id = uuid.uuid4().hex

    platform.observability.record_span(
        _make_span(
            trace_id,
            "s-root",
            service="gateway",
            operation="gateway.handle_request",
            attrs={"workflow_id": "patient-intake"},
        )
    )
    platform.observability.record_span(
        _make_span(
            trace_id,
            "s-data",
            service="data",
            operation="data.search",
            parent_span_id="s-root",
            attrs={"top_relevance_score": "0.81"},
        )
    )

    gen = Generation(
        span=_make_span(
            trace_id,
            "s-gen",
            service="models",
            operation="models.generation",
            parent_span_id="s-root",
        ),
        model="gpt-4o",
        provider="openai",
        prompt_tokens=120,
        completion_tokens=40,
        cost_usd=0.001,
    )
    platform.observability.record_generation(gen)

    platform.observability.record_span(
        _make_span(
            trace_id,
            "s-guard",
            service="guardrails",
            operation="guardrails.validate_output",
            parent_span_id="s-root",
        )
    )

    trace = platform.observability.get_trace(trace_id)
    print(f"  trace_id={trace.trace_id}")
    print(f"  spans={len(trace.spans)}  generations={len(trace.generations)}")
    print(f"  total_cost_usd={trace.total_cost_usd}  total_tokens={trace.total_tokens}")
    return trace_id


def test_scores(platform: GenAIPlatform, trace_id: str):
    section("TEST 2: scores (Listing 7.11)")
    platform.observability.record_score(
        trace_id=trace_id, name="helpfulness", value=0.85, source="MODEL_JUDGE"
    )
    platform.observability.record_score(
        trace_id=trace_id, name="resolved", value=True, source="USER_FEEDBACK"
    )
    platform.observability.record_score(
        trace_id=trace_id, name="tone", value="warm", source="HUMAN"
    )
    helpful = platform.observability.query_scores(trace_id=trace_id, name="helpfulness")
    print(f"  helpfulness scores: {[s.value for s in helpful]}")


def test_logs(platform: GenAIPlatform, trace_id: str):
    section("TEST 3: logs (Listing 7.2)")
    platform.observability.log(
        event_type="model_call",
        message="primary completed",
        severity="INFO",
        trace_id=trace_id,
    )
    platform.observability.log(
        event_type="model_fallback",
        message="primary timed out -> fallback",
        severity="WARNING",
        trace_id=trace_id,
        attributes={"original_provider": "openai", "fallback_provider": "anthropic"},
    )
    platform.observability.flush()
    result = platform.observability.query_logs(trace_id=trace_id)
    print(f"  total_matched={result['total_matched']} events")
    for e in result["events"]:
        print(f"   - [{e.severity}] {e.event_type}: {e.message}")


def test_metrics(platform: GenAIPlatform):
    section("TEST 4: metrics (Listing 7.4) — p50/p95/p99")
    for v in range(1, 51):
        platform.observability.record_histogram(
            "ai.platform.models.request_duration_ms",
            float(v) * 10.0,
            labels={"workflow_id": "patient-intake", "model": "gpt-4o"},
        )
    platform.observability.flush()
    result = platform.observability.query_metrics(
        name="ai.platform.models.request_duration_ms",
        aggregation="p95",
        label_filters={"workflow_id": "patient-intake"},
    )
    print(f"  p95 latency = {result['value']:.1f}ms")
    print(
        f"  percentiles  = p50={result['percentiles'].get('p50', 0):.1f}  "
        f"p95={result['percentiles'].get('p95', 0):.1f}  "
        f"p99={result['percentiles'].get('p99', 0):.1f}"
    )


def test_cost_report(platform: GenAIPlatform):
    section("TEST 5: cost drill-down (Listing 7.13)")
    now = datetime.now(timezone.utc)
    for team, model, cost in [
        ("engineering", "gpt-4o", 0.40),
        ("engineering", "gpt-4o", 0.50),
        ("engineering", "gpt-4o-mini", 0.05),
        ("data-science", "claude-sonnet-4-5", 0.20),
    ]:
        platform.observability.record_generation(
            Generation(
                span=_make_span(
                    uuid.uuid4().hex,
                    uuid.uuid4().hex,
                    service="models",
                    operation="models.generation",
                    attrs={"team": team},
                ),
                model=model,
                provider="openai" if "gpt" in model else "anthropic",
                cost_usd=cost,
            )
        )
    by_team = platform.observability.get_cost_report(
        start_time=now - timedelta(hours=1),
        end_time=now + timedelta(hours=1),
        group_by=["team"],
    )
    print(f"  total ${by_team.total_cost_usd:.2f}")
    for b in by_team.buckets:
        print(f"   - {b.dimensions}: ${b.cost_usd:.2f} ({b.request_count} requests)")
    by_model = platform.observability.get_cost_report(
        start_time=now - timedelta(hours=1),
        end_time=now + timedelta(hours=1),
        group_by=["team", "model"],
        filters={"team": "engineering"},
    )
    print("  drill-down (engineering by model):")
    for b in by_model.buckets:
        print(f"   - {b.dimensions}: ${b.cost_usd:.2f}")


def test_budget(platform: GenAIPlatform):
    section("TEST 6: budgets (SetBudgetAlert / GetBudgetStatus)")
    name = platform.observability.set_budget_alert(
        name="engineering-monthly",
        scope_type="team",
        scope_value="engineering",
        limit_usd=1.0,
        period="monthly",
        thresholds=[0.5, 0.8, 1.0],
        notification_channels=["#eng-budgets"],
    )
    status = platform.observability.get_budget_status(name)
    print(f"  budget '{status.alert.name}' limit=${status.alert.limit_usd:.2f}")
    print(f"  current=${status.current_spend_usd:.2f}  projected=${status.projected_spend_usd:.2f}")
    print(
        f"  percent_used={status.percent_used:.1%}  thresholds_crossed={status.thresholds_crossed}"
    )


def test_service_health(platform: GenAIPlatform):
    section("TEST 7: GetServiceHealth")
    h = platform.observability.get_service_health("models", lookback_seconds=3600)
    print(
        f"  models: status={h.status}  span_count={h.span_count}  "
        f"error_rate={h.error_rate:.2%}  detail={h.detail}"
    )


def main():
    print("=" * 60)
    print("  Observability Service Comprehensive Test")
    print("=" * 60)
    print("\nStarting Observability service and Gateway...")
    start_in_thread(start_observability, "ObservabilityService")
    time.sleep(1)
    start_in_thread(start_gateway, "Gateway")
    time.sleep(1)
    print("Services ready.\n")

    platform = GenAIPlatform()
    try:
        trace_id = test_traces_and_generations(platform)
        test_scores(platform, trace_id)
        test_logs(platform, trace_id)
        test_metrics(platform)
        test_cost_report(platform)
        test_budget(platform)
        test_service_health(platform)
        print("\n" + "=" * 60)
        print("  All Observability Service tests completed")
        print("=" * 60)
    except Exception as e:  # noqa: BLE001
        import traceback

        print(f"\nError: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
