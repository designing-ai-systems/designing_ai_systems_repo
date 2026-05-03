"""
Quickstart: platform.observability — custom span + cost drill-down.

This is the smallest possible demo of the Observability Service from a
workflow developer's point of view. It boots the Observability Service
and Gateway in-process and shows:

  1. Listing 7.10 — wrap a workflow step in `trace_operation` so the
     service automatically captures a span for it.
  2. Listing 7.13 — query a cost report by team, then drill down into
     models for a specific team.

Run:  python examples/quickstart_observability.py
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


def start_observability():
    run_aio_service_main("observability", ObservabilityServiceImpl)


def start_in_thread(target, name):
    threading.Thread(target=target, daemon=True, name=name).start()


def _make_span(trace_id, span_id, *, service, operation, attrs=None):
    now = datetime.now(timezone.utc)
    return Span(
        trace_id=trace_id,
        span_id=span_id,
        service=service,
        operation=operation,
        start_time=now - timedelta(milliseconds=10),
        end_time=now,
        attributes=attrs or {},
    )


def custom_workflow_step(platform: GenAIPlatform):
    """Listing 7.10: a workflow function that owns its own custom span."""
    print("\n[Listing 7.10] custom 'trace_operation' inside a workflow function")
    with platform.observability.trace_operation(
        "custom_rerank",
        document_count=20,
        algorithm="cross_encoder",
    ) as ctx:
        # Pretend to do work...
        time.sleep(0.05)
        print(f"  trace_id = {ctx.trace_id}")
        print(f"  span_id  = {ctx.span_id}")
    return ctx


def cost_drill_down(platform: GenAIPlatform):
    """Listing 7.13: cost drill-down over a synthetic month of usage."""
    print("\n[Listing 7.13] cost drill-down by team, then by model")
    now = datetime.now(timezone.utc)
    for team, model, cost in [
        ("engineering", "gpt-4o", 1.30),
        ("engineering", "gpt-4o-mini", 0.10),
        ("data-science", "claude-sonnet-4-5", 0.50),
        ("data-science", "claude-haiku-4-5", 0.05),
        ("product", "gpt-4o", 0.20),
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
        start_time=now - timedelta(days=30),
        end_time=now + timedelta(hours=1),
        group_by=["team"],
    )
    print(f"  total ${by_team.total_cost_usd:.2f}")
    for b in by_team.buckets:
        print(
            f"   - team={b.dimensions['team']:>14}: "
            f"${b.cost_usd:>5.2f}  ({b.request_count} requests)"
        )

    print("  drill-down on engineering by model:")
    by_model = platform.observability.get_cost_report(
        start_time=now - timedelta(days=30),
        end_time=now + timedelta(hours=1),
        group_by=["model"],
        filters={"team": "engineering"},
    )
    for b in by_model.buckets:
        print(
            f"     {b.dimensions['model']:>14}: ${b.cost_usd:>5.2f}  ({b.request_count} requests)"
        )


def view_trace(platform: GenAIPlatform, ctx):
    print("\n[Listing 7.5] read the custom span back as a Trace")
    trace = platform.observability.get_trace(ctx.trace_id)
    if trace is None:
        print("  (trace not yet visible — flushing and retrying)")
        platform.observability.flush()
        trace = platform.observability.get_trace(ctx.trace_id)
    print(
        f"  trace_id={trace.trace_id}  spans={len(trace.spans)}  "
        f"duration_ms={trace.total_duration_ms:.1f}"
    )
    for s in trace.spans:
        print(f"   - {s.service}.{s.operation}  [{s.status}]  {s.duration_ms:.1f}ms")


def main():
    print("=" * 60)
    print("  Quickstart: platform.observability")
    print("=" * 60)
    print("\nStarting Observability service and Gateway...")
    start_in_thread(start_observability, "ObservabilityService")
    time.sleep(1)
    start_in_thread(start_gateway, "Gateway")
    time.sleep(1)
    print("Services ready.")

    platform = GenAIPlatform()
    try:
        ctx = custom_workflow_step(platform)
        # Make sure background flush has time to ship the span.
        platform.observability.flush()
        view_trace(platform, ctx)
        cost_drill_down(platform)
        print("\n" + "=" * 60)
        print("  Quickstart complete")
        print("=" * 60)
    except Exception as e:  # noqa: BLE001
        import traceback

        print(f"\nError: {e}")
        traceback.print_exc()
    finally:
        platform.observability.shutdown()


if __name__ == "__main__":
    main()
