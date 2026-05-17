"""
Streamlit dashboard over the Observability Service.

Reads through the platform SDK — the same path any developer would use.
Multi-page layout via st.navigation: Traces, Cost, Metrics, Service
Health, and Logs. Each page is intentionally short so a reader can see
exactly which SDK calls power which view.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 7.1:  Observability Service contract
  - Listing 7.10: trace_operation (custom spans through the SDK)
  - Listing 7.13: cost drill-down
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st

from genai_platform import GenAIPlatform


def get_platform() -> GenAIPlatform:
    """Cache the GenAIPlatform SDK across reruns."""
    cache = st.session_state.setdefault("_genai_platform_cache", {})
    url = os.environ.get("GENAI_GATEWAY_URL", "localhost:50051")
    if cache.get("url") != url:
        cache.clear()
        cache["url"] = url
        cache["platform"] = GenAIPlatform(gateway_url=url)
    return cache["platform"]


def _utc_window(hours: int) -> tuple[datetime, datetime]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours)
    return start, end


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def page_traces() -> None:
    st.title("Traces")
    st.caption("Search and inspect distributed traces (Listing 7.5).")

    platform = get_platform()
    with st.sidebar:
        workflow_id = st.text_input("Workflow ID")
        user_id = st.text_input("User ID")
        session_id = st.text_input("Session ID")
        hours_back = st.slider("Lookback (hours)", min_value=1, max_value=168, value=24)
        min_duration_ms = st.number_input("Min duration (ms)", min_value=0, value=0)
        limit = st.number_input("Limit", min_value=1, max_value=500, value=50)

    start_time, end_time = _utc_window(hours_back)
    traces = platform.observability.query_traces(
        workflow_id=workflow_id or "",
        user_id=user_id or "",
        session_id=session_id or "",
        start_time=start_time,
        end_time=end_time,
        min_duration_ms=float(min_duration_ms),
        limit=int(limit),
    )

    if not traces:
        st.info("No traces matched the filter. Try widening the time window.")
        return

    rows = []
    for trace in traces:
        rows.append(
            {
                "trace_id": trace.trace_id,
                "workflow_id": trace.workflow_id,
                "duration_ms": round(trace.total_duration_ms, 1),
                "cost_usd": round(trace.total_cost_usd, 4),
                "tokens": trace.total_tokens,
                "spans": len(trace.spans),
                "generations": len(trace.generations),
                "scores": len(trace.scores),
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch")

    trace_id = st.selectbox(
        "Inspect a trace:",
        options=[""] + [r["trace_id"] for r in rows],
        format_func=lambda x: x or "—",
    )
    if trace_id:
        trace = platform.observability.get_trace(trace_id)
        if trace is None:
            st.warning(f"Trace {trace_id} not found")
            return
        st.subheader(f"Trace {trace.trace_id}")
        waterfall = []
        for span in trace.spans:
            waterfall.append(
                {
                    "operation": span.operation,
                    "service": span.service,
                    "kind": "span",
                    "start": span.start_time,
                    "end": span.end_time,
                    "duration_ms": round(span.duration_ms, 1),
                    "status": span.status,
                }
            )
        for gen in trace.generations:
            waterfall.append(
                {
                    "operation": gen.span.operation or "models.generation",
                    "service": gen.span.service,
                    "kind": f"generation ({gen.model})",
                    "start": gen.span.start_time,
                    "end": gen.span.end_time,
                    "duration_ms": round(gen.span.duration_ms, 1),
                    "status": gen.span.status,
                }
            )
        waterfall.sort(key=lambda row: row["start"])
        st.dataframe(pd.DataFrame(waterfall), width="stretch")
        if trace.scores:
            st.markdown("**Scores:**")
            st.dataframe(
                pd.DataFrame(
                    [{"name": s.name, "value": s.value, "source": s.source} for s in trace.scores]
                ),
                width="stretch",
            )


def page_cost() -> None:
    st.title("Cost")
    st.caption("Listing 7.13 — drill into cost by team, workflow, or model.")

    platform = get_platform()
    with st.sidebar:
        group_by = st.multiselect(
            "Group by", options=["model", "provider", "workflow_id", "team"], default=["model"]
        )
        days_back = st.slider("Lookback (days)", min_value=1, max_value=90, value=30)
        granularity = st.selectbox("Granularity", options=["monthly", "weekly", "daily"], index=0)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days_back)
    report = platform.observability.get_cost_report(
        start_time=start,
        end_time=end,
        group_by=group_by or ["model"],
        granularity=granularity,
    )
    st.metric("Total cost (USD)", f"${report.total_cost_usd:,.2f}")

    if not report.buckets:
        st.info("No generations recorded in this window. Run a workflow to populate cost data.")
        return

    rows = []
    for bucket in report.buckets:
        row: Dict[str, Any] = dict(bucket.dimensions)
        row["cost_usd"] = round(bucket.cost_usd, 4)
        row["prompt_tokens"] = bucket.prompt_tokens
        row["completion_tokens"] = bucket.completion_tokens
        row["requests"] = bucket.request_count
        rows.append(row)
    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch")

    if group_by and "cost_usd" in df.columns:
        label_col = group_by[0]
        if label_col in df.columns:
            fig = px.bar(df, x=label_col, y="cost_usd", title=f"Cost by {label_col}")
            st.plotly_chart(fig, width="stretch")


def page_metrics() -> None:
    st.title("Metrics")
    st.caption("Per-metric percentiles over the lookback window.")

    platform = get_platform()
    with st.sidebar:
        metric_name = st.text_input("Metric name", value="ai.platform.models.request_duration_ms")
        hours_back = st.slider("Lookback (hours)", min_value=1, max_value=168, value=24)

    start, end = _utc_window(hours_back)
    aggregations = ["p50", "p95", "p99", "avg", "count", "sum"]
    rows = []
    for agg in aggregations:
        result = platform.observability.query_metrics(
            name=metric_name, aggregation=agg, start_time=start, end_time=end
        )
        rows.append({"aggregation": agg, "value": result.get("value", 0.0)})
    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch")

    sample_count = next((r["value"] for r in rows if r["aggregation"] == "count"), 0.0)
    st.metric("Sample count", int(sample_count))


def page_service_health() -> None:
    st.title("Service Health")
    st.caption("Span error rates over the lookback window, per platform service.")

    platform = get_platform()
    services = [
        "sessions",
        "models",
        "data",
        "tools",
        "guardrails",
        "workflow",
        "observability",
        "experiments",
    ]
    hours_back = st.sidebar.slider("Lookback (hours)", min_value=1, max_value=24, value=1)
    rows = []
    for service in services:
        health = platform.observability.get_service_health(
            service=service, lookback_seconds=hours_back * 3600
        )
        rows.append(
            {
                "service": service,
                "status": health.status,
                "span_count": health.span_count,
                "error_rate": round(health.error_rate, 4),
                "last_span_at": health.last_span_at,
                "detail": health.detail,
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch")


def page_logs() -> None:
    st.title("Logs")
    st.caption("Structured-log search by trace_id, service, or event_type.")

    platform = get_platform()
    with st.sidebar:
        trace_id = st.text_input("Trace ID")
        service = st.text_input("Service")
        event_type = st.text_input("Event type")
        min_severity = st.selectbox(
            "Min severity", ["", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], index=0
        )
        limit = st.number_input("Limit", min_value=10, max_value=500, value=100)

    if not any([trace_id, service, event_type, min_severity]):
        st.info("Set at least one filter on the left to start a search.")
        return

    response = platform.observability.query_logs(
        trace_id=trace_id or "",
        service=service or "",
        event_type=event_type or "",
        min_severity=min_severity or "",
        limit=int(limit),
    )
    events: List[Dict[str, Any]] = response.get("events", []) if isinstance(response, dict) else []
    if not events:
        st.info("No log events matched.")
        return
    st.dataframe(pd.DataFrame(events), width="stretch")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="GenAI Platform — Observability", layout="wide")
    pg = st.navigation(
        [
            st.Page(page_traces, title="Traces", icon="🌊"),
            st.Page(page_cost, title="Cost", icon="💸"),
            st.Page(page_metrics, title="Metrics", icon="📈"),
            st.Page(page_service_health, title="Service Health", icon="❤️"),
            st.Page(page_logs, title="Logs", icon="📝"),
        ]
    )
    pg.run()


if __name__ == "__main__":
    main()
