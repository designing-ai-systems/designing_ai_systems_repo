"""
Streamlit dashboard over the Observability Service.

Reads through the platform SDK — the same path any developer would use.
Six pages, one per book primitive plus the operational views:

  - Sessions:  Multi-turn conversations (groups traces by session_id)
  - Traces:    Single requests. Includes the **waterfall** from Figure
               7.6 (span/generation nesting via parent_span_id), the
               trace's input/output (Trace definition, section 7.2), a
               Generations breakout for the LLM-specific fields, and
               cross-page jump buttons that implement the
               trace → logs → metrics workflow from Figure 7.7.
  - Cost:      Drill-down by team / workflow / model (Listing 7.13)
  - Metrics:   p50 / p95 / p99 **time series** over the lookback window
  - Health:    Span error rates per platform service
  - Logs:      Structured log search by trace_id / service / event_type,
               with a "back to the trace" button to close the loop

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Section 7.2:  the five primitives — session / trace / span /
                  generation / score
  - Figure 7.6:   distributed trace timeline (waterfall)
  - Figure 7.7:   three-level debugging workflow (trace → logs → metrics)
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


# Stable colour assignment per platform service so a reader can scan any
# trace and recognise "blue means data, orange means guardrails." Generations
# share the models colour but get a heavier border so they pop out as the
# chapter's "where most cost + latency lives."
_SERVICE_COLORS: Dict[str, str] = {
    "gateway": "#6b7280",
    "sessions": "#14b8a6",
    "data": "#3b82f6",
    "guardrails": "#f59e0b",
    "models": "#8b5cf6",
    "tools": "#ec4899",
    "workflow": "#10b981",
    "observability": "#0ea5e9",
    "experiments": "#a855f7",
    "sdk": "#94a3b8",
}
_SERVICE_COLOR_FALLBACK = "#94a3b8"


# One-paragraph explainer per page, surfaced behind a collapsible
# "📖 How to read this page" expander. Keeps the data model from
# Chapter 7 section 7.2 within reach for a first-time reader without
# the page feeling busy on rereads.
_PAGE_GLOSSARY: Dict[str, str] = {
    "Sessions": (
        "A **session** is one conversation — multiple back-and-forth turns "
        "between a user and the assistant. Each row groups every trace "
        "that shares a `session_id`. Drill into a session to see each "
        "turn's full waterfall and the quality scores attached to it. "
        "The Observability Service doesn't manage session lifecycle "
        "(that's the Session Service from Chapter 4); this page just "
        "reads the `session_id` field on each Trace."
    ),
    "Traces": (
        "A **trace** is one user request — everything the platform did to "
        "produce one assistant response. Inside a trace, **spans** are "
        "units of work in a service (`sessions.get_messages`, "
        "`data.search`, `guardrails.validate_input`). A **generation** is "
        "a specialized span for an LLM call — it carries model name, "
        "token counts, cost, time-to-first-token, and lives in its own "
        "indexed table. **Scores** attach asynchronously and rate "
        "quality (helpfulness, correctness, retrieval relevance)."
    ),
    "Cost": (
        "Per-request costs aggregated by dimension. Listing 7.13's "
        "drill-down: pick a `group_by` (team / workflow / model / "
        "provider), pick a lookback window, see who spent what. The "
        "data comes from `Generation.cost_usd` on every LLM call."
    ),
    "Metrics": (
        "Time-series view of any platform metric. **Counters** "
        "accumulate (total requests, total cost). **Histograms** track "
        "distributions (latency, where p50 / p95 / p99 actually mean "
        "something). The chapter recommends histograms over averages "
        "because AI latency is heavy-tailed — a p50 of 800 ms can hide "
        "a p99 of 12 s."
    ),
    "Service Health": (
        "**Telemetry-derived health, not a liveness probe.** Section 7.3 "
        "is explicit about the distinction: every platform service still "
        "has its own liveness endpoint that load balancers use; this "
        "page reads spans the Observability Service has already received "
        "and reports a derived status per service. < 10 % error rate "
        "= **healthy**, 10–50 % = **degraded**, ≥ 50 % = **critical**, "
        "no spans in the window = **unknown** (not the same as healthy)."
    ),
    "Logs": (
        "Structured log events emitted from within spans. Each event "
        "carries a `trace_id` / `span_id` so you can navigate from a "
        "trace's waterfall straight into its detailed log trail. "
        "Filter by trace_id, service, event_type, or severity."
    ),
}


# Rubric / source descriptions used in the trace-detail scores subtable.
# A score's own `comment` field wins if non-empty; this dict is the
# fallback when a producer didn't set one.
_SCORE_DESCRIPTIONS: Dict[str, str] = {
    "helpfulness": (
        "0.0–1.0 rating of how directly the assistant's response answers "
        "the user's question. Typically a MODEL_JUDGE score sampled at "
        "5–20 % of production traffic."
    ),
    "correctness": (
        "Categorical reviewer label (correct / partially_correct / "
        "incorrect). Typically a HUMAN score sampled at ~1 % for ground "
        "truth, used to calibrate automated scorers (Figure 7.8)."
    ),
    "retrieval_relevance": (
        "Quality of the retrieved context. Typically AUTOMATED — pulled "
        "from the Data Service span's relevance score, no model call "
        "required."
    ),
    "safety_compliance": (
        "Whether the response adhered to safety policies. Usually "
        "categorical (pass / fail) sourced from the Guardrails Service."
    ),
    "content_accuracy": (
        "Factual accuracy of the response. Typically a MODEL_JUDGE or "
        "HUMAN score; expensive to run on all traffic."
    ),
}


_SOURCE_DESCRIPTIONS: Dict[str, str] = {
    "AUTOMATED": "Deterministic heuristic. Cheap, fast, scores every trace.",
    "MODEL_JUDGE": "Another LLM rated this response. Sampled (5–20%).",
    "HUMAN": "Reviewer manually labeled this. Ground truth, sampled (~1%).",
    "USER_FEEDBACK": "End-user signal (e.g. thumbs up/down).",
}


def _glossary_expander(page_name: str) -> None:
    """Render the per-page 'How to read this page' expander."""
    text = _PAGE_GLOSSARY.get(page_name)
    if not text:
        return
    with st.expander("📖 How to read this page", expanded=False):
        st.markdown(text)


def _source_legend_markdown() -> str:
    """Tiny markdown block explaining the four score sources."""
    rows = "  \n".join(f"**{src}** — {desc}" for src, desc in _SOURCE_DESCRIPTIONS.items())
    return rows


# ---------------------------------------------------------------------------
# Helpers for the trace detail view
# ---------------------------------------------------------------------------


def _build_span_tree(spans: List[Any], generations: List[Any]) -> List[Dict[str, Any]]:
    """Return spans+generations in parent→child order with a `depth` column.

    Implements the nesting the chapter describes for Figure 7.6: a
    ``models.chat`` span "might contain child spans for cache lookup,
    provider API call, and response parsing." Each item carries its
    nesting depth so the waterfall renders with indentation.
    """
    items: List[Dict[str, Any]] = []
    for span in spans:
        items.append(
            {
                "id": span.span_id,
                "parent": span.parent_span_id or "",
                "span": span,
                "kind": "span",
                "generation": None,
            }
        )
    for gen in generations:
        items.append(
            {
                "id": gen.span.span_id,
                "parent": gen.span.parent_span_id or "",
                "span": gen.span,
                "kind": "generation",
                "generation": gen,
            }
        )

    all_ids = {item["id"] for item in items}
    by_parent: Dict[str, List[Dict[str, Any]]] = {}
    for item in items:
        by_parent.setdefault(item["parent"], []).append(item)

    roots = [item for item in items if not item["parent"] or item["parent"] not in all_ids]
    roots.sort(key=lambda i: i["span"].start_time or datetime.max.replace(tzinfo=timezone.utc))

    ordered: List[Dict[str, Any]] = []

    def walk(node: Dict[str, Any], depth: int) -> None:
        node["depth"] = depth
        ordered.append(node)
        children = by_parent.get(node["id"], [])
        children.sort(
            key=lambda c: c["span"].start_time or datetime.max.replace(tzinfo=timezone.utc)
        )
        for child in children:
            walk(child, depth + 1)

    for root in roots:
        walk(root, 0)
    return ordered


# Span attribute keys we'll scan when the trace's input/output fields
# aren't populated by the producer. Order matters — first match wins.
_INPUT_KEYS = ("input", "user_message", "query", "question", "prompt")
_OUTPUT_KEYS = ("output", "response", "answer", "assistant_message")


def _extract_io(trace: Any) -> tuple[str, str]:
    """Best-effort: pull the trace's input + output for display.

    The book defines Trace as carrying "the user's input, the final output."
    The platform's services don't populate those fields yet, so we fall
    back to scanning span attributes for common keys. Returns ("", "") if
    nothing is found — the page prints a clarifying note in that case.
    """
    if trace.input:
        return trace.input, trace.output or ""
    for span in trace.spans:
        attrs = getattr(span, "attributes", {}) or {}
        i = next((attrs[k] for k in _INPUT_KEYS if k in attrs), "")
        o = next((attrs[k] for k in _OUTPUT_KEYS if k in attrs), "")
        if i or o:
            return i, o
    for gen in trace.generations:
        attrs = getattr(gen.span, "attributes", {}) or {}
        i = next((attrs[k] for k in _INPUT_KEYS if k in attrs), "")
        o = next((attrs[k] for k in _OUTPUT_KEYS if k in attrs), "")
        if i or o:
            return i, o
    return "", ""


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def page_traces() -> None:
    st.title("Traces")
    st.caption(
        "Single requests, span/generation waterfall, and attached scores. "
        "The primary unit of analysis when investigating a problem (Chapter 7, section 7.2)."
    )
    _glossary_expander("Traces")

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

    # Honour a "view this trace" pre-fill set by other pages.
    pre_selected = st.session_state.pop("_traces_focus_id", "")

    if not traces and not pre_selected:
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
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch")

    trace_id = st.selectbox(
        "Inspect a trace:",
        options=[""] + [r["trace_id"] for r in rows],
        format_func=lambda x: x or "—",
        index=([""] + [r["trace_id"] for r in rows]).index(pre_selected)
        if pre_selected in [r["trace_id"] for r in rows]
        else 0,
    )
    # Allow direct lookup by trace_id even when it isn't in the current list.
    typed = st.text_input("…or paste a trace_id directly:", value=pre_selected)
    if typed:
        trace_id = typed
    if not trace_id:
        return

    trace = platform.observability.get_trace(trace_id)
    if trace is None:
        st.warning(f"Trace {trace_id} not found")
        return

    _render_trace_detail(trace, key_prefix=f"t-{trace.trace_id}")


def _render_trace_detail(trace: Any, *, key_prefix: str) -> None:
    """Render the full detail view for one trace.

    Used by both ``page_traces`` (single-trace drill-down) and
    ``page_sessions`` (one per turn in the session drill-down). The
    ``key_prefix`` keeps Streamlit widget keys unique when the helper
    runs multiple times on a single page.
    """
    # ------------------------------------------------------------------
    # 1) Summary row
    # ------------------------------------------------------------------
    st.subheader(f"Trace {trace.trace_id}")
    cols = st.columns(5)
    cols[0].metric("Duration (ms)", f"{trace.total_duration_ms:,.1f}")
    cols[1].metric("Cost (USD)", f"${trace.total_cost_usd:,.4f}")
    cols[2].metric("Tokens", f"{trace.total_tokens:,}")
    cols[3].metric("Spans", len(trace.spans))
    cols[4].metric("Generations", len(trace.generations))
    meta_cols = st.columns(3)
    meta_cols[0].markdown(f"**workflow_id:** `{trace.workflow_id or '—'}`")
    meta_cols[1].markdown(f"**user_id:** `{trace.user_id or '—'}`")
    meta_cols[2].markdown(f"**session_id:** `{trace.session_id or '—'}`")

    # Services-touched chip row — colored pills, one per unique service in
    # this trace. A quick read of "which parts of the platform handled this
    # request" without scanning the waterfall.
    services_touched: List[str] = []
    for s in trace.spans:
        if s.service and s.service not in services_touched:
            services_touched.append(s.service)
    for g in trace.generations:
        if g.span.service and g.span.service not in services_touched:
            services_touched.append(g.span.service)
    if services_touched:
        pills = " ".join(
            (
                f'<span style="background:{_SERVICE_COLORS.get(svc, _SERVICE_COLOR_FALLBACK)};'
                f"color:white;padding:2px 8px;border-radius:9999px;"
                f'font-size:0.85em;margin-right:6px;">{svc}</span>'
            )
            for svc in services_touched
        )
        st.markdown(f"🌊 **Services touched:** {pills}", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # 2) Cross-page jump buttons (Figure 7.7: trace → logs → metrics)
    # ------------------------------------------------------------------
    jump_cols = st.columns(2)
    if jump_cols[0].button("📝 View logs for this trace", key=f"{key_prefix}-logs"):
        st.session_state["_logs_trace_id_filter"] = trace.trace_id
        _switch_to("Logs")
    if jump_cols[1].button("📈 Metrics around this window", key=f"{key_prefix}-metrics"):
        spans_start = [s.start_time for s in trace.spans if s.start_time]
        spans_end = [s.end_time for s in trace.spans if s.end_time]
        if spans_start and spans_end:
            window_hours = max(
                1,
                int((max(spans_end) - min(spans_start)).total_seconds() / 3600) + 1,
            )
            st.session_state["_metrics_hours_back"] = window_hours
        _switch_to("Metrics")

    # ------------------------------------------------------------------
    # 3) Input / output (Trace primitive definition, section 7.2)
    # ------------------------------------------------------------------
    user_input, model_output = _extract_io(trace)
    st.markdown("### Conversation")
    if user_input or model_output:
        io_cols = st.columns(2)
        with io_cols[0]:
            st.markdown("**User input**")
            st.write(user_input or "_(not recorded on this trace)_")
        with io_cols[1]:
            st.markdown("**Assistant output**")
            st.write(model_output or "_(not recorded on this trace)_")
    else:
        st.info(
            "This trace doesn't carry `input` / `output` fields. The platform's "
            "services don't populate them by default; instrument your workflow "
            "to set them as span attributes (`input`, `output`) for full visibility."
        )

    # ------------------------------------------------------------------
    # 4) Waterfall — the visualisation from Figure 7.6
    # ------------------------------------------------------------------
    st.markdown("### Waterfall")
    ordered = _build_span_tree(trace.spans, trace.generations)
    if ordered:
        waterfall_rows = []
        for item in ordered:
            span = item["span"]
            indent = "│ " * max(0, item["depth"] - 1) + ("└─ " if item["depth"] > 0 else "")
            label = f"{indent}{span.operation}"
            kind = (
                f"generation ({item['generation'].model})"
                if item["kind"] == "generation"
                else "span"
            )
            waterfall_rows.append(
                {
                    "Step": label,
                    "Service": span.service or "unknown",
                    "Kind": kind,
                    "Start": span.start_time,
                    "Finish": span.end_time or span.start_time,
                    "Duration (ms)": round(span.duration_ms, 1),
                    "Status": span.status,
                    "IsGeneration": item["kind"] == "generation",
                }
            )
        df_w = pd.DataFrame(waterfall_rows)
        # Color by Service so the same service is always the same colour
        # across every trace (gateway gray, data blue, etc.).
        services_in_trace = list(df_w["Service"].unique())
        color_map = {
            svc: _SERVICE_COLORS.get(svc, _SERVICE_COLOR_FALLBACK) for svc in services_in_trace
        }
        fig = px.timeline(
            df_w,
            x_start="Start",
            x_end="Finish",
            y="Step",
            color="Service",
            hover_data=["Kind", "Duration (ms)", "Status"],
            color_discrete_map=color_map,
        )
        # Per-bar styling: generations get a thicker border to stand out;
        # ERROR spans get a red border so failures pop visually.
        for i, row in enumerate(waterfall_rows):
            border_color = (
                "#dc2626"
                if row["Status"] == "ERROR"
                else ("#1f2937" if row["IsGeneration"] else None)
            )
            if border_color:
                fig.data[0].marker.line.color = border_color  # default for trace 0
        # Apply per-bar borders by iterating over the traces produced by
        # plotly (one trace per Service legend entry).
        # Transparent fallback so plotly accepts the list (it rejects None
        # entries). Bars that don't need a visible border get this colour
        # paired with width=0, so it never actually renders.
        invisible = "rgba(0,0,0,0)"
        for fig_trace in fig.data:
            svc = fig_trace.name
            # `fig_trace.y` holds the Step labels assigned to this colour.
            labels = list(fig_trace.y)
            line_colors: List[str] = []
            line_widths: List[int] = []
            for label in labels:
                match = next(
                    (r for r in waterfall_rows if r["Step"] == label and r["Service"] == svc),
                    None,
                )
                if match is None:
                    line_colors.append(invisible)
                    line_widths.append(0)
                    continue
                if match["Status"] == "ERROR":
                    line_colors.append("#dc2626")
                    line_widths.append(3)
                elif match["IsGeneration"]:
                    line_colors.append("#1f2937")
                    line_widths.append(2)
                else:
                    line_colors.append(invisible)
                    line_widths.append(0)
            fig_trace.marker.line.color = line_colors
            fig_trace.marker.line.width = line_widths
        fig.update_yaxes(autorange="reversed", title=None)
        fig.update_layout(height=max(160, 32 * len(df_w) + 80), margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, width="stretch", key=f"{key_prefix}-waterfall")
        st.caption(
            "Each bar is one span or generation. Color = service. A thick "
            "dark border marks the LLM **generation**. A red border marks "
            "an **ERROR** span."
        )
    else:
        st.info("No spans recorded on this trace.")

    # Cost split across services — visually proves the chapter's claim
    # that the generation is where the money goes.
    cost_rows: List[Dict[str, Any]] = []
    by_service: Dict[str, float] = {}
    for g in trace.generations:
        svc = g.span.service or "models"
        by_service[svc] = by_service.get(svc, 0.0) + (g.cost_usd or 0.0)
    # Non-generation spans don't carry cost; include them with $0 so the
    # bar chart visibly shows the disparity.
    for s in trace.spans:
        by_service.setdefault(s.service or "unknown", 0.0)
    for svc, cost in sorted(by_service.items(), key=lambda kv: -kv[1]):
        cost_rows.append({"service": svc, "cost_usd": round(cost, 6)})
    if cost_rows and any(r["cost_usd"] > 0 for r in cost_rows):
        st.markdown("### Cost split across services")
        df_cost = pd.DataFrame(cost_rows)
        fig_cost = px.bar(
            df_cost,
            x="cost_usd",
            y="service",
            orientation="h",
            color="service",
            color_discrete_map={
                row["service"]: _SERVICE_COLORS.get(row["service"], _SERVICE_COLOR_FALLBACK)
                for row in cost_rows
            },
        )
        fig_cost.update_layout(
            height=max(120, 28 * len(cost_rows) + 60),
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
            yaxis=dict(autorange="reversed", title=None),
            xaxis=dict(title="cost (USD)"),
        )
        st.plotly_chart(fig_cost, width="stretch", key=f"{key_prefix}-cost-split")

    # ------------------------------------------------------------------
    # 5) Generations — the LLM-specific fields the book calls out
    # ------------------------------------------------------------------
    if trace.generations:
        st.markdown("### Generations")
        st.caption(
            "Listing 7.5: generations are specialized spans for LLM calls. "
            "Indexed independently so cost / model / token queries are cheap."
        )
        gen_rows = []
        for gen in trace.generations:
            gen_rows.append(
                {
                    "model": gen.model,
                    "provider": gen.provider,
                    "prompt_tokens": gen.prompt_tokens,
                    "completion_tokens": gen.completion_tokens,
                    "cost_usd": round(gen.cost_usd, 6),
                    "ttft_ms": round(gen.time_to_first_token_ms, 1),
                    "cache_hit": gen.cache_hit,
                    "fallback_used": gen.fallback_used,
                    "duration_ms": round(gen.span.duration_ms, 1),
                    "status": gen.span.status,
                }
            )
        st.dataframe(pd.DataFrame(gen_rows), width="stretch")

    # ------------------------------------------------------------------
    # 6) Scores
    # ------------------------------------------------------------------
    if trace.scores:
        st.markdown("### Scores")
        st.caption(
            "Quality signals attached to this trace. The `description` column "
            "explains what each score measures: prefer the score's own "
            "`comment` if set; else fall back to a built-in rubric; else "
            'show "no description registered".'
        )
        score_rows = []
        for s in trace.scores:
            description = (
                s.comment
                or _SCORE_DESCRIPTIONS.get(s.name)
                or "no description registered — set `comment` on `record_score()`"
            )
            # Coerce value to str so the column has a single dtype.
            # Pandas/pyarrow can't render a mixed-type column (0.85 + 'correct'
            # + 0.82) and Streamlit's st.dataframe crashes when it tries.
            if isinstance(s.value, bool):
                value_display = "true" if s.value else "false"
            elif isinstance(s.value, (int, float)):
                value_display = f"{float(s.value):.4g}"
            else:
                value_display = str(s.value or "")
            score_rows.append(
                {
                    "name": s.name,
                    "value": value_display,
                    "source": s.source,
                    "description": description,
                    "span_id": s.span_id,
                    "generation_id": s.generation_id,
                }
            )
        st.dataframe(pd.DataFrame(score_rows), width="stretch")
        with st.expander("ℹ️ What do the source values mean?", expanded=False):
            st.markdown(_source_legend_markdown())


def page_cost() -> None:
    st.title("Cost")
    st.caption("Listing 7.13 — drill into cost by team, workflow, or model.")
    _glossary_expander("Cost")

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
    st.caption(
        "Percentile time series for any platform metric. Section 7.4.2 calls "
        "out histograms over averages because AI latency is heavy-tailed."
    )
    _glossary_expander("Metrics")

    platform = get_platform()
    # Cross-page link from Traces may pre-fill the lookback.
    default_hours = int(st.session_state.pop("_metrics_hours_back", 24) or 24)
    with st.sidebar:
        st.markdown("**Common metric names** (Listing 7.4):")
        st.code(
            "ai.platform.models.request_duration_ms\n"
            "ai.platform.models.requests_total\n"
            "ai.platform.models.cost_usd\n"
            "ai.platform.models.fallbacks_total\n"
            "ai.platform.data.search_duration_ms\n"
            "ai.platform.data.relevance_score\n"
            "ai.platform.guardrails.evaluation_duration_ms",
            language=None,
        )
        metric_name = st.text_input("Metric name", value="ai.platform.models.request_duration_ms")
        hours_back = st.slider("Lookback (hours)", min_value=1, max_value=168, value=default_hours)
        buckets = st.slider("Buckets", min_value=6, max_value=96, value=24)

    start, end = _utc_window(hours_back)

    # ------------------------------------------------------------------
    # Time series (one query per bucket — simple and good enough for the
    # demo dataset; production would push this into the query layer).
    # ------------------------------------------------------------------
    bucket_size = (end - start) / max(1, buckets)
    series_rows = []
    for i in range(buckets):
        b_start = start + bucket_size * i
        b_end = start + bucket_size * (i + 1)
        result = platform.observability.query_metrics(
            name=metric_name, aggregation="p95", start_time=b_start, end_time=b_end
        )
        p50 = platform.observability.query_metrics(
            name=metric_name, aggregation="p50", start_time=b_start, end_time=b_end
        )
        p99 = platform.observability.query_metrics(
            name=metric_name, aggregation="p99", start_time=b_start, end_time=b_end
        )
        series_rows.append(
            {
                "bucket": b_start,
                "p50": p50.get("value", 0.0) or None,
                "p95": result.get("value", 0.0) or None,
                "p99": p99.get("value", 0.0) or None,
                "samples": result.get("sample_count", 0),
            }
        )
    series_df = pd.DataFrame(series_rows)

    if series_df["samples"].sum() == 0:
        st.info(f"No samples for `{metric_name}` in the last {hours_back}h. Try a different name.")
        return

    plot_df = series_df.melt(
        id_vars="bucket",
        value_vars=["p50", "p95", "p99"],
        var_name="percentile",
        value_name="value",
    ).dropna(subset=["value"])
    fig = px.line(
        plot_df,
        x="bucket",
        y="value",
        color="percentile",
        markers=True,
        title=f"{metric_name} — p50 / p95 / p99",
    )
    fig.update_layout(yaxis_title="value", xaxis_title="time", height=380)
    st.plotly_chart(fig, width="stretch")

    # ------------------------------------------------------------------
    # Headline aggregates for the whole window — kept so the page is
    # useful at a glance.
    # ------------------------------------------------------------------
    cols = st.columns(4)
    overall = {
        agg: platform.observability.query_metrics(
            name=metric_name, aggregation=agg, start_time=start, end_time=end
        ).get("value", 0.0)
        for agg in ("p50", "p95", "p99", "count")
    }
    cols[0].metric("p50", f"{overall['p50']:,.2f}")
    cols[1].metric("p95", f"{overall['p95']:,.2f}")
    cols[2].metric("p99", f"{overall['p99']:,.2f}")
    cols[3].metric("Samples", int(overall["count"]))


def page_service_health() -> None:
    st.title("Service Health")
    st.caption(
        "Derived from span error rates the Observability Service has "
        "observed — not a liveness probe. A service with no spans in "
        "the window shows up as `unknown`, not `healthy`."
    )
    _glossary_expander("Service Health")

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


def page_sessions() -> None:
    st.title("Sessions")
    st.caption(
        "Multi-turn conversations grouped by session_id. The Observability "
        "Service doesn't manage session lifecycle (that's the Session Service "
        "from Chapter 4); this page just groups traces by their session_id "
        "field — Chapter 7, section 7.3."
    )
    _glossary_expander("Sessions")

    platform = get_platform()
    with st.sidebar:
        hours_back = st.slider("Lookback (hours)", min_value=1, max_value=168, value=24)
        user_id = st.text_input("Filter by user_id")
        limit = st.number_input("Trace limit", min_value=10, max_value=2000, value=500)

    start_time, end_time = _utc_window(hours_back)
    traces = platform.observability.query_traces(
        start_time=start_time,
        end_time=end_time,
        user_id=user_id or "",
        limit=int(limit),
    )

    if not traces:
        st.info("No traces in the lookback window.")
        return

    # Bucket traces by session_id (traces without a session_id are dropped
    # from this view — they're visible on the Traces page instead).
    sessions: Dict[str, List[Any]] = {}
    for trace in traces:
        if not trace.session_id:
            continue
        sessions.setdefault(trace.session_id, []).append(trace)

    if not sessions:
        st.info(
            "No traces in the window carried a session_id. "
            "Sessions show up here once a workflow propagates one "
            "(e.g. the Claw assistant via platform.sessions.get_or_create)."
        )
        return

    rows = []
    for session_id, session_traces in sessions.items():
        starts = [
            min(
                (s.start_time for s in t.spans if s.start_time),
                default=None,
            )
            for t in session_traces
        ]
        ends = [
            max(
                (s.end_time for s in t.spans if s.end_time),
                default=None,
            )
            for t in session_traces
        ]
        starts = [s for s in starts if s is not None]
        ends = [e for e in ends if e is not None]
        rows.append(
            {
                "session_id": session_id,
                "turns": len(session_traces),
                "first_trace_at": min(starts) if starts else None,
                "last_trace_at": max(ends) if ends else None,
                "total_cost_usd": round(sum(t.total_cost_usd for t in session_traces), 4),
                "total_tokens": sum(t.total_tokens for t in session_traces),
                "spans": sum(len(t.spans) for t in session_traces),
                "generations": sum(len(t.generations) for t in session_traces),
                "scores": sum(len(t.scores) for t in session_traces),
                "user_id": next((t.user_id for t in session_traces if t.user_id), ""),
                "workflow_id": next((t.workflow_id for t in session_traces if t.workflow_id), ""),
            }
        )

    df = pd.DataFrame(rows).sort_values("last_trace_at", ascending=False)
    st.dataframe(df, width="stretch")

    chosen = st.selectbox(
        "Drill into a session:",
        options=[""] + df["session_id"].tolist(),
        format_func=lambda x: x or "—",
    )
    if not chosen:
        return

    st.subheader(f"Session {chosen}")
    far_future = datetime.max.replace(tzinfo=timezone.utc)
    session_traces = sorted(
        sessions[chosen],
        key=lambda t: min((s.start_time for s in t.spans if s.start_time), default=far_future),
    )

    # Summary table of all traces in the session.
    st.caption(
        "Each row is one **trace** — a single user request and the "
        "assistant's response. A multi-turn conversation produces one "
        "trace per turn. Expand a trace below to see the full waterfall "
        "of services that handled that request."
    )
    summary_rows = []
    for i, trace in enumerate(session_traces, start=1):
        span_starts = [s.start_time for s in trace.spans if s.start_time]
        summary_rows.append(
            {
                "#": i,
                "trace_id": trace.trace_id,
                "started_at": min(span_starts) if span_starts else None,
                "duration_ms": round(trace.total_duration_ms, 1),
                "cost_usd": round(trace.total_cost_usd, 4),
                "tokens": trace.total_tokens,
                "spans": len(trace.spans),
                "generations": len(trace.generations),
                "scores": ", ".join(f"{s.name}={s.value}" for s in trace.scores),
            }
        )
    st.dataframe(pd.DataFrame(summary_rows), width="stretch")

    # Per-trace expanders. First one open by default so the dashboard
    # immediately surfaces the conversation's first turn.
    st.markdown("---")
    st.markdown("### Per-trace detail")
    for i, trace in enumerate(session_traces, start=1):
        short_id = trace.trace_id[:8]
        header = (
            f"Trace #{i} — {short_id} — "
            f"{trace.total_duration_ms:,.0f} ms — ${trace.total_cost_usd:,.4f}"
        )
        with st.expander(header, expanded=(i == 1)):
            _render_trace_detail(trace, key_prefix=f"s-{chosen}-{trace.trace_id}")


def page_logs() -> None:
    st.title("Logs")
    st.caption("Structured-log search by trace_id, service, or event_type.")
    _glossary_expander("Logs")

    platform = get_platform()
    # Cross-page link from Traces sets this so the filter pre-fills.
    default_trace = st.session_state.pop("_logs_trace_id_filter", "")
    with st.sidebar:
        trace_id = st.text_input("Trace ID", value=default_trace)
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
    log_rows = []
    for evt in events:
        log_rows.append(
            {
                "timestamp": getattr(evt, "timestamp", None),
                "severity": getattr(evt, "severity", ""),
                "service": getattr(evt, "service", ""),
                "event_type": getattr(evt, "event_type", ""),
                "message": getattr(evt, "message", ""),
                "trace_id": getattr(evt, "trace_id", ""),
                "span_id": getattr(evt, "span_id", ""),
            }
        )
    st.dataframe(pd.DataFrame(log_rows), width="stretch")

    # Round-trip back to the trace view (Figure 7.7 says these are linked).
    if trace_id:
        if st.button("🌊 Back to this trace", key=f"back-{trace_id}"):
            st.session_state["_traces_focus_id"] = trace_id
            _switch_to("Traces")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="GenAI Platform — Observability", layout="wide")
    # Build StreamlitPage objects once and cache on the module so the
    # cross-page jump buttons (st.switch_page) can pass the page object
    # directly — switch_page rejects bare title strings under st.navigation.
    pages = {
        "Sessions": st.Page(page_sessions, title="Sessions", icon="🧵"),
        "Traces": st.Page(page_traces, title="Traces", icon="🌊"),
        "Cost": st.Page(page_cost, title="Cost", icon="💸"),
        "Metrics": st.Page(page_metrics, title="Metrics", icon="📈"),
        "Service Health": st.Page(page_service_health, title="Service Health", icon="❤️"),
        "Logs": st.Page(page_logs, title="Logs", icon="📝"),
    }
    st.session_state["_pages"] = pages
    pg = st.navigation(list(pages.values()))
    pg.run()


def _switch_to(page_name: str) -> None:
    """Switch to a page by name, using the StreamlitPage object that ``main``
    cached. Streamlit's ``switch_page`` doesn't accept bare title strings
    when pages are declared via ``st.navigation``; only the Page object or
    a file path works."""
    pages = st.session_state.get("_pages") or {}
    target = pages.get(page_name)
    if target is None:
        return
    st.switch_page(target)


if __name__ == "__main__":
    main()
