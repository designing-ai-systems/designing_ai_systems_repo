"""
Streamlit dashboard over the Experimentation Service.

Reads through the platform SDK. Five pages cover the chapter's
improvement loop:
  - Targets: lifecycle list + Promote action (Listing 7.15)
  - Datasets: hand-curated + add-from-production (Listing 7.17)
  - Evaluations: run streaming eval + view results (Listing 7.16)
  - A/B Tests: variants, p-values, sample sizes (Listing 7.20)
  - Annotations: review queue, submit rubric scores (Listing 7.19)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from genai_platform import GenAIPlatform

_TARGET_STATUS = {0: "UNSPECIFIED", 1: "DRAFT", 2: "TESTING", 3: "ACTIVE", 4: "DEPRECATED"}
_TARGET_TYPE = {0: "UNSPECIFIED", 1: "PROMPT", 2: "MODEL_CONFIG", 3: "RETRIEVAL_CONFIG"}


def get_platform() -> GenAIPlatform:
    cache = st.session_state.setdefault("_genai_platform_cache", {})
    url = os.environ.get("GENAI_GATEWAY_URL", "localhost:50051")
    if cache.get("url") != url:
        cache.clear()
        cache["url"] = url
        cache["platform"] = GenAIPlatform(gateway_url=url)
    return cache["platform"]


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def page_targets() -> None:
    st.title("Targets")
    st.caption("Versioned prompts / configs and their lifecycle (Listing 7.15).")

    platform = get_platform()
    name = st.text_input("Target name", value=st.session_state.get("_target_name", ""))
    st.session_state["_target_name"] = name
    if not name:
        st.info("Enter a target name on the left to view its history.")
        return

    history = platform.experiments.get_target_history(name)
    versions = history.versions
    if not versions:
        st.info(f"No versions registered for target '{name}'.")
        return

    rows: List[Dict[str, Any]] = []
    for target in versions:
        summary = target.evaluation_summary
        rows.append(
            {
                "name": target.name,
                "version": target.version,
                "type": _TARGET_TYPE.get(target.type, str(target.type)),
                "status": _TARGET_STATUS.get(target.status, str(target.status)),
                "overall_score": round(summary.overall_score, 3),
                "change_description": target.change_description,
                "author": target.author,
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch")

    promote_version = st.number_input("Promote version to ACTIVE", min_value=0, value=0, step=1)
    if st.button("Promote", disabled=promote_version == 0):
        result = platform.experiments.promote_target(name=name, version=int(promote_version))
        st.success(
            f"Promoted {result.name}:{result.version} → "
            f"{_TARGET_STATUS.get(result.status, str(result.status))}"
        )
        st.rerun()


def page_datasets() -> None:
    st.title("Datasets")
    st.caption("Curated test cases + add-from-production (Listing 7.17).")

    platform = get_platform()
    with st.form("create_dataset"):
        new_name = st.text_input("Create a new dataset")
        submitted = st.form_submit_button("Create")
        if submitted and new_name:
            platform.experiments.create_dataset(name=new_name)
            st.success(f"Created dataset {new_name}.")

    dataset_name = st.text_input("View dataset", value=st.session_state.get("_dataset_name", ""))
    st.session_state["_dataset_name"] = dataset_name
    if not dataset_name:
        return

    response = platform.experiments.add_from_production(
        dataset_name=dataset_name,
        trace_ids=[],  # no-op fetch; we use it to read the current dataset state.
        require_human_review=False,
    )
    rows: List[Dict[str, Any]] = []
    for case in response.test_cases:
        rows.append(
            {
                "id": case.id,
                "input_query": case.input_query[:80] + ("…" if len(case.input_query) > 80 else ""),
                "tags": list(case.tags),
                "source_trace_id": case.source_trace_id,
                "needs_review": case.needs_review,
            }
        )
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch")
    else:
        st.info("Dataset is empty.")

    with st.form("add_from_production"):
        st.subheader("Add traces from production")
        trace_ids_raw = st.text_area("Trace IDs (one per line)")
        require_review = st.checkbox("Require human review", value=True)
        if st.form_submit_button("Add"):
            trace_ids = [t.strip() for t in trace_ids_raw.splitlines() if t.strip()]
            if trace_ids:
                platform.experiments.add_from_production(
                    dataset_name=dataset_name,
                    trace_ids=trace_ids,
                    require_human_review=require_review,
                )
                st.success(f"Added {len(trace_ids)} trace(s).")
                st.rerun()


def page_evaluations() -> None:
    st.title("Evaluations")
    st.caption("Offline evaluation: targets vs. dataset (Listing 7.16).")

    platform = get_platform()
    with st.form("run_eval"):
        dataset_name = st.text_input("Dataset name")
        target_ids_raw = st.text_input("Target IDs (comma-separated, e.g. 'p:1,p:2')")
        metrics_raw = st.text_input("Metrics (comma-separated)", value="key_elements")
        repeats = st.number_input("Repeats per case", min_value=1, max_value=10, value=1)
        if st.form_submit_button("Run evaluation"):
            target_ids = [t.strip() for t in target_ids_raw.split(",") if t.strip()]
            metrics = [m.strip() for m in metrics_raw.split(",") if m.strip()]
            placeholder = st.empty()
            last_progress = None
            for progress in platform.experiments.run_evaluation(
                dataset_name=dataset_name,
                targets=target_ids,
                metrics=metrics,
                repeats_per_case=int(repeats),
            ):
                placeholder.write(
                    f"[{progress.status}] {progress.completed_cases}/{progress.total_cases}"
                    f" — current target: {progress.current_target or '—'}"
                )
                last_progress = progress
            if last_progress and last_progress.results.target_results:
                rows = []
                for tr in last_progress.results.target_results:
                    row = {
                        "target_id": tr.target_id,
                        "overall_score": round(tr.overall_score, 3),
                        "cases_run": tr.cases_run,
                    }
                    row.update({k: round(v, 3) for k, v in tr.metric_scores.items()})
                    rows.append(row)
                st.dataframe(pd.DataFrame(rows), width="stretch")


def page_ab_tests() -> None:
    st.title("A/B Tests")
    st.caption("Variant results with statistical significance (Listing 7.20).")

    platform = get_platform()
    experiment_name = st.text_input("Experiment name")
    if not experiment_name:
        return

    results = platform.experiments.get_experiment_results(experiment_name)
    if results.status == "not_found":
        st.warning(f"Experiment '{experiment_name}' not found.")
        return

    st.metric("Minimum sample size", results.minimum_sample_size)
    st.metric("Ready to conclude?", "yes" if results.ready_to_conclude else "no")

    variant_rows = []
    for vs in results.variant_summaries:
        row = {"variant": vs.variant_name, "samples": vs.sample_size}
        for metric in vs.metrics:
            row[f"{metric.metric_name}_mean"] = round(metric.mean, 4)
            row[f"{metric.metric_name}_stddev"] = round(metric.std_dev, 4)
        variant_rows.append(row)
    if variant_rows:
        st.subheader("Variant summaries")
        st.dataframe(pd.DataFrame(variant_rows), width="stretch")

    comp_rows = []
    for c in results.comparisons:
        comp_rows.append(
            {
                "metric": c.metric_name,
                "winner": c.winner,
                "effect_size": round(c.effect_size, 4),
                "p_value": round(c.p_value, 4),
                "significant": c.is_significant,
            }
        )
    if comp_rows:
        st.subheader("Comparisons")
        st.dataframe(pd.DataFrame(comp_rows), width="stretch")


def page_annotations() -> None:
    st.title("Annotation Queue")
    st.caption("Review pending traces, submit rubric scores (Listing 7.19).")

    platform = get_platform()
    queue_name = st.text_input("Queue name")
    reviewer = st.text_input("Reviewer email")
    if not queue_name or not reviewer:
        st.info("Provide a queue name and your reviewer email to pick up items.")
        return

    if st.button("Get next item"):
        item = platform.experiments.get_next_annotation_item(
            queue_name=queue_name, reviewer=reviewer
        )
        if not item.item_id:
            st.warning("No pending items in this queue.")
            return
        st.session_state["_current_item"] = {
            "item_id": item.item_id,
            "trace_id": item.trace_id,
            "queue_name": item.queue_name,
        }

    current = st.session_state.get("_current_item")
    if not current:
        return

    st.markdown(f"**Item:** `{current['item_id']}` — trace `{current['trace_id']}`")
    with st.form("submit_annotation"):
        helpfulness = st.slider("helpfulness", 0.0, 1.0, 0.5)
        correctness = st.selectbox(
            "correctness",
            options=["correct", "partially_correct", "incorrect"],
            index=0,
        )
        add_to_dataset = st.checkbox("Add this to the evaluation dataset")
        comment = st.text_area("Comments")
        if st.form_submit_button("Submit"):
            platform.experiments.submit_annotation(
                item_id=current["item_id"],
                queue_name=current["queue_name"],
                reviewer=reviewer,
                numeric_values={"helpfulness": helpfulness},
                categorical_values={"correctness": correctness},
                boolean_values={"add_to_dataset": add_to_dataset},
                comment=comment,
            )
            st.success("Annotation submitted.")
            st.session_state.pop("_current_item", None)


def main() -> None:
    st.set_page_config(page_title="GenAI Platform — Experimentation", layout="wide")
    pg = st.navigation(
        [
            st.Page(page_targets, title="Targets", icon="🎯"),
            st.Page(page_datasets, title="Datasets", icon="📚"),
            st.Page(page_evaluations, title="Evaluations", icon="🧪"),
            st.Page(page_ab_tests, title="A/B Tests", icon="🅰️"),
            st.Page(page_annotations, title="Annotations", icon="✍️"),
        ]
    )
    pg.run()


if __name__ == "__main__":
    main()
