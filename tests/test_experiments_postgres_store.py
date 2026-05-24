"""Integration smoke test for the Postgres-backed experiments store.

Skipped automatically unless a Postgres DSN is reachable. Two
environment variables are honoured:

- ``EXPERIMENTS_POSTGRES_DSN`` — local development.
- ``DB_TEST_URL`` — the env var CI sets when it spins up the pgvector
  service container.

Mirrors the observability-store integration test shape so both
services follow the same testing pattern.
"""

from __future__ import annotations

import os
import uuid

import pytest

from services.experiments.models import (
    AnnotationItem,
    AnnotationQueue,
    Dataset,
    EvaluationSummary,
    Experiment,
    ExperimentTarget,
    RubricItem,
    ScoringRule,
    TestCase,
    Variant,
)

psycopg2 = pytest.importorskip("psycopg2", reason="psycopg2 not installed")


def _dsn() -> str | None:
    return os.environ.get("EXPERIMENTS_POSTGRES_DSN") or os.environ.get("DB_TEST_URL")


def _can_connect(dsn: str) -> bool:
    try:
        conn = psycopg2.connect(dsn)
        conn.close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _dsn() or not _can_connect(_dsn() or ""),
    reason="EXPERIMENTS_POSTGRES_DSN / DB_TEST_URL unset or Postgres unreachable",
)


@pytest.fixture
def store():
    from services.experiments.postgres_store import PostgresExperimentStore

    s = PostgresExperimentStore(connection_string=_dsn())
    with s.conn.cursor() as cur:
        # Clean slate per test run. The order matters for FK constraints.
        cur.execute(
            "TRUNCATE annotation_items, annotation_queues, outcomes, variant_assignments, "
            "experiments, scoring_rules, evaluation_results, evaluations, test_cases, "
            "datasets, experiment_targets RESTART IDENTITY CASCADE"
        )
        s.conn.commit()
    yield s
    s.conn.close()


class TestPostgresExperimentStore:
    def test_target_lifecycle_with_auto_deprecate(self, store):
        name = f"prompt-{uuid.uuid4().hex[:6]}"
        store.register_target(ExperimentTarget(name=name, version=1))
        store.register_target(ExperimentTarget(name=name, version=2))

        # Promote v1 → ACTIVE.
        v1 = store.update_target_status(name, 1, "ACTIVE")
        assert v1.status == "ACTIVE"
        # Promote v2 → ACTIVE; v1 must auto-deprecate.
        v2 = store.update_target_status(name, 2, "ACTIVE")
        assert v2.status == "ACTIVE"
        history = store.get_target_history(name)
        statuses = {t.version: t.status for t in history}
        assert statuses[1] == "DEPRECATED"
        assert statuses[2] == "ACTIVE"

    def test_dataset_create_and_add(self, store):
        ds_name = f"ds-{uuid.uuid4().hex[:6]}"
        store.create_dataset(Dataset(name=ds_name))
        store.add_test_cases(
            ds_name,
            [TestCase(input_query="q1"), TestCase(input_query="q2")],
        )
        dataset = store.get_dataset(ds_name)
        assert dataset is not None
        assert len(dataset.test_cases) == 2

    def test_experiment_assign_and_record(self, store):
        exp_name = f"exp-{uuid.uuid4().hex[:6]}"
        store.save_experiment(
            Experiment(
                name=exp_name,
                workflow_id="wf",
                variants=[
                    Variant(name="control", traffic_allocation=0.5),
                    Variant(name="treat", traffic_allocation=0.5),
                ],
                success_metrics=["score"],
                minimum_sample_size=10,
                status="running",
            )
        )
        loaded = store.get_experiment(exp_name)
        assert loaded is not None
        assert {v.name for v in loaded.variants} == {"control", "treat"}

    def test_scoring_rule_round_trip(self, store):
        store.save_scoring_rule(ScoringRule(name="r-1", workflow_id="wf", sample_rate=0.5))
        rule = store.get_scoring_rule("r-1")
        assert rule is not None
        assert rule.sample_rate == 0.5

    def test_annotation_lock_to_reviewer(self, store):
        queue_name = f"q-{uuid.uuid4().hex[:6]}"
        store.save_annotation_queue(
            AnnotationQueue(name=queue_name, rubric=[RubricItem(name="x", type="numeric")])
        )
        store.add_annotation_item(AnnotationItem(queue_name=queue_name, trace_id="t1"))
        store.add_annotation_item(AnnotationItem(queue_name=queue_name, trace_id="t2"))

        first = store.get_next_annotation_item(queue_name, reviewer="alice")
        second = store.get_next_annotation_item(queue_name, reviewer="bob")
        assert first.trace_id == "t1"
        assert second.trace_id == "t2"

    def test_summary_round_trip(self, store):
        name = f"p-{uuid.uuid4().hex[:6]}"
        store.register_target(ExperimentTarget(name=name, version=1))
        store.update_target_summary(
            name,
            1,
            EvaluationSummary(overall_score=0.91, metric_scores={"x": 0.9}),
        )
        loaded = store.get_target(name, 1)
        assert loaded is not None
        assert loaded.evaluation_summary.overall_score == pytest.approx(0.91)
