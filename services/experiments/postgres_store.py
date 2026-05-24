"""
PostgreSQL-backed ExperimentStore.

Mirrors the in-memory store's interface so the servicer is store-agnostic.
Each table maps onto one experimentation primitive (targets, datasets,
test cases, evaluations, scoring rules, experiments, assignments,
outcomes, annotation queues, annotation items).

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 7.14: Experimentation Service contract
  - Listing 7.15: ExperimentTarget lifecycle persistence
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import List, Optional

import psycopg2
import psycopg2.extras

from services.experiments.models import (
    AlertCondition,
    AnnotationItem,
    AnnotationQueue,
    Dataset,
    Evaluation,
    EvaluationResults,
    EvaluationSummary,
    Experiment,
    ExperimentTarget,
    ModelConfigVariant,
    Outcome,
    PerCaseResult,
    PromptVariant,
    RetrievalConfigVariant,
    RoutingRule,
    RubricItem,
    ScorerConfig,
    ScoringRule,
    TargetResult,
    TestCase,
    Variant,
    VariantAssignment,
)
from services.experiments.store import ExperimentStore


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PostgresExperimentStore(ExperimentStore):
    """PostgreSQL implementation of ExperimentStore.

    Selection is driven by the ``EXPERIMENTS_POSTGRES_DSN`` env var in
    ``services.experiments.main``. Tests skip when no DSN is set so the
    unit-test suite stays Postgres-free.
    """

    def __init__(self, connection_string: Optional[str] = None) -> None:
        if not connection_string:
            connection_string = os.getenv(
                "EXPERIMENTS_POSTGRES_DSN",
                "postgresql://localhost/genai_platform",
            )
        self.conn = psycopg2.connect(
            connection_string,
            cursor_factory=psycopg2.extras.RealDictCursor,
        )
        # Autocommit so reads release their locks immediately instead of
        # holding "idle in transaction" connections that block writes /
        # TRUNCATEs. Explicit ``self.conn.commit()`` calls below become
        # no-ops but stay for documentation. See the matching comment in
        # services/observability/postgres_store.py.
        self.conn.autocommit = True
        self._create_tables()

    def _create_tables(self) -> None:
        schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
        with open(schema_path) as f:
            sql = f.read()
        with self.conn.cursor() as cur:
            cur.execute(sql)
        self.conn.commit()

    # ------------------------------------------------------------------
    # Targets
    # ------------------------------------------------------------------
    def register_target(self, target: ExperimentTarget) -> ExperimentTarget:
        if target.created_at is None:
            target.created_at = _utcnow()
        with self.conn.cursor() as cur:
            cur.execute(
                """INSERT INTO experiment_targets (
                       name, version, type, author, change_description,
                       created_at, status, evaluation_summary, metadata)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
                   ON CONFLICT (name, version) DO UPDATE SET
                       type = EXCLUDED.type,
                       author = EXCLUDED.author,
                       change_description = EXCLUDED.change_description,
                       metadata = EXCLUDED.metadata""",
                (
                    target.name,
                    target.version,
                    target.type,
                    target.author,
                    target.change_description,
                    target.created_at,
                    target.status,
                    json.dumps(self._summary_to_dict(target.evaluation_summary)),
                    json.dumps(target.metadata),
                ),
            )
            self.conn.commit()
        return target

    def get_target_history(self, name: str) -> List[ExperimentTarget]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM experiment_targets WHERE name = %s ORDER BY version",
                (name,),
            )
            return [self._row_to_target(r) for r in cur.fetchall()]

    def get_target(self, name: str, version: int) -> Optional[ExperimentTarget]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM experiment_targets WHERE name = %s AND version = %s",
                (name, version),
            )
            row = cur.fetchone()
        return self._row_to_target(row) if row else None

    def update_target_status(self, name: str, version: int, status: str) -> ExperimentTarget:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM experiment_targets WHERE name = %s AND version = %s",
                (name, version),
            )
            row = cur.fetchone()
            if row is None:
                raise KeyError(f"Target {name}:{version} not registered")
            cur.execute(
                "UPDATE experiment_targets SET status = %s WHERE name = %s AND version = %s",
                (status, name, version),
            )
            if status == "ACTIVE":
                cur.execute(
                    """UPDATE experiment_targets
                          SET status = 'DEPRECATED'
                        WHERE name = %s AND version <> %s AND status = 'ACTIVE'""",
                    (name, version),
                )
            self.conn.commit()
            cur.execute(
                "SELECT * FROM experiment_targets WHERE name = %s AND version = %s",
                (name, version),
            )
            return self._row_to_target(cur.fetchone())

    def update_target_summary(
        self, name: str, version: int, summary: EvaluationSummary
    ) -> ExperimentTarget:
        with self.conn.cursor() as cur:
            cur.execute(
                """UPDATE experiment_targets
                      SET evaluation_summary = %s::jsonb
                    WHERE name = %s AND version = %s
                RETURNING *""",
                (json.dumps(self._summary_to_dict(summary)), name, version),
            )
            row = cur.fetchone()
            if row is None:
                raise KeyError(f"Target {name}:{version} not registered")
            self.conn.commit()
        return self._row_to_target(row)

    # ------------------------------------------------------------------
    # Datasets / test cases
    # ------------------------------------------------------------------
    def create_dataset(self, dataset: Dataset) -> Dataset:
        dataset.created_at = dataset.created_at or _utcnow()
        dataset.updated_at = dataset.updated_at or _utcnow()
        with self.conn.cursor() as cur:
            cur.execute(
                """INSERT INTO datasets (name, metadata, created_at, updated_at)
                   VALUES (%s, %s::jsonb, %s, %s)
                   ON CONFLICT (name) DO UPDATE SET metadata = EXCLUDED.metadata,
                       updated_at = EXCLUDED.updated_at""",
                (
                    dataset.name,
                    json.dumps(dataset.metadata),
                    dataset.created_at,
                    dataset.updated_at,
                ),
            )
            for case in dataset.test_cases:
                if not case.id:
                    case.id = uuid.uuid4().hex
                self._insert_test_case(cur, dataset.name, case)
            self.conn.commit()
        return dataset

    def add_test_cases(self, name: str, cases: List[TestCase]) -> Dataset:
        with self.conn.cursor() as cur:
            cur.execute("SELECT name FROM datasets WHERE name = %s", (name,))
            if cur.fetchone() is None:
                raise KeyError(f"Dataset '{name}' not found")
            for case in cases:
                if not case.id:
                    case.id = uuid.uuid4().hex
                self._insert_test_case(cur, name, case)
            cur.execute("UPDATE datasets SET updated_at = %s WHERE name = %s", (_utcnow(), name))
            self.conn.commit()
        return self.get_dataset(name)

    def get_dataset(self, name: str) -> Optional[Dataset]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM datasets WHERE name = %s", (name,))
            row = cur.fetchone()
            if row is None:
                return None
            cur.execute("SELECT * FROM test_cases WHERE dataset_name = %s", (name,))
            cases = [self._row_to_test_case(c) for c in cur.fetchall()]
        return Dataset(
            name=row["name"],
            metadata=dict(row["metadata"] or {}),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            test_cases=cases,
        )

    # ------------------------------------------------------------------
    # Evaluations
    # ------------------------------------------------------------------
    def save_evaluation(self, evaluation: Evaluation) -> Evaluation:
        if not evaluation.evaluation_id:
            evaluation.evaluation_id = uuid.uuid4().hex
        if not evaluation.created_at:
            evaluation.created_at = _utcnow()
        with self.conn.cursor() as cur:
            cur.execute(
                """INSERT INTO evaluations (
                       evaluation_id, dataset_name, target_ids, metrics, created_at, status)
                   VALUES (%s, %s, %s::jsonb, %s::jsonb, %s, %s)
                   ON CONFLICT (evaluation_id) DO UPDATE SET status = EXCLUDED.status""",
                (
                    evaluation.evaluation_id,
                    evaluation.dataset_name,
                    json.dumps(evaluation.target_ids),
                    json.dumps(evaluation.metrics),
                    evaluation.created_at,
                    evaluation.status,
                ),
            )
            self.conn.commit()
        return evaluation

    def get_evaluation(self, evaluation_id: str) -> Optional[Evaluation]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM evaluations WHERE evaluation_id = %s", (evaluation_id,))
            row = cur.fetchone()
        if row is None:
            return None
        return Evaluation(
            evaluation_id=row["evaluation_id"],
            dataset_name=row["dataset_name"],
            target_ids=list(row["target_ids"] or []),
            metrics=list(row["metrics"] or []),
            created_at=row["created_at"],
            status=row["status"],
        )

    def save_evaluation_results(self, results: EvaluationResults) -> EvaluationResults:
        with self.conn.cursor() as cur:
            cur.execute(
                """INSERT INTO evaluation_results (
                       evaluation_id, dataset_name, target_results, per_case_results)
                   VALUES (%s, %s, %s::jsonb, %s::jsonb)
                   ON CONFLICT (evaluation_id) DO UPDATE SET
                       target_results = EXCLUDED.target_results,
                       per_case_results = EXCLUDED.per_case_results""",
                (
                    results.evaluation_id,
                    results.dataset_name,
                    json.dumps(
                        [
                            {
                                "target_id": tr.target_id,
                                "metric_scores": tr.metric_scores,
                                "overall_score": tr.overall_score,
                                "cases_run": tr.cases_run,
                            }
                            for tr in results.target_results
                        ]
                    ),
                    json.dumps(
                        [
                            {
                                "target_id": p.target_id,
                                "test_case_id": p.test_case_id,
                                "output": p.output,
                                "metric_scores": p.metric_scores,
                            }
                            for p in results.per_case_results
                        ]
                    ),
                ),
            )
            self.conn.commit()
        return results

    def get_evaluation_results(self, evaluation_id: str) -> Optional[EvaluationResults]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM evaluation_results WHERE evaluation_id = %s", (evaluation_id,)
            )
            row = cur.fetchone()
        if row is None:
            return None
        return EvaluationResults(
            evaluation_id=row["evaluation_id"],
            dataset_name=row["dataset_name"],
            target_results=[
                TargetResult(
                    target_id=tr.get("target_id", ""),
                    metric_scores=dict(tr.get("metric_scores", {})),
                    overall_score=tr.get("overall_score", 0.0),
                    cases_run=tr.get("cases_run", 0),
                )
                for tr in (row["target_results"] or [])
            ],
            per_case_results=[
                PerCaseResult(
                    target_id=p.get("target_id", ""),
                    test_case_id=p.get("test_case_id", ""),
                    output=p.get("output", ""),
                    metric_scores=dict(p.get("metric_scores", {})),
                )
                for p in (row["per_case_results"] or [])
            ],
        )

    # ------------------------------------------------------------------
    # Scoring rules
    # ------------------------------------------------------------------
    def save_scoring_rule(self, rule: ScoringRule) -> ScoringRule:
        if not rule.created_at:
            rule.created_at = _utcnow()
        with self.conn.cursor() as cur:
            cur.execute(
                """INSERT INTO scoring_rules (
                       name, workflow_id, sample_rate, scorers, alerts, created_at)
                   VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s)
                   ON CONFLICT (name) DO UPDATE SET
                       workflow_id = EXCLUDED.workflow_id,
                       sample_rate = EXCLUDED.sample_rate,
                       scorers = EXCLUDED.scorers,
                       alerts = EXCLUDED.alerts""",
                (
                    rule.name,
                    rule.workflow_id,
                    rule.sample_rate,
                    json.dumps([self._scorer_to_dict(s) for s in rule.scorers]),
                    json.dumps([self._alert_to_dict(a) for a in rule.alerts]),
                    rule.created_at,
                ),
            )
            self.conn.commit()
        return rule

    def list_scoring_rules(self, workflow_id: str = "") -> List[ScoringRule]:
        with self.conn.cursor() as cur:
            if workflow_id:
                cur.execute("SELECT * FROM scoring_rules WHERE workflow_id = %s", (workflow_id,))
            else:
                cur.execute("SELECT * FROM scoring_rules")
            return [self._row_to_scoring_rule(r) for r in cur.fetchall()]

    def get_scoring_rule(self, name: str) -> Optional[ScoringRule]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM scoring_rules WHERE name = %s", (name,))
            row = cur.fetchone()
        return self._row_to_scoring_rule(row) if row else None

    # ------------------------------------------------------------------
    # Experiments + outcomes
    # ------------------------------------------------------------------
    def save_experiment(self, experiment: Experiment) -> Experiment:
        with self.conn.cursor() as cur:
            cur.execute(
                """INSERT INTO experiments (
                       name, workflow_id, variants, success_metrics,
                       minimum_sample_size, created_at, status)
                   VALUES (%s, %s, %s::jsonb, %s::jsonb, %s, %s, %s)
                   ON CONFLICT (name) DO UPDATE SET
                       workflow_id = EXCLUDED.workflow_id,
                       variants = EXCLUDED.variants,
                       success_metrics = EXCLUDED.success_metrics,
                       minimum_sample_size = EXCLUDED.minimum_sample_size,
                       status = EXCLUDED.status""",
                (
                    experiment.name,
                    experiment.workflow_id,
                    json.dumps([self._variant_to_dict(v) for v in experiment.variants]),
                    json.dumps(experiment.success_metrics),
                    experiment.minimum_sample_size,
                    experiment.created_at or _utcnow(),
                    experiment.status,
                ),
            )
            self.conn.commit()
        return experiment

    def get_experiment(self, name: str) -> Optional[Experiment]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM experiments WHERE name = %s", (name,))
            row = cur.fetchone()
        return self._row_to_experiment(row) if row else None

    def save_assignment(self, assignment: VariantAssignment) -> VariantAssignment:
        if not assignment.assignment_id:
            assignment.assignment_id = uuid.uuid4().hex
        with self.conn.cursor() as cur:
            cur.execute(
                """INSERT INTO variant_assignments (
                       assignment_id, experiment_name, assignment_key, variant, assigned_at)
                   VALUES (%s, %s, %s, %s::jsonb, %s)
                   ON CONFLICT (experiment_name, assignment_key) DO UPDATE SET
                       variant = EXCLUDED.variant""",
                (
                    assignment.assignment_id,
                    assignment.experiment_name,
                    assignment.assignment_key,
                    json.dumps(
                        self._variant_to_dict(assignment.variant) if assignment.variant else {}
                    ),
                    assignment.assigned_at or _utcnow(),
                ),
            )
            self.conn.commit()
        return assignment

    def get_assignment(self, assignment_id: str) -> Optional[VariantAssignment]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM variant_assignments WHERE assignment_id = %s", (assignment_id,)
            )
            row = cur.fetchone()
        return self._row_to_assignment(row) if row else None

    def find_assignment(
        self, experiment_name: str, assignment_key: str
    ) -> Optional[VariantAssignment]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM variant_assignments "
                "WHERE experiment_name = %s AND assignment_key = %s",
                (experiment_name, assignment_key),
            )
            row = cur.fetchone()
        return self._row_to_assignment(row) if row else None

    def record_outcome(self, outcome: Outcome) -> Outcome:
        if not outcome.recorded_at:
            outcome.recorded_at = _utcnow()
        with self.conn.cursor() as cur:
            cur.execute(
                """INSERT INTO outcomes (
                       experiment_name, assignment_id, variant_name, outcomes, recorded_at)
                   VALUES (%s, %s, %s, %s::jsonb, %s)""",
                (
                    outcome.experiment_name,
                    outcome.assignment_id,
                    outcome.variant_name,
                    json.dumps(outcome.outcomes),
                    outcome.recorded_at,
                ),
            )
            self.conn.commit()
        return outcome

    def list_outcomes(self, experiment_name: str) -> List[Outcome]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM outcomes WHERE experiment_name = %s ORDER BY id",
                (experiment_name,),
            )
            return [
                Outcome(
                    experiment_name=row["experiment_name"],
                    assignment_id=row["assignment_id"],
                    variant_name=row["variant_name"],
                    outcomes=dict(row["outcomes"] or {}),
                    recorded_at=row["recorded_at"],
                )
                for row in cur.fetchall()
            ]

    # ------------------------------------------------------------------
    # Annotation queues
    # ------------------------------------------------------------------
    def save_annotation_queue(self, queue: AnnotationQueue) -> AnnotationQueue:
        if not queue.created_at:
            queue.created_at = _utcnow()
        with self.conn.cursor() as cur:
            cur.execute(
                """INSERT INTO annotation_queues (
                       name, workflow_id, rubric, routing_rules, reviewers, created_at)
                   VALUES (%s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s)
                   ON CONFLICT (name) DO UPDATE SET
                       workflow_id = EXCLUDED.workflow_id,
                       rubric = EXCLUDED.rubric,
                       routing_rules = EXCLUDED.routing_rules,
                       reviewers = EXCLUDED.reviewers""",
                (
                    queue.name,
                    queue.workflow_id,
                    json.dumps([self._rubric_to_dict(r) for r in queue.rubric]),
                    json.dumps([self._routing_to_dict(r) for r in queue.routing_rules]),
                    json.dumps(queue.reviewers),
                    queue.created_at,
                ),
            )
            self.conn.commit()
        return queue

    def get_annotation_queue(self, name: str) -> Optional[AnnotationQueue]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM annotation_queues WHERE name = %s", (name,))
            row = cur.fetchone()
        return self._row_to_queue(row) if row else None

    def add_annotation_item(self, item: AnnotationItem) -> AnnotationItem:
        if not item.item_id:
            item.item_id = uuid.uuid4().hex
        if not item.added_at:
            item.added_at = _utcnow()
        with self.conn.cursor() as cur:
            cur.execute(
                """INSERT INTO annotation_items (
                       item_id, queue_name, trace_id, status, assigned_to, added_at)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (
                    item.item_id,
                    item.queue_name,
                    item.trace_id,
                    item.status,
                    item.assigned_to,
                    item.added_at,
                ),
            )
            self.conn.commit()
        return item

    def get_next_annotation_item(self, queue_name: str, reviewer: str) -> Optional[AnnotationItem]:
        with self.conn.cursor() as cur:
            # SELECT ... FOR UPDATE SKIP LOCKED gives us the same lock-it-to-this-reviewer
            # semantic as the in-memory store (Listing 7.19's intent).
            cur.execute(
                """SELECT * FROM annotation_items
                    WHERE queue_name = %s AND status = 'pending'
                    ORDER BY added_at ASC
                    FOR UPDATE SKIP LOCKED LIMIT 1""",
                (queue_name,),
            )
            row = cur.fetchone()
            if row is None:
                self.conn.commit()
                return None
            cur.execute(
                "UPDATE annotation_items SET status = 'assigned', assigned_to = %s "
                "WHERE item_id = %s",
                (reviewer, row["item_id"]),
            )
            self.conn.commit()
        item = self._row_to_item(row)
        item.status = "assigned"
        item.assigned_to = reviewer
        return item

    def complete_annotation_item(self, item_id: str) -> Optional[AnnotationItem]:
        with self.conn.cursor() as cur:
            cur.execute(
                "UPDATE annotation_items SET status = 'completed' WHERE item_id = %s RETURNING *",
                (item_id,),
            )
            row = cur.fetchone()
            self.conn.commit()
        return self._row_to_item(row) if row else None

    # ------------------------------------------------------------------
    # Row → domain helpers
    # ------------------------------------------------------------------
    def _row_to_target(self, row) -> ExperimentTarget:
        summary_data = row["evaluation_summary"] or {}
        return ExperimentTarget(
            name=row["name"],
            version=row["version"],
            type=row["type"],
            author=row["author"] or "",
            change_description=row["change_description"] or "",
            created_at=row["created_at"],
            status=row["status"],
            evaluation_summary=EvaluationSummary(
                overall_score=summary_data.get("overall_score", 0.0),
                metric_scores=dict(summary_data.get("metric_scores", {})),
                evaluation_id=summary_data.get("evaluation_id", ""),
                test_cases_evaluated=summary_data.get("test_cases_evaluated", 0),
            ),
            metadata=dict(row["metadata"] or {}),
        )

    def _row_to_test_case(self, row) -> TestCase:
        return TestCase(
            id=row["id"],
            input_query=row["input_query"] or "",
            ideal_response=row["ideal_response"] or "",
            key_elements=list(row["key_elements"] or []),
            tags=list(row["tags"] or []),
            metadata=dict(row["metadata"] or {}),
            source_trace_id=row["source_trace_id"] or "",
            needs_review=bool(row["needs_review"]),
        )

    def _row_to_scoring_rule(self, row) -> ScoringRule:
        return ScoringRule(
            name=row["name"],
            workflow_id=row["workflow_id"] or "",
            sample_rate=float(row["sample_rate"] or 1.0),
            scorers=[
                ScorerConfig(
                    name=s.get("name", ""),
                    type=s.get("type", "automated"),
                    criterion=s.get("criterion", ""),
                    judge_model=s.get("judge_model", ""),
                    required_elements=list(s.get("required_elements", [])),
                    options=dict(s.get("options", {})),
                )
                for s in (row["scorers"] or [])
            ],
            alerts=[
                AlertCondition(
                    scorer_name=a.get("scorer_name", ""),
                    threshold_below=a.get("threshold_below", 0.0),
                    threshold_above=a.get("threshold_above", 0.0),
                    window=a.get("window", "1h"),
                )
                for a in (row["alerts"] or [])
            ],
            created_at=row["created_at"],
        )

    def _row_to_experiment(self, row) -> Experiment:
        variants = [self._dict_to_variant(v) for v in (row["variants"] or [])]
        return Experiment(
            name=row["name"],
            workflow_id=row["workflow_id"] or "",
            variants=variants,
            success_metrics=list(row["success_metrics"] or []),
            minimum_sample_size=row["minimum_sample_size"],
            created_at=row["created_at"],
            status=row["status"],
        )

    def _row_to_assignment(self, row) -> VariantAssignment:
        variant_data = row["variant"] or {}
        variant = self._dict_to_variant(variant_data) if variant_data else None
        return VariantAssignment(
            assignment_id=row["assignment_id"],
            experiment_name=row["experiment_name"],
            assignment_key=row["assignment_key"],
            variant=variant,
            assigned_at=row["assigned_at"],
        )

    def _row_to_queue(self, row) -> AnnotationQueue:
        return AnnotationQueue(
            name=row["name"],
            workflow_id=row["workflow_id"] or "",
            rubric=[
                RubricItem(
                    name=r.get("name", ""),
                    type=r.get("type", "numeric"),
                    options=list(r.get("options", [])),
                    min=r.get("min", 0.0),
                    max=r.get("max", 1.0),
                    description=r.get("description", ""),
                )
                for r in (row["rubric"] or [])
            ],
            routing_rules=[
                RoutingRule(
                    condition=r.get("condition", ""),
                    source=r.get("source", ""),
                    rate=r.get("rate", 0.0),
                )
                for r in (row["routing_rules"] or [])
            ],
            reviewers=list(row["reviewers"] or []),
            created_at=row["created_at"],
        )

    def _row_to_item(self, row) -> AnnotationItem:
        return AnnotationItem(
            item_id=row["item_id"],
            queue_name=row["queue_name"],
            trace_id=row["trace_id"] or "",
            status=row["status"],
            assigned_to=row["assigned_to"] or "",
            added_at=row["added_at"],
        )

    # ------------------------------------------------------------------
    # Domain → dict helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _summary_to_dict(summary: EvaluationSummary) -> dict:
        return {
            "overall_score": summary.overall_score,
            "metric_scores": dict(summary.metric_scores),
            "evaluation_id": summary.evaluation_id,
            "test_cases_evaluated": summary.test_cases_evaluated,
        }

    @staticmethod
    def _scorer_to_dict(scorer: ScorerConfig) -> dict:
        return {
            "name": scorer.name,
            "type": scorer.type,
            "criterion": scorer.criterion,
            "judge_model": scorer.judge_model,
            "required_elements": list(scorer.required_elements),
            "options": dict(scorer.options),
        }

    @staticmethod
    def _alert_to_dict(alert: AlertCondition) -> dict:
        return {
            "scorer_name": alert.scorer_name,
            "threshold_below": alert.threshold_below,
            "threshold_above": alert.threshold_above,
            "window": alert.window,
        }

    @staticmethod
    def _rubric_to_dict(item: RubricItem) -> dict:
        return {
            "name": item.name,
            "type": item.type,
            "options": list(item.options),
            "min": item.min,
            "max": item.max,
            "description": item.description,
        }

    @staticmethod
    def _routing_to_dict(rule: RoutingRule) -> dict:
        return {"condition": rule.condition, "source": rule.source, "rate": rule.rate}

    @staticmethod
    def _variant_to_dict(variant: Optional[Variant]) -> dict:
        if variant is None:
            return {}
        out: dict = {
            "name": variant.name,
            "traffic_allocation": variant.traffic_allocation,
            "metadata": dict(variant.metadata),
        }
        if variant.prompt_variant:
            out["prompt_variant"] = {
                "prompt_name": variant.prompt_variant.prompt_name,
                "version": variant.prompt_variant.version,
            }
        if variant.model_config_variant:
            out["model_config_variant"] = {
                "model": variant.model_config_variant.model,
                "temperature": variant.model_config_variant.temperature,
                "max_tokens": variant.model_config_variant.max_tokens,
            }
        if variant.retrieval_config_variant:
            out["retrieval_config_variant"] = {
                "index_name": variant.retrieval_config_variant.index_name,
                "top_k": variant.retrieval_config_variant.top_k,
            }
        return out

    @staticmethod
    def _dict_to_variant(data: dict) -> Variant:
        variant = Variant(
            name=data.get("name", ""),
            traffic_allocation=data.get("traffic_allocation", 0.0),
            metadata=dict(data.get("metadata", {})),
        )
        if "prompt_variant" in data and data["prompt_variant"]:
            pv = data["prompt_variant"]
            variant.prompt_variant = PromptVariant(
                prompt_name=pv.get("prompt_name", ""),
                version=pv.get("version", 0),
            )
        if "model_config_variant" in data and data["model_config_variant"]:
            mv = data["model_config_variant"]
            variant.model_config_variant = ModelConfigVariant(
                model=mv.get("model", ""),
                temperature=mv.get("temperature", 0.7),
                max_tokens=mv.get("max_tokens", 0),
            )
        if "retrieval_config_variant" in data and data["retrieval_config_variant"]:
            rv = data["retrieval_config_variant"]
            variant.retrieval_config_variant = RetrievalConfigVariant(
                index_name=rv.get("index_name", ""),
                top_k=rv.get("top_k", 5),
            )
        return variant

    @staticmethod
    def _insert_test_case(cur, dataset_name: str, case: TestCase) -> None:
        cur.execute(
            """INSERT INTO test_cases (
                   id, dataset_name, input_query, ideal_response,
                   key_elements, tags, metadata, source_trace_id, needs_review)
               VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s)
               ON CONFLICT (id) DO NOTHING""",
            (
                case.id,
                dataset_name,
                case.input_query,
                case.ideal_response,
                json.dumps(case.key_elements),
                json.dumps(case.tags),
                json.dumps(case.metadata),
                case.source_trace_id,
                case.needs_review,
            ),
        )
