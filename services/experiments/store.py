"""
Experimentation store — in-memory persistence for the improvement loop.

This holds experiment targets, datasets, evaluations, scoring rules,
A/B experiments, and annotation queues. The structure mirrors the
public API of the Experimentation Service so the gRPC servicer can
remain a thin proto<->domain layer.
"""

from __future__ import annotations

import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import replace
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from services.experiments.models import (
    AnnotationItem,
    AnnotationQueue,
    Dataset,
    Evaluation,
    EvaluationResults,
    Experiment,
    ExperimentTarget,
    Outcome,
    ScoringRule,
    TestCase,
    VariantAssignment,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ExperimentStore(ABC):
    # Targets
    @abstractmethod
    def register_target(self, target: ExperimentTarget) -> ExperimentTarget: ...

    @abstractmethod
    def get_target_history(self, name: str) -> List[ExperimentTarget]: ...

    @abstractmethod
    def get_target(self, name: str, version: int) -> Optional[ExperimentTarget]: ...

    @abstractmethod
    def update_target_status(self, name: str, version: int, status: str) -> ExperimentTarget: ...

    @abstractmethod
    def update_target_summary(self, name: str, version: int, summary) -> ExperimentTarget: ...

    # Datasets
    @abstractmethod
    def create_dataset(self, dataset: Dataset) -> Dataset: ...

    @abstractmethod
    def add_test_cases(self, name: str, cases: List[TestCase]) -> Dataset: ...

    @abstractmethod
    def get_dataset(self, name: str) -> Optional[Dataset]: ...

    # Evaluations
    @abstractmethod
    def save_evaluation(self, evaluation: Evaluation) -> Evaluation: ...

    @abstractmethod
    def get_evaluation(self, evaluation_id: str) -> Optional[Evaluation]: ...

    @abstractmethod
    def save_evaluation_results(self, results: EvaluationResults) -> EvaluationResults: ...

    @abstractmethod
    def get_evaluation_results(self, evaluation_id: str) -> Optional[EvaluationResults]: ...

    # Scoring rules
    @abstractmethod
    def save_scoring_rule(self, rule: ScoringRule) -> ScoringRule: ...

    @abstractmethod
    def list_scoring_rules(self, workflow_id: str = "") -> List[ScoringRule]: ...

    @abstractmethod
    def get_scoring_rule(self, name: str) -> Optional[ScoringRule]: ...

    # Experiments
    @abstractmethod
    def save_experiment(self, experiment: Experiment) -> Experiment: ...

    @abstractmethod
    def get_experiment(self, name: str) -> Optional[Experiment]: ...

    @abstractmethod
    def save_assignment(self, assignment: VariantAssignment) -> VariantAssignment: ...

    @abstractmethod
    def get_assignment(self, assignment_id: str) -> Optional[VariantAssignment]: ...

    @abstractmethod
    def find_assignment(
        self, experiment_name: str, assignment_key: str
    ) -> Optional[VariantAssignment]: ...

    @abstractmethod
    def record_outcome(self, outcome: Outcome) -> Outcome: ...

    @abstractmethod
    def list_outcomes(self, experiment_name: str) -> List[Outcome]: ...

    # Annotation queues
    @abstractmethod
    def save_annotation_queue(self, queue: AnnotationQueue) -> AnnotationQueue: ...

    @abstractmethod
    def get_annotation_queue(self, name: str) -> Optional[AnnotationQueue]: ...

    @abstractmethod
    def add_annotation_item(self, item: AnnotationItem) -> AnnotationItem: ...

    @abstractmethod
    def get_next_annotation_item(
        self, queue_name: str, reviewer: str
    ) -> Optional[AnnotationItem]: ...

    @abstractmethod
    def complete_annotation_item(self, item_id: str) -> Optional[AnnotationItem]: ...


class InMemoryExperimentStore(ExperimentStore):
    """In-memory implementation suitable for tests, quickstarts, and Docker dev."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._targets: Dict[Tuple[str, int], ExperimentTarget] = {}
        self._datasets: Dict[str, Dataset] = {}
        self._evaluations: Dict[str, Evaluation] = {}
        self._evaluation_results: Dict[str, EvaluationResults] = {}
        self._scoring_rules: Dict[str, ScoringRule] = {}
        self._experiments: Dict[str, Experiment] = {}
        self._assignments: Dict[str, VariantAssignment] = {}
        self._assignment_index: Dict[Tuple[str, str], str] = {}
        self._outcomes: List[Outcome] = []
        self._queues: Dict[str, AnnotationQueue] = {}
        self._queue_items: Dict[str, List[AnnotationItem]] = {}

    # ------------------------------------------------------------------
    # Targets
    # ------------------------------------------------------------------
    def register_target(self, target: ExperimentTarget) -> ExperimentTarget:
        with self._lock:
            stored = replace(target)
            if stored.created_at is None:
                stored.created_at = _utcnow()
            self._targets[(stored.name, stored.version)] = stored
            return stored

    def get_target_history(self, name: str) -> List[ExperimentTarget]:
        with self._lock:
            return sorted(
                (t for (n, _), t in self._targets.items() if n == name),
                key=lambda t: t.version,
            )

    def get_target(self, name: str, version: int) -> Optional[ExperimentTarget]:
        with self._lock:
            return self._targets.get((name, version))

    def update_target_status(self, name: str, version: int, status: str) -> ExperimentTarget:
        with self._lock:
            target = self._targets.get((name, version))
            if target is None:
                raise KeyError(f"Target {name}:{version} not registered")
            target.status = status
            if status == "ACTIVE":
                for (other_name, _), other in self._targets.items():
                    if other_name == name and other is not target and other.status == "ACTIVE":
                        other.status = "DEPRECATED"
            return target

    def update_target_summary(self, name: str, version: int, summary) -> ExperimentTarget:
        with self._lock:
            target = self._targets.get((name, version))
            if target is None:
                raise KeyError(f"Target {name}:{version} not registered")
            target.evaluation_summary = summary
            return target

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    def create_dataset(self, dataset: Dataset) -> Dataset:
        with self._lock:
            stored = replace(dataset)
            stored.created_at = stored.created_at or _utcnow()
            stored.updated_at = stored.updated_at or _utcnow()
            for case in stored.test_cases:
                if not case.id:
                    case.id = uuid.uuid4().hex
            self._datasets[stored.name] = stored
            return stored

    def add_test_cases(self, name: str, cases: List[TestCase]) -> Dataset:
        with self._lock:
            dataset = self._datasets.get(name)
            if dataset is None:
                raise KeyError(f"Dataset '{name}' not found")
            for case in cases:
                if not case.id:
                    case.id = uuid.uuid4().hex
                dataset.test_cases.append(case)
            dataset.updated_at = _utcnow()
            return dataset

    def get_dataset(self, name: str) -> Optional[Dataset]:
        with self._lock:
            return self._datasets.get(name)

    # ------------------------------------------------------------------
    # Evaluations
    # ------------------------------------------------------------------
    def save_evaluation(self, evaluation: Evaluation) -> Evaluation:
        with self._lock:
            if not evaluation.evaluation_id:
                evaluation.evaluation_id = uuid.uuid4().hex
            self._evaluations[evaluation.evaluation_id] = evaluation
            return evaluation

    def get_evaluation(self, evaluation_id: str) -> Optional[Evaluation]:
        with self._lock:
            return self._evaluations.get(evaluation_id)

    def save_evaluation_results(self, results: EvaluationResults) -> EvaluationResults:
        with self._lock:
            self._evaluation_results[results.evaluation_id] = results
            return results

    def get_evaluation_results(self, evaluation_id: str) -> Optional[EvaluationResults]:
        with self._lock:
            return self._evaluation_results.get(evaluation_id)

    # ------------------------------------------------------------------
    # Scoring rules
    # ------------------------------------------------------------------
    def save_scoring_rule(self, rule: ScoringRule) -> ScoringRule:
        with self._lock:
            self._scoring_rules[rule.name] = rule
            return rule

    def list_scoring_rules(self, workflow_id: str = "") -> List[ScoringRule]:
        with self._lock:
            rules = list(self._scoring_rules.values())
        if workflow_id:
            rules = [r for r in rules if r.workflow_id == workflow_id]
        return rules

    def get_scoring_rule(self, name: str) -> Optional[ScoringRule]:
        with self._lock:
            return self._scoring_rules.get(name)

    # ------------------------------------------------------------------
    # Experiments
    # ------------------------------------------------------------------
    def save_experiment(self, experiment: Experiment) -> Experiment:
        with self._lock:
            self._experiments[experiment.name] = experiment
            return experiment

    def get_experiment(self, name: str) -> Optional[Experiment]:
        with self._lock:
            return self._experiments.get(name)

    def save_assignment(self, assignment: VariantAssignment) -> VariantAssignment:
        with self._lock:
            if not assignment.assignment_id:
                assignment.assignment_id = uuid.uuid4().hex
            self._assignments[assignment.assignment_id] = assignment
            self._assignment_index[(assignment.experiment_name, assignment.assignment_key)] = (
                assignment.assignment_id
            )
            return assignment

    def get_assignment(self, assignment_id: str) -> Optional[VariantAssignment]:
        with self._lock:
            return self._assignments.get(assignment_id)

    def find_assignment(
        self, experiment_name: str, assignment_key: str
    ) -> Optional[VariantAssignment]:
        with self._lock:
            assignment_id = self._assignment_index.get((experiment_name, assignment_key))
            if assignment_id is None:
                return None
            return self._assignments.get(assignment_id)

    def record_outcome(self, outcome: Outcome) -> Outcome:
        with self._lock:
            if not outcome.recorded_at:
                outcome.recorded_at = _utcnow()
            self._outcomes.append(outcome)
            return outcome

    def list_outcomes(self, experiment_name: str) -> List[Outcome]:
        with self._lock:
            return [o for o in self._outcomes if o.experiment_name == experiment_name]

    # ------------------------------------------------------------------
    # Annotation queues
    # ------------------------------------------------------------------
    def save_annotation_queue(self, queue: AnnotationQueue) -> AnnotationQueue:
        with self._lock:
            queue.created_at = queue.created_at or _utcnow()
            self._queues[queue.name] = queue
            self._queue_items.setdefault(queue.name, [])
            return queue

    def get_annotation_queue(self, name: str) -> Optional[AnnotationQueue]:
        with self._lock:
            return self._queues.get(name)

    def add_annotation_item(self, item: AnnotationItem) -> AnnotationItem:
        with self._lock:
            if not item.item_id:
                item.item_id = uuid.uuid4().hex
            item.added_at = item.added_at or _utcnow()
            self._queue_items.setdefault(item.queue_name, []).append(item)
            return item

    def get_next_annotation_item(self, queue_name: str, reviewer: str) -> Optional[AnnotationItem]:
        with self._lock:
            items = self._queue_items.get(queue_name, [])
            for item in items:
                if item.status == "pending":
                    item.assigned_to = reviewer
                    return item
            return None

    def complete_annotation_item(self, item_id: str) -> Optional[AnnotationItem]:
        with self._lock:
            for items in self._queue_items.values():
                for item in items:
                    if item.item_id == item_id:
                        item.status = "completed"
                        return item
            return None
