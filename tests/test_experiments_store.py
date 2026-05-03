"""Unit tests for InMemoryExperimentStore."""

import pytest

from services.experiments.models import (
    AnnotationItem,
    AnnotationQueue,
    Dataset,
    Experiment,
    ExperimentTarget,
    Outcome,
    RubricItem,
    ScoringRule,
    TestCase,
    Variant,
    VariantAssignment,
)
from services.experiments.store import InMemoryExperimentStore


class TestDatasets:
    def test_create_then_add_test_cases(self):
        store = InMemoryExperimentStore()
        store.create_dataset(Dataset(name="ds-1"))
        store.add_test_cases("ds-1", [TestCase(input_query="q1"), TestCase(input_query="q2")])
        ds = store.get_dataset("ds-1")
        assert len(ds.test_cases) == 2
        assert all(c.id for c in ds.test_cases)  # auto IDs

    def test_add_to_missing_dataset_raises(self):
        store = InMemoryExperimentStore()
        with pytest.raises(KeyError):
            store.add_test_cases("missing", [TestCase()])


class TestScoringRules:
    def test_save_and_list(self):
        store = InMemoryExperimentStore()
        store.save_scoring_rule(ScoringRule(name="r1", workflow_id="wf-1"))
        store.save_scoring_rule(ScoringRule(name="r2", workflow_id="wf-2"))
        wf1_rules = store.list_scoring_rules(workflow_id="wf-1")
        assert [r.name for r in wf1_rules] == ["r1"]


class TestExperimentsAndAssignments:
    def test_assignment_lookup_is_idempotent(self):
        store = InMemoryExperimentStore()
        exp = Experiment(name="exp", variants=[Variant(name="control", traffic_allocation=1.0)])
        store.save_experiment(exp)
        a1 = store.save_assignment(
            VariantAssignment(experiment_name="exp", assignment_key="u-1", variant=exp.variants[0])
        )
        a2 = store.find_assignment("exp", "u-1")
        assert a2.assignment_id == a1.assignment_id

    def test_record_outcome_filters_per_experiment(self):
        store = InMemoryExperimentStore()
        store.record_outcome(Outcome(experiment_name="a", assignment_id="x", outcomes={"x": 1}))
        store.record_outcome(Outcome(experiment_name="b", assignment_id="y", outcomes={"x": 1}))
        a_outcomes = store.list_outcomes("a")
        assert len(a_outcomes) == 1


class TestAnnotationQueue:
    def test_queue_routing_first_pending_only(self):
        store = InMemoryExperimentStore()
        queue = AnnotationQueue(
            name="q", workflow_id="wf-1", rubric=[RubricItem(name="x", type="numeric")]
        )
        store.save_annotation_queue(queue)
        store.add_annotation_item(AnnotationItem(queue_name="q", trace_id="t1"))
        store.add_annotation_item(AnnotationItem(queue_name="q", trace_id="t2"))
        next_item = store.get_next_annotation_item("q", reviewer="alice")
        assert next_item.trace_id == "t1"
        assert next_item.assigned_to == "alice"

    def test_complete_marks_done(self):
        store = InMemoryExperimentStore()
        store.save_annotation_queue(AnnotationQueue(name="q"))
        item = store.add_annotation_item(AnnotationItem(queue_name="q", trace_id="t"))
        store.complete_annotation_item(item.item_id)
        assert store.get_next_annotation_item("q", reviewer="alice") is None


class TestTargetSummaryUpdate:
    def test_update_target_summary_propagates(self):
        from services.experiments.models import EvaluationSummary

        store = InMemoryExperimentStore()
        store.register_target(ExperimentTarget(name="p", version=1))
        store.update_target_summary("p", 1, EvaluationSummary(overall_score=0.91))
        target = store.get_target("p", 1)
        assert target.evaluation_summary.overall_score == 0.91
