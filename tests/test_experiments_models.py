"""Unit tests for experimentation domain dataclasses (Listing 7.15)."""

from services.experiments.models import EvaluationSummary, ExperimentTarget
from services.experiments.store import InMemoryExperimentStore


class TestTargetLifecycle:
    def test_default_status_draft(self):
        t = ExperimentTarget(name="prompt-x", version=1)
        assert t.status == "DRAFT"

    def test_target_id_format(self):
        assert ExperimentTarget(name="prompt-x", version=3).target_id == "prompt-x:3"

    def test_active_deprecates_previous_active(self):
        store = InMemoryExperimentStore()
        store.register_target(ExperimentTarget(name="p", version=1, status="DRAFT"))
        store.register_target(ExperimentTarget(name="p", version=2, status="DRAFT"))
        store.update_target_status("p", 1, "ACTIVE")
        store.update_target_status("p", 2, "ACTIVE")
        assert store.get_target("p", 1).status == "DEPRECATED"
        assert store.get_target("p", 2).status == "ACTIVE"


class TestEvaluationSummary:
    def test_metric_scores_default(self):
        es = EvaluationSummary(overall_score=0.87)
        assert es.overall_score == 0.87
        assert es.metric_scores == {}
        assert es.test_cases_evaluated == 0
