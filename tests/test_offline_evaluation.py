"""Unit tests for the offline evaluation pipeline (Listing 7.16)."""

from services.experiments.evaluation import OfflineEvaluationPipeline
from services.experiments.models import (
    Dataset,
    ExperimentTarget,
    TestCase,
)


class TestOfflineEvaluation:
    def test_runs_each_case_for_each_target(self):
        targets = [
            ExperimentTarget(name="p", version=1),
            ExperimentTarget(name="p", version=2),
        ]
        dataset = Dataset(
            name="ds",
            test_cases=[
                TestCase(
                    id="tc1", input_query="q1", ideal_response="answer1", key_elements=["answer1"]
                ),
                TestCase(
                    id="tc2", input_query="q2", ideal_response="answer2", key_elements=["answer2"]
                ),
            ],
        )
        pipeline = OfflineEvaluationPipeline()
        progresses = list(
            pipeline.run(
                evaluation_id="eval-1",
                dataset=dataset,
                targets=targets,
                metrics=["key_elements"],
            )
        )
        last = progresses[-1]
        assert last.status == "completed"
        assert {tr.target_id for tr in last.results.target_results} == {"p:1", "p:2"}
        # 2 test cases × 2 targets = 4 per-case rows
        assert len(last.results.per_case_results) == 4

    def test_target_aggregates_match_per_case_results(self):
        targets = [ExperimentTarget(name="p", version=1)]
        dataset = Dataset(
            name="ds",
            test_cases=[
                TestCase(id="tc1", ideal_response="a", key_elements=["a"]),
                TestCase(id="tc2", ideal_response="b", key_elements=["b"]),
            ],
        )
        pipeline = OfflineEvaluationPipeline()
        last = list(
            pipeline.run(
                evaluation_id="eval-2",
                dataset=dataset,
                targets=targets,
                metrics=["key_elements"],
            )
        )[-1]
        # Both cases score 1.0 because the synthetic runner echoes the ideal answer.
        assert last.results.target_results[0].metric_scores["key_elements"] == 1.0
