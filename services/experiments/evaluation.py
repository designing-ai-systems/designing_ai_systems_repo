"""
Offline evaluation pipeline.

Runs each ExperimentTarget against every test case in a Dataset, scores
the results, and aggregates them. The pipeline supports synthetic
"runners" so the chapter-7 examples can demonstrate the flow without
external API keys; production deployments would inject a runner that
calls the Model Service through the SDK.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 7.16: register and run an evaluation
"""

from __future__ import annotations

import statistics
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional

from services.experiments.models import (
    Dataset,
    EvaluationResults,
    ExperimentTarget,
    PerCaseResult,
    TargetResult,
    TestCase,
)
from services.experiments.scorers import KeyElementScorer, Scorer
from services.observability.models import Trace

# A runner answers: given a target and a test case input, what's the response?
TargetRunner = Callable[[ExperimentTarget, TestCase], str]


def default_runner(target: ExperimentTarget, case: TestCase) -> str:
    """Synthetic runner — returns a deterministic response for tests/quickstarts."""
    base = case.ideal_response or case.input_query
    suffix = ""
    if target.type == "PROMPT":
        suffix = f" (variant {target.target_id})"
    return f"{base}{suffix}"


@dataclass
class EvaluationProgress:
    evaluation_id: str = ""
    status: str = "running"
    completed_cases: int = 0
    total_cases: int = 0
    current_target: str = ""
    results: Optional[EvaluationResults] = None


@dataclass
class OfflineEvaluationPipeline:
    """Runs targets against a dataset and computes per-target aggregates."""

    runner: TargetRunner = field(default=default_runner)
    repeats_per_case: int = 1
    extra_scorers: List[Scorer] = field(default_factory=list)

    def run(
        self,
        *,
        evaluation_id: str,
        dataset: Dataset,
        targets: List[ExperimentTarget],
        metrics: List[str],
    ) -> Iterator[EvaluationProgress]:
        evaluation_id = evaluation_id or uuid.uuid4().hex
        per_case: List[PerCaseResult] = []
        per_target: Dict[str, Dict[str, List[float]]] = {
            t.target_id: {m: [] for m in metrics} for t in targets
        }
        total_cases = len(dataset.test_cases) * len(targets) * max(1, self.repeats_per_case)
        completed = 0

        for target in targets:
            for case in dataset.test_cases:
                aggregated_metrics: Dict[str, float] = {}
                outputs: List[str] = []
                for _ in range(max(1, self.repeats_per_case)):
                    output = self.runner(target, case)
                    outputs.append(output)
                    fake_trace = Trace(
                        trace_id=f"eval-{evaluation_id}-{target.target_id}-{case.id or 'x'}",
                        input=case.input_query,
                        output=output,
                    )
                    case_metrics = self._score_case(case, fake_trace, metrics)
                    for name, value in case_metrics.items():
                        aggregated_metrics.setdefault(name, 0.0)
                        aggregated_metrics[name] += value
                    completed += 1
                # Average across repeats
                for name in list(aggregated_metrics.keys()):
                    aggregated_metrics[name] /= max(1, self.repeats_per_case)

                per_case.append(
                    PerCaseResult(
                        target_id=target.target_id,
                        test_case_id=case.id,
                        output=outputs[-1] if outputs else "",
                        metric_scores=aggregated_metrics,
                    )
                )
                for name, value in aggregated_metrics.items():
                    per_target.setdefault(target.target_id, {}).setdefault(name, []).append(value)

                yield EvaluationProgress(
                    evaluation_id=evaluation_id,
                    status="running",
                    completed_cases=completed,
                    total_cases=total_cases,
                    current_target=target.target_id,
                )

        target_results: List[TargetResult] = []
        for target in targets:
            metric_means: Dict[str, float] = {}
            for name in metrics:
                values = per_target.get(target.target_id, {}).get(name, [])
                metric_means[name] = statistics.fmean(values) if values else 0.0
            overall = statistics.fmean(metric_means.values()) if metric_means else 0.0
            target_results.append(
                TargetResult(
                    target_id=target.target_id,
                    metric_scores=metric_means,
                    overall_score=overall,
                    cases_run=len(dataset.test_cases) * max(1, self.repeats_per_case),
                )
            )

        results = EvaluationResults(
            evaluation_id=evaluation_id,
            dataset_name=dataset.name,
            target_results=target_results,
            per_case_results=per_case,
        )
        yield EvaluationProgress(
            evaluation_id=evaluation_id,
            status="completed",
            completed_cases=total_cases,
            total_cases=total_cases,
            current_target="",
            results=results,
        )

    def _score_case(self, case: TestCase, trace: Trace, metrics: List[str]) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for metric_name in metrics:
            scorer: Scorer
            extra = next((s for s in self.extra_scorers if s.name == metric_name), None)
            if extra is not None:
                scorer = extra
            else:
                scorer = KeyElementScorer(
                    name=metric_name, required_elements=list(case.key_elements)
                )
            score = scorer.score(trace)
            value = score.value
            if isinstance(value, bool):
                scores[metric_name] = 1.0 if value else 0.0
            elif isinstance(value, (int, float)):
                scores[metric_name] = float(value)
            else:
                scores[metric_name] = 0.0
        return scores
