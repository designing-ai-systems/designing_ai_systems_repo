"""
Experimentation Service — gRPC service implementation (grpc.aio).

The improvement loop: target lifecycle, datasets, offline evaluation,
online scoring, A/B testing, and human annotation. The service is
stateful but ephemeral: an in-memory store backs every entity, and
trace/score lookups are delegated to the Observability Service through
an injected stub (or skipped when no stub is configured).

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 7.14: ExperimentationService gRPC contract
  - Listing 7.16: register_target + run_evaluation flow
  - Listing 7.17: AddFromProduction (low-score traces → dataset)
  - Listing 7.18: scoring rule for production traffic
  - Listing 7.19: annotation queue
  - Listing 7.20: experiment-aware workflow integration
"""

from __future__ import annotations

import logging
import statistics
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import grpc
import grpc.aio
from google.protobuf.timestamp_pb2 import Timestamp

from proto import experiments_pb2, experiments_pb2_grpc, observability_pb2
from services.experiments.ab_testing import assign_variant, welch_t_test
from services.experiments.evaluation import OfflineEvaluationPipeline
from services.experiments.models import (
    AlertCondition,
    AnnotationItem,
    AnnotationQueue,
    Dataset,
    Evaluation,
    EvaluationSummary,
    Experiment,
    ExperimentResults,
    ExperimentTarget,
    MetricComparison,
    MetricSummary,
    ModelConfigVariant,
    Outcome,
    PromptVariant,
    RetrievalConfigVariant,
    RoutingRule,
    RubricItem,
    ScorerConfig,
    ScoringRule,
    TestCase,
    Variant,
    VariantAssignment,
    VariantSummary,
)
from services.experiments.scorers import build_scorer
from services.experiments.store import ExperimentStore, InMemoryExperimentStore
from services.observability.models import Score, Trace
from services.shared.servicer_base import BaseAioServicer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enum mapping
# ---------------------------------------------------------------------------

_TYPE_TO_PROTO = {
    "PROMPT": experiments_pb2.PROMPT,
    "MODEL_CONFIG": experiments_pb2.MODEL_CONFIG,
    "RETRIEVAL_CONFIG": experiments_pb2.RETRIEVAL_CONFIG,
}
_TYPE_FROM_PROTO = {v: k for k, v in _TYPE_TO_PROTO.items()}

_STATUS_TO_PROTO = {
    "DRAFT": experiments_pb2.DRAFT,
    "TESTING": experiments_pb2.TESTING,
    "ACTIVE": experiments_pb2.ACTIVE,
    "DEPRECATED": experiments_pb2.DEPRECATED,
}
_STATUS_FROM_PROTO = {v: k for k, v in _STATUS_TO_PROTO.items()}


def _ts_to_dt(ts: Optional[Timestamp]) -> Optional[datetime]:
    if ts is None or (ts.seconds == 0 and ts.nanos == 0):
        return None
    return datetime.fromtimestamp(ts.seconds + ts.nanos / 1e9, tz=timezone.utc)


def _dt_to_ts(dt: Optional[datetime]) -> Optional[Timestamp]:
    if dt is None:
        return None
    ts = Timestamp()
    ts.FromDatetime(dt)
    return ts


class ExperimentationServiceImpl(
    experiments_pb2_grpc.ExperimentationServiceServicer, BaseAioServicer
):
    """The Experimentation gRPC servicer."""

    def __init__(
        self,
        store: Optional[ExperimentStore] = None,
        *,
        observability_stub: Optional[Any] = None,
        evaluation_pipeline: Optional[OfflineEvaluationPipeline] = None,
    ) -> None:
        self.store: ExperimentStore = store or InMemoryExperimentStore()
        self.observability_stub = observability_stub
        self.evaluation_pipeline = evaluation_pipeline or OfflineEvaluationPipeline()

    def add_to_aio_server(self, server: grpc.aio.Server) -> None:
        experiments_pb2_grpc.add_ExperimentationServiceServicer_to_server(self, server)

    # ------------------------------------------------------------------
    # Targets
    # ------------------------------------------------------------------
    async def RegisterTarget(self, request, context):
        target = ExperimentTarget(
            name=request.name,
            version=request.version or self._next_target_version(request.name),
            type=_TYPE_FROM_PROTO.get(request.type, "PROMPT"),
            author=request.author,
            change_description=request.change_description,
            metadata=dict(request.metadata),
            status="DRAFT",
        )
        target = self.store.register_target(target)
        return self._target_to_proto(target)

    async def GetTargetHistory(self, request, context):
        history = self.store.get_target_history(request.name)
        return experiments_pb2.TargetHistory(
            name=request.name,
            versions=[self._target_to_proto(t) for t in history],
        )

    async def CompareTargets(self, request, context):
        rows = []
        for tid in request.target_ids:
            name, _, version = tid.partition(":")
            try:
                target = self.store.get_target(name, int(version))
            except (TypeError, ValueError):
                target = None
            if target is None:
                continue
            summary = target.evaluation_summary or EvaluationSummary()
            rows.append(
                experiments_pb2.TargetComparisonRow(
                    target_id=tid,
                    overall_score=summary.overall_score,
                    metric_scores=summary.metric_scores,
                )
            )
        return experiments_pb2.CompareTargetsResponse(rows=rows)

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    async def CreateDataset(self, request, context):
        dataset = Dataset(
            name=request.name,
            test_cases=[self._proto_test_case_to_domain(c) for c in request.test_cases],
            metadata=dict(request.metadata),
        )
        dataset = self.store.create_dataset(dataset)
        return self._dataset_to_proto(dataset)

    async def AddTestCases(self, request, context):
        dataset = self.store.add_test_cases(
            request.name,
            [self._proto_test_case_to_domain(c) for c in request.test_cases],
        )
        return self._dataset_to_proto(dataset)

    async def AddFromProduction(self, request, context):
        cases: List[TestCase] = []
        traces = self._fetch_traces(request.trace_ids)
        for trace in traces:
            cases.append(
                TestCase(
                    id=uuid.uuid4().hex,
                    input_query=trace.input or "",
                    ideal_response=trace.output or "",
                    tags=list(trace.tags),
                    source_trace_id=trace.trace_id,
                    needs_review=request.require_human_review,
                )
            )
        dataset = self.store.add_test_cases(request.dataset_name, cases)
        return self._dataset_to_proto(dataset)

    # ------------------------------------------------------------------
    # Offline evaluation
    # ------------------------------------------------------------------
    async def CreateEvaluation(self, request, context):
        evaluation = Evaluation(
            evaluation_id=uuid.uuid4().hex,
            dataset_name=request.dataset_name,
            target_ids=list(request.target_ids),
            metrics=list(request.metrics),
        )
        self.store.save_evaluation(evaluation)
        return self._evaluation_to_proto(evaluation)

    async def RunEvaluation(self, request, context):
        dataset = self.store.get_dataset(request.dataset_name)
        if dataset is None:
            await context.abort(
                grpc.StatusCode.NOT_FOUND, f"Dataset '{request.dataset_name}' not found"
            )
            return
        targets: List[ExperimentTarget] = []
        for tid in request.target_ids:
            name, _, version = tid.partition(":")
            try:
                target = self.store.get_target(name, int(version))
            except (TypeError, ValueError):
                target = None
            if target is None:
                await context.abort(grpc.StatusCode.NOT_FOUND, f"Target '{tid}' not registered")
                return
            target.status = "TESTING"
            self.store.update_target_status(target.name, target.version, "TESTING")
            targets.append(target)

        evaluation = Evaluation(
            evaluation_id=uuid.uuid4().hex,
            dataset_name=request.dataset_name,
            target_ids=list(request.target_ids),
            metrics=list(request.metrics),
            status="running",
        )
        self.store.save_evaluation(evaluation)

        pipeline = self.evaluation_pipeline
        if request.repeats_per_case and request.repeats_per_case > 0:
            pipeline = OfflineEvaluationPipeline(
                runner=self.evaluation_pipeline.runner,
                repeats_per_case=request.repeats_per_case,
                extra_scorers=list(self.evaluation_pipeline.extra_scorers),
            )

        last_progress = None
        for progress in pipeline.run(
            evaluation_id=evaluation.evaluation_id,
            dataset=dataset,
            targets=targets,
            metrics=list(request.metrics),
        ):
            last_progress = progress
            yield self._progress_to_proto(progress)

        if last_progress and last_progress.results is not None:
            evaluation.status = "completed"
            self.store.save_evaluation(evaluation)
            self.store.save_evaluation_results(last_progress.results)
            for target_result in last_progress.results.target_results:
                summary = EvaluationSummary(
                    overall_score=target_result.overall_score,
                    metric_scores=dict(target_result.metric_scores),
                    evaluation_id=evaluation.evaluation_id,
                    test_cases_evaluated=target_result.cases_run,
                )
                name, _, version = target_result.target_id.partition(":")
                try:
                    self.store.update_target_summary(name, int(version), summary)
                except (KeyError, ValueError):
                    pass

    async def GetEvaluationResults(self, request, context):
        results = self.store.get_evaluation_results(request.evaluation_id)
        if results is None:
            return experiments_pb2.EvaluationResults(evaluation_id=request.evaluation_id)
        return experiments_pb2.EvaluationResults(
            evaluation_id=results.evaluation_id,
            dataset_name=results.dataset_name,
            target_results=[
                experiments_pb2.TargetResult(
                    target_id=tr.target_id,
                    metric_scores=tr.metric_scores,
                    overall_score=tr.overall_score,
                    cases_run=tr.cases_run,
                )
                for tr in results.target_results
            ],
            per_case_results=[
                experiments_pb2.PerCaseResult(
                    target_id=p.target_id,
                    test_case_id=p.test_case_id,
                    output=p.output,
                    metric_scores=p.metric_scores,
                )
                for p in results.per_case_results
            ],
        )

    # ------------------------------------------------------------------
    # Online scoring
    # ------------------------------------------------------------------
    async def CreateScoringRule(self, request, context):
        rule = self._proto_rule_to_domain(request.rule)
        rule = self.store.save_scoring_rule(rule)
        return self._rule_to_proto(rule)

    async def ListScoringRules(self, request, context):
        rules = self.store.list_scoring_rules(request.workflow_id)
        return experiments_pb2.ListScoringRulesResponse(
            rules=[self._rule_to_proto(r) for r in rules],
        )

    async def RunScoringRule(self, request, context):
        rule = self.store.get_scoring_rule(request.rule_name)
        if rule is None:
            await context.abort(
                grpc.StatusCode.NOT_FOUND, f"Scoring rule '{request.rule_name}' not found"
            )
            return experiments_pb2.RunScoringRuleResponse()
        traces = self._fetch_traces(request.trace_ids)
        traces_scored = 0
        scores_recorded = 0
        for trace in traces:
            traces_scored += 1
            for cfg in rule.scorers:
                try:
                    scorer = build_scorer(cfg)
                    score = scorer.score(trace)
                    self._record_score(score)
                    scores_recorded += 1
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning(
                        "Scorer %s failed for trace %s: %s", cfg.name, trace.trace_id, exc
                    )
        return experiments_pb2.RunScoringRuleResponse(
            traces_scored=traces_scored,
            scores_recorded=scores_recorded,
        )

    # ------------------------------------------------------------------
    # A/B testing
    # ------------------------------------------------------------------
    async def CreateExperiment(self, request, context):
        exp = self._proto_experiment_to_domain(request.experiment)
        exp.status = "running"
        exp = self.store.save_experiment(exp)
        return self._experiment_to_proto(exp)

    async def AssignVariant(self, request, context):
        existing = self.store.find_assignment(request.experiment_name, request.assignment_key)
        if existing is not None:
            return self._assignment_to_proto(existing)

        experiment = self.store.get_experiment(request.experiment_name)
        if experiment is None:
            return experiments_pb2.VariantAssignment(
                experiment_name=request.experiment_name,
                assignment_key=request.assignment_key,
            )

        variant = assign_variant(experiment, request.assignment_key)
        if variant is None:
            return experiments_pb2.VariantAssignment(
                experiment_name=request.experiment_name,
                assignment_key=request.assignment_key,
            )

        assignment = VariantAssignment(
            experiment_name=request.experiment_name,
            assignment_key=request.assignment_key,
            variant=variant,
        )
        assignment = self.store.save_assignment(assignment)
        return self._assignment_to_proto(assignment)

    async def RecordOutcome(self, request, context):
        assignment = self.store.get_assignment(request.assignment_id)
        if assignment is None:
            return experiments_pb2.RecordOutcomeResponse(recorded=False)
        outcome = Outcome(
            experiment_name=request.experiment_name,
            assignment_id=request.assignment_id,
            variant_name=assignment.variant.name if assignment.variant else "",
            outcomes=dict(request.outcomes),
        )
        self.store.record_outcome(outcome)
        return experiments_pb2.RecordOutcomeResponse(recorded=True)

    async def GetExperimentResults(self, request, context):
        experiment = self.store.get_experiment(request.experiment_name)
        if experiment is None:
            return experiments_pb2.ExperimentResults(
                experiment_name=request.experiment_name,
                status="not_found",
            )
        outcomes = self.store.list_outcomes(experiment.name)

        per_variant: Dict[str, List[Outcome]] = {v.name: [] for v in experiment.variants}
        for o in outcomes:
            per_variant.setdefault(o.variant_name, []).append(o)

        variant_summaries = []
        for variant_name, variant_outcomes in per_variant.items():
            metric_summaries = []
            metric_names = set()
            for o in variant_outcomes:
                metric_names.update(o.outcomes.keys())
            for metric in sorted(metric_names):
                values = [o.outcomes[metric] for o in variant_outcomes if metric in o.outcomes]
                mean = statistics.fmean(values) if values else 0.0
                std = statistics.stdev(values) if len(values) >= 2 else 0.0
                metric_summaries.append(
                    MetricSummary(
                        metric_name=metric,
                        sample_size=len(values),
                        mean=mean,
                        std_dev=std,
                    )
                )
            variant_summaries.append(
                VariantSummary(
                    variant_name=variant_name,
                    sample_size=len(variant_outcomes),
                    metrics=metric_summaries,
                )
            )

        comparisons: List[MetricComparison] = []
        if len(per_variant) >= 2 and experiment.variants:
            baseline_name = experiment.variants[0].name
            metric_names = set()
            for vs in variant_summaries:
                for m in vs.metrics:
                    metric_names.add(m.metric_name)
            for metric in sorted(metric_names):
                baseline_values = [
                    o.outcomes[metric]
                    for o in per_variant.get(baseline_name, [])
                    if metric in o.outcomes
                ]
                best_winner = baseline_name
                best_diff = 0.0
                best_p = 1.0
                for variant_name, variant_outcomes in per_variant.items():
                    if variant_name == baseline_name:
                        continue
                    values = [o.outcomes[metric] for o in variant_outcomes if metric in o.outcomes]
                    _, p_value, diff = welch_t_test(values, baseline_values)
                    if diff > best_diff:
                        best_diff = diff
                        best_winner = variant_name
                        best_p = p_value
                comparisons.append(
                    MetricComparison(
                        metric_name=metric,
                        winner=best_winner,
                        effect_size=best_diff,
                        p_value=best_p,
                        is_significant=best_p < 0.05,
                    )
                )

        ready = (
            all(vs.sample_size >= experiment.minimum_sample_size for vs in variant_summaries)
            if variant_summaries
            else False
        )

        result = ExperimentResults(
            experiment_name=experiment.name,
            status=experiment.status,
            variant_summaries=variant_summaries,
            comparisons=comparisons,
            minimum_sample_size=experiment.minimum_sample_size,
            ready_to_conclude=ready,
        )
        return self._results_to_proto(result)

    # ------------------------------------------------------------------
    # Annotation queues
    # ------------------------------------------------------------------
    async def CreateAnnotationQueue(self, request, context):
        queue = self._proto_queue_to_domain(request.queue)
        queue = self.store.save_annotation_queue(queue)
        return self._queue_to_proto(queue)

    async def GetNextAnnotationItem(self, request, context):
        item = self.store.get_next_annotation_item(request.queue_name, request.reviewer)
        if item is None:
            return experiments_pb2.AnnotationItem(queue_name=request.queue_name)
        return self._item_to_proto(item)

    async def SubmitAnnotation(self, request, context):
        item = self.store.complete_annotation_item(request.annotation.item_id)
        if item is None:
            return experiments_pb2.SubmitAnnotationResponse(recorded=False)
        emitted = 0
        if self.observability_stub is not None and item.trace_id:
            for name, value in request.annotation.numeric_values.items():
                self._record_score(
                    Score(
                        trace_id=item.trace_id,
                        name=name,
                        value=float(value),
                        source="HUMAN",
                        comment=request.annotation.comment,
                        metadata={"reviewer": request.annotation.reviewer},
                    )
                )
                emitted += 1
            for name, value in request.annotation.boolean_values.items():
                self._record_score(
                    Score(
                        trace_id=item.trace_id,
                        name=name,
                        value=bool(value),
                        source="HUMAN",
                        comment=request.annotation.comment,
                        metadata={"reviewer": request.annotation.reviewer},
                    )
                )
                emitted += 1
            for name, value in request.annotation.categorical_values.items():
                self._record_score(
                    Score(
                        trace_id=item.trace_id,
                        name=name,
                        value=str(value),
                        source="HUMAN",
                        comment=request.annotation.comment,
                        metadata={"reviewer": request.annotation.reviewer},
                    )
                )
                emitted += 1
        return experiments_pb2.SubmitAnnotationResponse(recorded=True, scores_emitted=emitted)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _next_target_version(self, name: str) -> int:
        history = self.store.get_target_history(name)
        return (max((t.version for t in history), default=0) or 0) + 1

    def _fetch_traces(self, trace_ids) -> List[Trace]:
        if not self.observability_stub or not trace_ids:
            return []
        traces: List[Trace] = []
        for trace_id in trace_ids:
            try:
                proto = self.observability_stub.GetTrace(
                    observability_pb2.GetTraceRequest(trace_id=trace_id)
                )
                traces.append(self._proto_trace_to_domain(proto))
            except Exception as exc:  # pragma: no cover - network defence
                logger.warning("Failed to fetch trace %s: %s", trace_id, exc)
        return traces

    def _record_score(self, score: Score) -> None:
        if self.observability_stub is None:
            return
        proto_score = observability_pb2.Score(
            trace_id=score.trace_id,
            span_id=score.span_id,
            generation_id=score.generation_id,
            name=score.name,
            comment=score.comment,
            metadata=score.metadata,
        )
        if isinstance(score.value, bool):
            proto_score.boolean_value = score.value
        elif isinstance(score.value, (int, float)):
            proto_score.numeric_value = float(score.value)
        elif isinstance(score.value, str):
            proto_score.categorical_value = score.value
        source_map = {
            "AUTOMATED": observability_pb2.AUTOMATED,
            "MODEL_JUDGE": observability_pb2.MODEL_JUDGE,
            "HUMAN": observability_pb2.HUMAN,
            "USER_FEEDBACK": observability_pb2.USER_FEEDBACK,
        }
        proto_score.source = source_map.get(score.source, observability_pb2.AUTOMATED)
        try:
            self.observability_stub.RecordScore(
                observability_pb2.RecordScoreRequest(score=proto_score)
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("RecordScore failed: %s", exc)

    # ------------------------------------------------------------------
    # Proto <-> domain conversion
    # ------------------------------------------------------------------
    def _target_to_proto(self, target: ExperimentTarget):
        proto = experiments_pb2.ExperimentTarget(
            name=target.name,
            version=target.version,
            type=_TYPE_TO_PROTO.get(target.type, experiments_pb2.PROMPT),
            author=target.author,
            change_description=target.change_description,
            status=_STATUS_TO_PROTO.get(target.status, experiments_pb2.DRAFT),
            metadata=target.metadata,
        )
        if target.created_at:
            proto.created_at.CopyFrom(_dt_to_ts(target.created_at))
        if target.evaluation_summary:
            proto.evaluation_summary.overall_score = target.evaluation_summary.overall_score
            proto.evaluation_summary.evaluation_id = target.evaluation_summary.evaluation_id
            proto.evaluation_summary.test_cases_evaluated = (
                target.evaluation_summary.test_cases_evaluated
            )
            for k, v in target.evaluation_summary.metric_scores.items():
                proto.evaluation_summary.metric_scores[k] = v
        return proto

    def _proto_test_case_to_domain(self, c) -> TestCase:
        return TestCase(
            id=c.id,
            input_query=c.input_query,
            ideal_response=c.ideal_response,
            key_elements=list(c.key_elements),
            tags=list(c.tags),
            metadata=dict(c.metadata),
            source_trace_id=c.source_trace_id,
            needs_review=c.needs_review,
        )

    def _dataset_to_proto(self, dataset: Dataset):
        proto = experiments_pb2.Dataset(
            name=dataset.name,
            metadata=dataset.metadata,
        )
        if dataset.created_at:
            proto.created_at.CopyFrom(_dt_to_ts(dataset.created_at))
        if dataset.updated_at:
            proto.updated_at.CopyFrom(_dt_to_ts(dataset.updated_at))
        for c in dataset.test_cases:
            test_case = proto.test_cases.add()
            test_case.id = c.id
            test_case.input_query = c.input_query
            test_case.ideal_response = c.ideal_response
            test_case.key_elements.extend(c.key_elements)
            test_case.tags.extend(c.tags)
            for k, v in c.metadata.items():
                test_case.metadata[k] = v
            test_case.source_trace_id = c.source_trace_id
            test_case.needs_review = c.needs_review
        return proto

    def _evaluation_to_proto(self, evaluation: Evaluation):
        proto = experiments_pb2.Evaluation(
            evaluation_id=evaluation.evaluation_id,
            dataset_name=evaluation.dataset_name,
            target_ids=evaluation.target_ids,
            metrics=evaluation.metrics,
            status=evaluation.status,
        )
        if evaluation.created_at:
            proto.created_at.CopyFrom(_dt_to_ts(evaluation.created_at))
        return proto

    def _progress_to_proto(self, progress):
        proto = experiments_pb2.EvaluationProgress(
            evaluation_id=progress.evaluation_id,
            status=progress.status,
            completed_cases=progress.completed_cases,
            total_cases=progress.total_cases,
            current_target=progress.current_target,
        )
        if progress.results is not None:
            results = experiments_pb2.EvaluationResults(
                evaluation_id=progress.results.evaluation_id,
                dataset_name=progress.results.dataset_name,
            )
            for tr in progress.results.target_results:
                target_result = results.target_results.add()
                target_result.target_id = tr.target_id
                target_result.overall_score = tr.overall_score
                target_result.cases_run = tr.cases_run
                for k, v in tr.metric_scores.items():
                    target_result.metric_scores[k] = v
            for p in progress.results.per_case_results:
                per_case = results.per_case_results.add()
                per_case.target_id = p.target_id
                per_case.test_case_id = p.test_case_id
                per_case.output = p.output
                for k, v in p.metric_scores.items():
                    per_case.metric_scores[k] = v
            proto.results.CopyFrom(results)
        return proto

    def _proto_rule_to_domain(self, proto_rule) -> ScoringRule:
        return ScoringRule(
            name=proto_rule.name,
            workflow_id=proto_rule.workflow_id,
            sample_rate=proto_rule.sample_rate,
            scorers=[
                ScorerConfig(
                    name=s.name,
                    type=s.type,
                    criterion=s.criterion,
                    judge_model=s.judge_model,
                    required_elements=list(s.required_elements),
                    options=dict(s.options),
                )
                for s in proto_rule.scorers
            ],
            alerts=[
                AlertCondition(
                    scorer_name=a.scorer_name,
                    threshold_below=a.threshold_below,
                    threshold_above=a.threshold_above,
                    window=a.window,
                )
                for a in proto_rule.alerts
            ],
        )

    def _rule_to_proto(self, rule: ScoringRule):
        proto = experiments_pb2.ScoringRule(
            name=rule.name,
            workflow_id=rule.workflow_id,
            sample_rate=rule.sample_rate,
        )
        for s in rule.scorers:
            scorer = proto.scorers.add()
            scorer.name = s.name
            scorer.type = s.type
            scorer.criterion = s.criterion
            scorer.judge_model = s.judge_model
            scorer.required_elements.extend(s.required_elements)
            for k, v in s.options.items():
                scorer.options[k] = v
        for a in rule.alerts:
            alert = proto.alerts.add()
            alert.scorer_name = a.scorer_name
            alert.threshold_below = a.threshold_below
            alert.threshold_above = a.threshold_above
            alert.window = a.window
        if rule.created_at:
            proto.created_at.CopyFrom(_dt_to_ts(rule.created_at))
        return proto

    def _proto_experiment_to_domain(self, proto_exp) -> Experiment:
        variants = []
        for v in proto_exp.variants:
            variant = Variant(
                name=v.name,
                traffic_allocation=v.traffic_allocation,
                metadata=dict(v.metadata),
            )
            if v.HasField("prompt_variant"):
                variant.prompt_variant = PromptVariant(
                    prompt_name=v.prompt_variant.prompt_name,
                    version=v.prompt_variant.version,
                )
            if v.HasField("model_config_variant"):
                variant.model_config_variant = ModelConfigVariant(
                    model=v.model_config_variant.model,
                    temperature=v.model_config_variant.temperature,
                    max_tokens=v.model_config_variant.max_tokens,
                )
            if v.HasField("retrieval_config_variant"):
                variant.retrieval_config_variant = RetrievalConfigVariant(
                    index_name=v.retrieval_config_variant.index_name,
                    top_k=v.retrieval_config_variant.top_k,
                )
            variants.append(variant)
        return Experiment(
            name=proto_exp.name,
            workflow_id=proto_exp.workflow_id,
            variants=variants,
            success_metrics=list(proto_exp.success_metrics),
            minimum_sample_size=proto_exp.minimum_sample_size or 100,
            status=proto_exp.status or "draft",
        )

    def _experiment_to_proto(self, exp: Experiment):
        proto = experiments_pb2.Experiment(
            name=exp.name,
            workflow_id=exp.workflow_id,
            success_metrics=exp.success_metrics,
            minimum_sample_size=exp.minimum_sample_size,
            status=exp.status,
        )
        for v in exp.variants:
            variant = proto.variants.add()
            variant.name = v.name
            variant.traffic_allocation = v.traffic_allocation
            for k, val in v.metadata.items():
                variant.metadata[k] = val
            if v.prompt_variant:
                variant.prompt_variant.prompt_name = v.prompt_variant.prompt_name
                variant.prompt_variant.version = v.prompt_variant.version
            if v.model_config_variant:
                variant.model_config_variant.model = v.model_config_variant.model
                variant.model_config_variant.temperature = v.model_config_variant.temperature
                variant.model_config_variant.max_tokens = v.model_config_variant.max_tokens
            if v.retrieval_config_variant:
                variant.retrieval_config_variant.index_name = v.retrieval_config_variant.index_name
                variant.retrieval_config_variant.top_k = v.retrieval_config_variant.top_k
        if exp.created_at:
            proto.created_at.CopyFrom(_dt_to_ts(exp.created_at))
        return proto

    def _assignment_to_proto(self, assignment: VariantAssignment):
        proto = experiments_pb2.VariantAssignment(
            assignment_id=assignment.assignment_id,
            experiment_name=assignment.experiment_name,
            assignment_key=assignment.assignment_key,
        )
        if assignment.variant is not None:
            proto.variant.name = assignment.variant.name
            proto.variant.traffic_allocation = assignment.variant.traffic_allocation
            for k, v in assignment.variant.metadata.items():
                proto.variant.metadata[k] = v
            if assignment.variant.prompt_variant:
                proto.variant.prompt_variant.prompt_name = (
                    assignment.variant.prompt_variant.prompt_name
                )
                proto.variant.prompt_variant.version = assignment.variant.prompt_variant.version
            if assignment.variant.model_config_variant:
                proto.variant.model_config_variant.model = (
                    assignment.variant.model_config_variant.model
                )
                proto.variant.model_config_variant.temperature = (
                    assignment.variant.model_config_variant.temperature
                )
                proto.variant.model_config_variant.max_tokens = (
                    assignment.variant.model_config_variant.max_tokens
                )
            if assignment.variant.retrieval_config_variant:
                proto.variant.retrieval_config_variant.index_name = (
                    assignment.variant.retrieval_config_variant.index_name
                )
                proto.variant.retrieval_config_variant.top_k = (
                    assignment.variant.retrieval_config_variant.top_k
                )
        if assignment.assigned_at:
            proto.assigned_at.CopyFrom(_dt_to_ts(assignment.assigned_at))
        return proto

    def _results_to_proto(self, result: ExperimentResults):
        proto = experiments_pb2.ExperimentResults(
            experiment_name=result.experiment_name,
            status=result.status,
            minimum_sample_size=result.minimum_sample_size,
            ready_to_conclude=result.ready_to_conclude,
        )
        for vs in result.variant_summaries:
            variant_summary = proto.variant_summaries.add()
            variant_summary.variant_name = vs.variant_name
            variant_summary.sample_size = vs.sample_size
            for ms in vs.metrics:
                metric = variant_summary.metrics.add()
                metric.metric_name = ms.metric_name
                metric.sample_size = ms.sample_size
                metric.mean = ms.mean
                metric.std_dev = ms.std_dev
        for c in result.comparisons:
            comparison = proto.comparisons.add()
            comparison.metric_name = c.metric_name
            comparison.winner = c.winner
            comparison.effect_size = c.effect_size
            comparison.p_value = c.p_value
            comparison.is_significant = c.is_significant
        return proto

    def _proto_queue_to_domain(self, proto_queue) -> AnnotationQueue:
        return AnnotationQueue(
            name=proto_queue.name,
            workflow_id=proto_queue.workflow_id,
            rubric=[
                RubricItem(
                    name=item.name,
                    type=item.type,
                    options=list(item.options),
                    min=item.min,
                    max=item.max,
                    description=item.description,
                )
                for item in proto_queue.rubric
            ],
            routing_rules=[
                RoutingRule(condition=r.condition, source=r.source, rate=r.rate)
                for r in proto_queue.routing_rules
            ],
            reviewers=list(proto_queue.reviewers),
        )

    def _queue_to_proto(self, queue: AnnotationQueue):
        proto = experiments_pb2.AnnotationQueue(
            name=queue.name,
            workflow_id=queue.workflow_id,
            reviewers=queue.reviewers,
        )
        for item in queue.rubric:
            rubric = proto.rubric.add()
            rubric.name = item.name
            rubric.type = item.type
            rubric.options.extend(item.options)
            rubric.min = item.min
            rubric.max = item.max
            rubric.description = item.description
        for r in queue.routing_rules:
            rule = proto.routing_rules.add()
            rule.condition = r.condition
            rule.source = r.source
            rule.rate = r.rate
        if queue.created_at:
            proto.created_at.CopyFrom(_dt_to_ts(queue.created_at))
        return proto

    def _item_to_proto(self, item: AnnotationItem):
        proto = experiments_pb2.AnnotationItem(
            item_id=item.item_id,
            queue_name=item.queue_name,
            trace_id=item.trace_id,
            status=item.status,
            assigned_to=item.assigned_to,
        )
        if item.added_at:
            proto.added_at.CopyFrom(_dt_to_ts(item.added_at))
        return proto

    def _proto_trace_to_domain(self, proto_trace) -> Trace:
        from services.observability.service import ObservabilityServiceImpl

        helper = ObservabilityServiceImpl()
        spans = [helper._proto_span_to_domain(span) for span in proto_trace.spans]
        gens = [helper._proto_generation_to_domain(gen) for gen in proto_trace.generations]
        scores = [helper._proto_score_to_domain(score) for score in proto_trace.scores]
        return Trace(
            trace_id=proto_trace.trace_id,
            session_id=proto_trace.session_id,
            workflow_id=proto_trace.workflow_id,
            user_id=proto_trace.user_id,
            spans=spans,
            generations=gens,
            input=proto_trace.input,
            output=proto_trace.output,
            total_duration_ms=proto_trace.total_duration_ms,
            total_cost_usd=proto_trace.total_cost_usd,
            total_tokens=proto_trace.total_tokens,
            scores=scores,
            tags=list(proto_trace.tags),
        )
