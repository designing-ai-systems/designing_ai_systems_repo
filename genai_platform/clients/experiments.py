"""
Experimentation Service client.

Surfaces the Experimentation Service through ``platform.experiments``.
The client returns rich proto messages because the experiment workflow
is heavily message-driven (Listings 7.16, 7.17, 7.18, 7.19, 7.20); the
calling code in :mod:`examples.quickstart_experiments` shows the
patterns users will follow.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 7.14: ExperimentationService gRPC contract
  - Listing 7.16: register_target + run_evaluation
  - Listing 7.17: dataset from production
  - Listing 7.18: scoring rule
  - Listing 7.19: annotation queue
  - Listing 7.20: experiment-aware workflow
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional

from proto import experiments_pb2, experiments_pb2_grpc

from .base import BaseClient

_TARGET_TYPES = {
    "PROMPT": experiments_pb2.PROMPT,
    "MODEL_CONFIG": experiments_pb2.MODEL_CONFIG,
    "RETRIEVAL_CONFIG": experiments_pb2.RETRIEVAL_CONFIG,
}


class ExperimentsClient(BaseClient):
    """SDK client for the Experimentation Service."""

    def __init__(self, platform):
        super().__init__(platform, "experiments")
        self._stub = experiments_pb2_grpc.ExperimentationServiceStub(self._channel)

    # ------------------------------------------------------------------
    # Targets — Listing 7.16
    # ------------------------------------------------------------------
    def register_target(
        self,
        *,
        name: str,
        version: int,
        target_type: str = "PROMPT",
        change_description: str = "",
        author: str = "",
        metadata: Optional[Dict[str, str]] = None,
    ):
        request = experiments_pb2.RegisterTargetRequest(
            name=name,
            version=version,
            type=_TARGET_TYPES.get(target_type.upper(), experiments_pb2.PROMPT),
            change_description=change_description,
            author=author,
            metadata=metadata or {},
        )
        return self._stub.RegisterTarget(request, metadata=self.metadata)

    def get_target_history(self, name: str):
        return self._stub.GetTargetHistory(
            experiments_pb2.GetTargetHistoryRequest(name=name), metadata=self.metadata
        )

    def compare_targets(self, target_ids: List[str]):
        return self._stub.CompareTargets(
            experiments_pb2.CompareTargetsRequest(target_ids=target_ids),
            metadata=self.metadata,
        )

    # ------------------------------------------------------------------
    # Datasets — Listing 7.17
    # ------------------------------------------------------------------
    def create_dataset(
        self,
        *,
        name: str,
        test_cases: Optional[Iterable[Mapping[str, Any]]] = None,
        metadata: Optional[Dict[str, str]] = None,
    ):
        request = experiments_pb2.CreateDatasetRequest(
            name=name,
            metadata=metadata or {},
        )
        for case in test_cases or []:
            self._populate_test_case(request.test_cases.add(), case)
        return self._stub.CreateDataset(request, metadata=self.metadata)

    def add_test_cases(self, *, name: str, test_cases: Iterable[Mapping[str, Any]]):
        request = experiments_pb2.AddTestCasesRequest(name=name)
        for case in test_cases:
            self._populate_test_case(request.test_cases.add(), case)
        return self._stub.AddTestCases(request, metadata=self.metadata)

    def add_from_production(
        self,
        *,
        dataset_name: str,
        trace_ids: Iterable[str],
        require_human_review: bool = False,
    ):
        request = experiments_pb2.AddFromProductionRequest(
            dataset_name=dataset_name,
            trace_ids=list(trace_ids),
            require_human_review=require_human_review,
        )
        return self._stub.AddFromProduction(request, metadata=self.metadata)

    # ------------------------------------------------------------------
    # Evaluation — Listing 7.16
    # ------------------------------------------------------------------
    def create_evaluation(
        self,
        *,
        dataset_name: str,
        target_ids: Iterable[str],
        metrics: Iterable[str],
    ):
        request = experiments_pb2.CreateEvaluationRequest(
            dataset_name=dataset_name,
            target_ids=list(target_ids),
            metrics=list(metrics),
        )
        return self._stub.CreateEvaluation(request, metadata=self.metadata)

    def run_evaluation(
        self,
        *,
        dataset_name: str,
        targets: Iterable[str],
        metrics: Iterable[str],
        repeats_per_case: int = 1,
    ):
        request = experiments_pb2.RunEvaluationRequest(
            dataset_name=dataset_name,
            target_ids=list(targets),
            metrics=list(metrics),
            repeats_per_case=repeats_per_case,
        )
        # Returns the streamed iterator so callers can observe progress;
        # the final message in the stream carries the populated results.
        return self._stub.RunEvaluation(request, metadata=self.metadata)

    def get_evaluation_results(self, evaluation_id: str):
        return self._stub.GetEvaluationResults(
            experiments_pb2.GetEvaluationResultsRequest(evaluation_id=evaluation_id),
            metadata=self.metadata,
        )

    # ------------------------------------------------------------------
    # Online scoring — Listing 7.18
    # ------------------------------------------------------------------
    def create_scoring_rule(
        self,
        *,
        name: str,
        workflow_id: str,
        sample_rate: float = 1.0,
        scorers: Optional[Iterable[Mapping[str, Any]]] = None,
        alert_on: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ):
        rule = experiments_pb2.ScoringRule(
            name=name,
            workflow_id=workflow_id,
            sample_rate=sample_rate,
        )
        for scorer in scorers or []:
            proto_scorer = rule.scorers.add()
            proto_scorer.name = scorer.get("name", "")
            proto_scorer.type = scorer.get("type", "automated")
            proto_scorer.criterion = scorer.get("criterion", "")
            proto_scorer.judge_model = scorer.get("judge_model", "")
            proto_scorer.required_elements.extend(scorer.get("required_elements", []))
            for k, v in (scorer.get("options") or {}).items():
                proto_scorer.options[k] = str(v)
        for scorer_name, condition in (alert_on or {}).items():
            alert = rule.alerts.add()
            alert.scorer_name = scorer_name
            alert.threshold_below = float(condition.get("below", 0.0))
            alert.threshold_above = float(condition.get("above", 0.0))
            alert.window = str(condition.get("window", "1h"))
        request = experiments_pb2.CreateScoringRuleRequest(rule=rule)
        return self._stub.CreateScoringRule(request, metadata=self.metadata)

    def list_scoring_rules(self, workflow_id: str = ""):
        return self._stub.ListScoringRules(
            experiments_pb2.ListScoringRulesRequest(workflow_id=workflow_id),
            metadata=self.metadata,
        )

    def run_scoring_rule(self, *, rule_name: str, trace_ids: Iterable[str]):
        return self._stub.RunScoringRule(
            experiments_pb2.RunScoringRuleRequest(rule_name=rule_name, trace_ids=list(trace_ids)),
            metadata=self.metadata,
        )

    # ------------------------------------------------------------------
    # A/B testing — Listing 7.20
    # ------------------------------------------------------------------
    def create_experiment(
        self,
        *,
        name: str,
        workflow_id: str,
        variants: Iterable[Mapping[str, Any]],
        success_metrics: Optional[Iterable[str]] = None,
        minimum_sample_size: int = 100,
    ):
        experiment = experiments_pb2.Experiment(
            name=name,
            workflow_id=workflow_id,
            success_metrics=list(success_metrics or []),
            minimum_sample_size=minimum_sample_size,
            status="running",
        )
        for variant in variants:
            self._populate_variant(experiment.variants.add(), variant)
        return self._stub.CreateExperiment(
            experiments_pb2.CreateExperimentRequest(experiment=experiment),
            metadata=self.metadata,
        )

    def assign_variant(
        self,
        *,
        experiment_name: str,
        assignment_key: str,
        workflow_id: str = "",
    ):
        response = self._stub.AssignVariant(
            experiments_pb2.AssignVariantRequest(
                experiment_name=experiment_name,
                assignment_key=assignment_key,
                workflow_id=workflow_id,
            ),
            metadata=self.metadata,
        )
        if not response.assignment_id:
            return None
        return response

    def record_outcome(
        self,
        *,
        experiment_name: str,
        assignment_id: str,
        outcomes: Mapping[str, float],
    ):
        return self._stub.RecordOutcome(
            experiments_pb2.RecordOutcomeRequest(
                experiment_name=experiment_name,
                assignment_id=assignment_id,
                outcomes={k: float(v) for k, v in outcomes.items()},
            ),
            metadata=self.metadata,
        )

    def get_experiment_results(self, experiment_name: str):
        return self._stub.GetExperimentResults(
            experiments_pb2.GetExperimentResultsRequest(experiment_name=experiment_name),
            metadata=self.metadata,
        )

    # ------------------------------------------------------------------
    # Annotation queues — Listing 7.19
    # ------------------------------------------------------------------
    def create_annotation_queue(
        self,
        *,
        name: str,
        workflow_id: str,
        rubric: Iterable[Mapping[str, Any]],
        routing_rules: Optional[Iterable[Mapping[str, Any]]] = None,
        reviewers: Optional[Iterable[str]] = None,
    ):
        queue = experiments_pb2.AnnotationQueue(
            name=name,
            workflow_id=workflow_id,
            reviewers=list(reviewers or []),
        )
        for item in rubric:
            rubric_proto = queue.rubric.add()
            rubric_proto.name = item.get("name", "")
            rubric_proto.type = item.get("type", "numeric")
            rubric_proto.options.extend(item.get("options", []))
            rubric_proto.min = float(item.get("min", 0.0))
            rubric_proto.max = float(item.get("max", 1.0))
            rubric_proto.description = item.get("description", "")
        for rule in routing_rules or []:
            rule_proto = queue.routing_rules.add()
            rule_proto.condition = rule.get("condition", "")
            rule_proto.source = rule.get("source", "")
            rule_proto.rate = float(rule.get("rate", 0.0))
        return self._stub.CreateAnnotationQueue(
            experiments_pb2.CreateAnnotationQueueRequest(queue=queue),
            metadata=self.metadata,
        )

    def get_next_annotation_item(self, *, queue_name: str, reviewer: str):
        return self._stub.GetNextAnnotationItem(
            experiments_pb2.GetNextAnnotationItemRequest(queue_name=queue_name, reviewer=reviewer),
            metadata=self.metadata,
        )

    def submit_annotation(
        self,
        *,
        item_id: str,
        queue_name: str,
        reviewer: str,
        numeric_values: Optional[Mapping[str, float]] = None,
        categorical_values: Optional[Mapping[str, str]] = None,
        boolean_values: Optional[Mapping[str, bool]] = None,
        comment: str = "",
    ):
        annotation = experiments_pb2.Annotation(
            item_id=item_id,
            queue_name=queue_name,
            reviewer=reviewer,
            numeric_values={k: float(v) for k, v in (numeric_values or {}).items()},
            categorical_values=dict(categorical_values or {}),
            boolean_values=dict(boolean_values or {}),
            comment=comment,
        )
        return self._stub.SubmitAnnotation(
            experiments_pb2.SubmitAnnotationRequest(annotation=annotation),
            metadata=self.metadata,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _populate_test_case(self, proto, case: Mapping[str, Any]) -> None:
        proto.id = case.get("id", "")
        proto.input_query = case.get("input_query", "")
        proto.ideal_response = case.get("ideal_response", "")
        proto.key_elements.extend(case.get("key_elements", []))
        proto.tags.extend(case.get("tags", []))
        for k, v in (case.get("metadata") or {}).items():
            proto.metadata[k] = str(v)
        proto.source_trace_id = case.get("source_trace_id", "")
        proto.needs_review = bool(case.get("needs_review", False))

    def _populate_variant(self, proto, variant: Mapping[str, Any]) -> None:
        proto.name = variant.get("name", "")
        proto.traffic_allocation = float(variant.get("traffic_allocation", 0.0))
        for k, v in (variant.get("metadata") or {}).items():
            proto.metadata[k] = str(v)
        prompt = variant.get("prompt_variant")
        if prompt:
            proto.prompt_variant.prompt_name = prompt.get("prompt_name", "")
            proto.prompt_variant.version = int(prompt.get("version", 0))
        model = variant.get("model_config_variant")
        if model:
            proto.model_config_variant.model = model.get("model", "")
            proto.model_config_variant.temperature = float(model.get("temperature", 0.7))
            proto.model_config_variant.max_tokens = int(model.get("max_tokens", 0))
        retrieval = variant.get("retrieval_config_variant")
        if retrieval:
            proto.retrieval_config_variant.index_name = retrieval.get("index_name", "")
            proto.retrieval_config_variant.top_k = int(retrieval.get("top_k", 5))
