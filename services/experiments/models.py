"""
Experimentation Service domain models.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 7.15: ExperimentTarget / EvaluationSummary / Target lifecycle
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Listing 7.15 — target lifecycle
# ---------------------------------------------------------------------------


TARGET_TYPES = {"PROMPT", "MODEL_CONFIG", "RETRIEVAL_CONFIG"}
TARGET_STATUSES = {"DRAFT", "TESTING", "ACTIVE", "DEPRECATED"}


@dataclass
class EvaluationSummary:
    overall_score: float = 0.0
    metric_scores: Dict[str, float] = field(default_factory=dict)
    evaluation_id: str = ""
    test_cases_evaluated: int = 0


@dataclass
class ExperimentTarget:
    name: str = ""
    version: int = 1
    type: str = "PROMPT"
    author: str = ""
    change_description: str = ""
    created_at: datetime = field(default_factory=_utcnow)
    status: str = "DRAFT"
    evaluation_summary: EvaluationSummary = field(default_factory=EvaluationSummary)
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def target_id(self) -> str:
        return f"{self.name}:{self.version}"


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


@dataclass
class TestCase:
    __test__ = False  # Tell pytest not to treat this dataclass as a test class.
    id: str = ""
    input_query: str = ""
    ideal_response: str = ""
    key_elements: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)
    source_trace_id: str = ""
    needs_review: bool = False


@dataclass
class Dataset:
    name: str = ""
    test_cases: List[TestCase] = field(default_factory=list)
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)
    metadata: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Evaluations
# ---------------------------------------------------------------------------


@dataclass
class Evaluation:
    evaluation_id: str = ""
    dataset_name: str = ""
    target_ids: List[str] = field(default_factory=list)
    metrics: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=_utcnow)
    status: str = "pending"  # pending | running | completed | failed


@dataclass
class PerCaseResult:
    target_id: str = ""
    test_case_id: str = ""
    output: str = ""
    metric_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class TargetResult:
    target_id: str = ""
    metric_scores: Dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0
    cases_run: int = 0


@dataclass
class EvaluationResults:
    evaluation_id: str = ""
    dataset_name: str = ""
    target_results: List[TargetResult] = field(default_factory=list)
    per_case_results: List[PerCaseResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Online scoring rules
# ---------------------------------------------------------------------------


@dataclass
class ScorerConfig:
    name: str = ""
    type: str = "automated"  # automated | llm_judge | key_elements | retrieval_relevance
    criterion: str = ""
    judge_model: str = ""
    required_elements: List[str] = field(default_factory=list)
    options: Dict[str, str] = field(default_factory=dict)


@dataclass
class AlertCondition:
    scorer_name: str = ""
    threshold_below: float = 0.0
    threshold_above: float = 0.0
    window: str = "1h"


@dataclass
class ScoringRule:
    name: str = ""
    workflow_id: str = ""
    sample_rate: float = 1.0
    scorers: List[ScorerConfig] = field(default_factory=list)
    alerts: List[AlertCondition] = field(default_factory=list)
    created_at: datetime = field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# A/B testing
# ---------------------------------------------------------------------------


@dataclass
class PromptVariant:
    prompt_name: str = ""
    version: int = 0


@dataclass
class ModelConfigVariant:
    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 0


@dataclass
class RetrievalConfigVariant:
    index_name: str = ""
    top_k: int = 5


@dataclass
class Variant:
    name: str = ""
    traffic_allocation: float = 0.0
    prompt_variant: Optional[PromptVariant] = None
    model_config_variant: Optional[ModelConfigVariant] = None
    retrieval_config_variant: Optional[RetrievalConfigVariant] = None
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class Experiment:
    name: str = ""
    workflow_id: str = ""
    variants: List[Variant] = field(default_factory=list)
    success_metrics: List[str] = field(default_factory=list)
    minimum_sample_size: int = 100
    created_at: datetime = field(default_factory=_utcnow)
    status: str = "draft"  # draft | running | concluded


@dataclass
class VariantAssignment:
    assignment_id: str = ""
    experiment_name: str = ""
    assignment_key: str = ""
    variant: Optional[Variant] = None
    assigned_at: datetime = field(default_factory=_utcnow)


@dataclass
class Outcome:
    experiment_name: str = ""
    assignment_id: str = ""
    variant_name: str = ""
    outcomes: Dict[str, float] = field(default_factory=dict)
    recorded_at: datetime = field(default_factory=_utcnow)


@dataclass
class MetricSummary:
    metric_name: str = ""
    sample_size: int = 0
    mean: float = 0.0
    std_dev: float = 0.0


@dataclass
class VariantSummary:
    variant_name: str = ""
    sample_size: int = 0
    metrics: List[MetricSummary] = field(default_factory=list)


@dataclass
class MetricComparison:
    metric_name: str = ""
    winner: str = ""
    effect_size: float = 0.0
    p_value: float = 1.0
    is_significant: bool = False


@dataclass
class ExperimentResults:
    experiment_name: str = ""
    status: str = ""
    variant_summaries: List[VariantSummary] = field(default_factory=list)
    comparisons: List[MetricComparison] = field(default_factory=list)
    minimum_sample_size: int = 0
    ready_to_conclude: bool = False


# ---------------------------------------------------------------------------
# Annotation queues
# ---------------------------------------------------------------------------


@dataclass
class RubricItem:
    name: str = ""
    type: str = "numeric"  # numeric | categorical | boolean
    options: List[str] = field(default_factory=list)
    min: float = 0.0
    max: float = 1.0
    description: str = ""


@dataclass
class RoutingRule:
    condition: str = ""
    source: str = ""
    rate: float = 0.0


@dataclass
class AnnotationQueue:
    name: str = ""
    workflow_id: str = ""
    rubric: List[RubricItem] = field(default_factory=list)
    routing_rules: List[RoutingRule] = field(default_factory=list)
    reviewers: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=_utcnow)


@dataclass
class AnnotationItem:
    item_id: str = ""
    queue_name: str = ""
    trace_id: str = ""
    status: str = "pending"  # pending | completed
    assigned_to: str = ""
    added_at: datetime = field(default_factory=_utcnow)


@dataclass
class Annotation:
    item_id: str = ""
    queue_name: str = ""
    reviewer: str = ""
    categorical_values: Dict[str, str] = field(default_factory=dict)
    numeric_values: Dict[str, float] = field(default_factory=dict)
    boolean_values: Dict[str, bool] = field(default_factory=dict)
    comment: str = ""
    submitted_at: datetime = field(default_factory=_utcnow)
