"""
Observability Service domain models.

Plain Python dataclasses for the observability primitives. The gRPC
servicer translates between these and the proto messages at the
boundary; everything internal to the service uses domain types.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 7.2: LogEvent
  - Listing 7.5: Span / Generation / Trace
  - Listing 7.11: Score / ScoreSource
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Listing 7.2 — structured log events
# ---------------------------------------------------------------------------


@dataclass
class LogEvent:
    event_id: str = ""
    trace_id: str = ""
    span_id: str = ""
    timestamp: datetime = field(default_factory=_utcnow)
    service: str = ""
    severity: str = "INFO"
    event_type: str = ""
    message: str = ""
    attributes: Dict[str, str] = field(default_factory=dict)
    numeric_attributes: Dict[str, float] = field(default_factory=dict)
    workflow_id: str = ""
    user_id: str = ""


# ---------------------------------------------------------------------------
# Listing 7.5 — spans, generations, traces
# ---------------------------------------------------------------------------


@dataclass
class SpanEvent:
    name: str = ""
    timestamp: datetime = field(default_factory=_utcnow)
    attributes: Dict[str, str] = field(default_factory=dict)


@dataclass
class Span:
    trace_id: str = ""
    span_id: str = ""
    parent_span_id: str = ""
    service: str = ""
    operation: str = ""
    start_time: datetime = field(default_factory=_utcnow)
    end_time: Optional[datetime] = None
    status: str = "OK"  # OK | ERROR
    error_message: str = ""
    attributes: Dict[str, str] = field(default_factory=dict)
    numeric_attributes: Dict[str, float] = field(default_factory=dict)
    events: List[SpanEvent] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time).total_seconds() * 1000.0


@dataclass
class Generation:
    span: Span = field(default_factory=Span)
    model: str = ""
    provider: str = ""
    requested_model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    cache_hit: bool = False
    fallback_used: bool = False
    time_to_first_token_ms: float = 0.0


# ---------------------------------------------------------------------------
# Listing 7.11 — scores
# ---------------------------------------------------------------------------


SCORE_SOURCES = {"AUTOMATED", "MODEL_JUDGE", "HUMAN", "USER_FEEDBACK"}


@dataclass
class Score:
    score_id: str = ""
    trace_id: str = ""
    span_id: str = ""
    generation_id: str = ""
    name: str = ""
    value: Union[float, str, bool, None] = None
    source: str = "AUTOMATED"
    comment: str = ""
    metadata: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# Trace assembly view
# ---------------------------------------------------------------------------


@dataclass
class Trace:
    trace_id: str = ""
    session_id: str = ""
    workflow_id: str = ""
    user_id: str = ""
    spans: List[Span] = field(default_factory=list)
    generations: List[Generation] = field(default_factory=list)
    input: str = ""
    output: str = ""
    total_duration_ms: float = 0.0
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    scores: List[Score] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class MetricRecord:
    name: str = ""
    type: str = "COUNTER"  # COUNTER | HISTOGRAM
    value: float = 0.0
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# Cost & budgets
# ---------------------------------------------------------------------------


@dataclass
class CostBucket:
    dimensions: Dict[str, str] = field(default_factory=dict)
    cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    request_count: int = 0


@dataclass
class CostReport:
    start_time: datetime = field(default_factory=_utcnow)
    end_time: datetime = field(default_factory=_utcnow)
    group_by: List[str] = field(default_factory=list)
    buckets: List[CostBucket] = field(default_factory=list)
    total_cost_usd: float = 0.0


@dataclass
class BudgetAlert:
    name: str = ""
    scope_type: str = "team"  # team | workflow | application
    scope_value: str = ""
    limit_usd: float = 0.0
    period: str = "monthly"
    thresholds: List[float] = field(default_factory=lambda: [0.7, 0.9, 1.0])
    notification_channels: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=_utcnow)


@dataclass
class BudgetStatus:
    alert: BudgetAlert = field(default_factory=BudgetAlert)
    current_spend_usd: float = 0.0
    projected_spend_usd: float = 0.0
    percent_used: float = 0.0
    thresholds_crossed: List[float] = field(default_factory=list)


@dataclass
class ServiceHealth:
    service: str = ""
    status: str = "unknown"  # healthy | degraded | unknown
    last_span_at: Optional[datetime] = None
    span_count: int = 0
    error_rate: float = 0.0
    detail: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def assemble_trace(
    trace_id: str,
    spans: List[Span],
    generations: List[Generation],
    scores: List[Score],
) -> Trace:
    """Aggregate the spans, generations, and scores for a single trace_id."""
    if not spans and not generations:
        return Trace(trace_id=trace_id, scores=list(scores))

    starts: List[datetime] = [s.start_time for s in spans if s.start_time]
    ends: List[datetime] = [s.end_time for s in spans if s.end_time is not None]
    starts.extend(g.span.start_time for g in generations if g.span.start_time)
    ends.extend(g.span.end_time for g in generations if g.span.end_time is not None)

    total_duration_ms = 0.0
    if starts and ends:
        total_duration_ms = (max(ends) - min(starts)).total_seconds() * 1000.0

    total_cost_usd = sum(g.cost_usd for g in generations)
    total_tokens = sum(g.prompt_tokens + g.completion_tokens for g in generations)

    workflow_id = ""
    user_id = ""
    session_id = ""
    for s in spans:
        workflow_id = workflow_id or s.attributes.get("workflow_id", "")
        user_id = user_id or s.attributes.get("user_id", "")
        session_id = session_id or s.attributes.get("session_id", "")

    return Trace(
        trace_id=trace_id,
        session_id=session_id,
        workflow_id=workflow_id,
        user_id=user_id,
        spans=list(spans),
        generations=list(generations),
        total_duration_ms=total_duration_ms,
        total_cost_usd=total_cost_usd,
        total_tokens=total_tokens,
        scores=list(scores),
    )
