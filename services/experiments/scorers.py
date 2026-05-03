"""
Scorers — automated and LLM-as-judge implementations.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 7.12: KeyElementScorer + LLMJudgeScorer
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional

from services.observability.models import Score, Trace

logger = logging.getLogger(__name__)


@dataclass
class Scorer(ABC):
    name: str = ""

    @abstractmethod
    def score(self, trace: Trace) -> Score: ...


@dataclass
class KeyElementScorer(Scorer):
    """Checks factual completeness without any model call (Listing 7.12)."""

    required_elements: List[str] = field(default_factory=list)

    def score(self, trace: Trace) -> Score:
        output = (trace.output or "").lower()
        if not self.required_elements:
            ratio = 1.0
        else:
            found = sum(1 for e in self.required_elements if e.lower() in output)
            ratio = found / len(self.required_elements)
        return Score(
            trace_id=trace.trace_id,
            name=self.name,
            value=ratio,
            source="AUTOMATED",
        )


@dataclass
class LLMJudgeScorer(Scorer):
    """Uses a model to evaluate another model's output (Listing 7.12)."""

    criterion: str = ""
    judge_model: str = "claude-haiku-4-5"
    model_service: Any = None

    def score(self, trace: Trace) -> Score:
        if self.model_service is None:
            raise ValueError("LLMJudgeScorer requires a model_service handle")
        result = self.model_service.invoke(
            model_name=self.judge_model,
            system_prompt=(
                "You are an expert evaluator. Return only a numeric score "
                "between 0.0 and 1.0 with no other text."
            ),
            query=(
                f"Evaluate on: {self.criterion}\n"
                f"Question: {trace.input}\n"
                f"Response: {trace.output}\n"
                "Score 0.0 to 1.0:"
            ),
        )
        value = _parse_first_float(getattr(result, "content", str(result)))
        return Score(
            trace_id=trace.trace_id,
            name=self.name,
            value=value,
            source="MODEL_JUDGE",
            metadata={"judge_model": self.judge_model},
        )


@dataclass
class RetrievalRelevanceScorer(Scorer):
    """Pulls retrieval relevance from the Data Service span attributes."""

    def score(self, trace: Trace) -> Score:
        relevance: Optional[float] = None
        for span in trace.spans:
            if "data" not in span.service:
                continue
            for key in ("top_relevance_score", "relevance_score", "max_score"):
                raw = span.attributes.get(key)
                if raw is not None:
                    try:
                        relevance = float(raw)
                        break
                    except ValueError:
                        continue
            if relevance is None:
                raw = span.numeric_attributes.get(
                    "top_relevance_score"
                ) or span.numeric_attributes.get("relevance_score")
                if raw is not None:
                    relevance = float(raw)
            if relevance is not None:
                break
        return Score(
            trace_id=trace.trace_id,
            name=self.name,
            value=relevance if relevance is not None else 0.0,
            source="AUTOMATED",
        )


def _parse_first_float(text: str) -> float:
    match = re.search(r"[-+]?\d*\.?\d+", text or "")
    if not match:
        return 0.0
    try:
        return float(match.group())
    except ValueError:
        return 0.0


def build_scorer(config) -> Scorer:
    """Construct a Scorer from a ScorerConfig dataclass.

    `config` must expose ``name``, ``type``, ``criterion``, ``judge_model``,
    and ``required_elements`` attributes (the ScorerConfig dataclass does).
    """
    kind = (config.type or "automated").lower()
    if kind in {"automated", "key_elements"}:
        return KeyElementScorer(
            name=config.name,
            required_elements=list(config.required_elements or []),
        )
    if kind == "llm_judge":
        return LLMJudgeScorer(
            name=config.name,
            criterion=config.criterion,
            judge_model=config.judge_model or "claude-haiku-4-5",
        )
    if kind == "retrieval_relevance":
        return RetrievalRelevanceScorer(name=config.name)
    raise ValueError(f"Unknown scorer type: {config.type}")
