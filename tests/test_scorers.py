"""Unit tests for the experimentation scorers (Listing 7.12)."""

from dataclasses import dataclass

from services.experiments.scorers import (
    KeyElementScorer,
    LLMJudgeScorer,
    RetrievalRelevanceScorer,
    build_scorer,
)
from services.observability.models import Span, Trace


@dataclass
class _FakeChatResponse:
    content: str


class _FakeModelService:
    def __init__(self, response: str):
        self._response = response

    def invoke(self, **_):
        return _FakeChatResponse(content=self._response)


class TestKeyElementScorer:
    def test_full_match(self):
        trace = Trace(
            trace_id="t",
            input="What documents?",
            output="Bring your insurance card and your photo ID.",
        )
        score = KeyElementScorer(
            name="key_elements", required_elements=["insurance", "photo", "id"]
        ).score(trace)
        assert score.value == 1.0
        assert score.source == "AUTOMATED"

    def test_partial_match_case_insensitive(self):
        trace = Trace(trace_id="t", output="Photo ID is required")
        score = KeyElementScorer(name="key_elements", required_elements=["INSURANCE", "id"]).score(
            trace
        )
        assert score.value == 0.5

    def test_no_required_elements_returns_one(self):
        trace = Trace(trace_id="t", output="anything")
        score = KeyElementScorer(name="ok", required_elements=[]).score(trace)
        assert score.value == 1.0


class TestLLMJudgeScorer:
    def test_parses_numeric_score(self):
        scorer = LLMJudgeScorer(
            name="helpfulness",
            criterion="helpfulness",
            judge_model="claude-haiku-4-5",
            model_service=_FakeModelService("0.78"),
        )
        score = scorer.score(Trace(trace_id="t", output="answer"))
        assert score.value == 0.78
        assert score.source == "MODEL_JUDGE"
        assert score.metadata.get("judge_model") == "claude-haiku-4-5"

    def test_parses_score_in_text(self):
        scorer = LLMJudgeScorer(
            name="x",
            criterion="x",
            model_service=_FakeModelService("Score is 0.42 because it covers half the points"),
        )
        score = scorer.score(Trace(trace_id="t"))
        assert score.value == 0.42

    def test_falls_back_to_zero_for_garbage_output(self):
        scorer = LLMJudgeScorer(
            name="x", criterion="x", model_service=_FakeModelService("not numeric")
        )
        score = scorer.score(Trace(trace_id="t"))
        assert score.value == 0.0


class TestRetrievalRelevanceScorer:
    def test_pulls_from_data_span(self):
        trace = Trace(
            trace_id="t",
            spans=[Span(service="data", numeric_attributes={"top_relevance_score": 0.91})],
        )
        score = RetrievalRelevanceScorer(name="retrieval_relevance").score(trace)
        assert score.value == 0.91


class TestBuildScorer:
    def test_unknown_type_raises(self):
        from services.experiments.models import ScorerConfig

        try:
            build_scorer(ScorerConfig(name="x", type="unknown"))
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for unknown scorer type")

    def test_key_elements_factory(self):
        from services.experiments.models import ScorerConfig

        scorer = build_scorer(
            ScorerConfig(name="key", type="key_elements", required_elements=["a"])
        )
        assert isinstance(scorer, KeyElementScorer)
        assert scorer.required_elements == ["a"]
