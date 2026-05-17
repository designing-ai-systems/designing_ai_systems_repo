"""Tests for the Claw chapter-9 reference application.

These tests cover the pure helpers (tool-argument validation, context
assembly within a token budget, tool dispatch) and the agent-loop
terminator. The platform-touching tests monkey-patch the module-level
``platform`` instance with light fakes — the real gRPC clients aren't
exercised here.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from claw import claw

# ---------------------------------------------------------------------------
# Pure-helper tests (no platform interaction)
# ---------------------------------------------------------------------------


class TestCheckToolArgumentsCalendar:
    def test_blocks_past_start_time(self):
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        allowed, reason = claw._check_tool_arguments(
            "s-1",
            "claw.calendar.create_event",
            {"start_time": past, "end_time": past, "title": "x"},
        )
        assert allowed is False
        assert "future" in reason

    def test_allows_future_start_time(self):
        future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        allowed, _ = claw._check_tool_arguments(
            "s-1",
            "claw.calendar.create_event",
            {"start_time": future, "end_time": future, "title": "x"},
        )
        assert allowed is True

    def test_blocks_invalid_start_time(self):
        allowed, reason = claw._check_tool_arguments(
            "s-1",
            "claw.calendar.create_event",
            {"start_time": "not-a-timestamp", "title": "x"},
        )
        assert allowed is False
        assert "ISO" in reason

    def test_blocks_too_many_attendees(self):
        future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        allowed, reason = claw._check_tool_arguments(
            "s-1",
            "claw.calendar.create_event",
            {
                "start_time": future,
                "title": "x",
                "attendees": [f"u{i}@example.com" for i in range(21)],
            },
        )
        assert allowed is False
        assert "20 or fewer" in reason


class TestCheckToolArgumentsTickets:
    def setup_method(self):
        # The action counter dict is module-level. Reset before each test
        # so test ordering doesn't leak between cases.
        claw._action_counters.clear()

    def test_allows_first_ten_ticket_creations(self):
        for _ in range(10):
            allowed, _ = claw._check_tool_arguments("s-tickets", "claw.tickets.create", {})
            assert allowed is True

    def test_blocks_eleventh_ticket_creation_in_same_session(self):
        for _ in range(10):
            claw._check_tool_arguments("s-tickets-2", "claw.tickets.create", {})
        allowed, reason = claw._check_tool_arguments("s-tickets-2", "claw.tickets.create", {})
        assert allowed is False
        assert "10 ticket" in reason

    def test_session_counters_are_isolated(self):
        for _ in range(10):
            claw._check_tool_arguments("session-a", "claw.tickets.create", {})
        # A different session still gets its own budget.
        allowed, _ = claw._check_tool_arguments("session-b", "claw.tickets.create", {})
        assert allowed is True


class TestEstimateTokens:
    def test_estimate_returns_at_least_one(self):
        assert claw._estimate_tokens("") == 1
        assert claw._estimate_tokens(None) == 1

    def test_roughly_four_chars_per_token(self):
        text = "x" * 400
        assert claw._estimate_tokens(text) == 100


class TestFormatDocsWithinBudget:
    def test_truncates_when_over_budget(self):
        docs = [SimpleNamespace(document_id=f"doc-{i}", text="word " * 20) for i in range(5)]
        # Each chunk is ~30 tokens. Budget of 60 should fit at most 2 docs.
        text = claw._format_docs_within_budget(docs, max_tokens=60)
        assert text.count("[Source: doc-") <= 2

    def test_empty_when_no_budget(self):
        docs = [SimpleNamespace(document_id="d", text="hello world")]
        assert claw._format_docs_within_budget(docs, max_tokens=0) == ""


# ---------------------------------------------------------------------------
# Agent-loop terminator: ensure a non-empty final answer even when the
# model burns through MAX_AGENT_ITERATIONS without producing one.
# ---------------------------------------------------------------------------


@dataclass
class _ChatResp:
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class _FakeSessions:
    def __init__(self):
        self.added: List[dict] = []
        self.memory: Dict[str, Dict[str, str]] = {}

    def get_or_create(self, *, user_id, session_id=None):
        return SimpleNamespace(session_id=session_id or f"s-{user_id}")

    def get_memory(self, *, user_id):
        return self.memory.get(user_id, {})

    def save_memory(self, *, user_id, key, value):
        self.memory.setdefault(user_id, {})[key] = value

    def get_messages(self, *, session_id, limit=50):
        return [], ""

    def add_messages(self, *, session_id, messages):
        self.added.extend(messages)


class _FakeData:
    def search(self, **kwargs):
        return []


class _FakeGuardrails:
    def validate_input(self, *, content, checks):
        return {"allowed": True, "denial_reason": "", "triggered_checks": []}

    def filter_output(self, *, content, filters):
        return {"content": content, "modified": False, "applied_filters": []}


class _FakeModels:
    def __init__(self, responses: List[_ChatResp]):
        self._responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def register_prompt(self, **kwargs):
        pass

    def get_prompt(self, name):
        return {"content": claw.CLAW_SYSTEM_PROMPT}

    def chat(self, *, model, messages, tools=None):
        self.calls.append({"messages": list(messages), "tools": tools})
        if not self._responses:
            return _ChatResp(content="default")
        return self._responses.pop(0)


class _FakeObservability:
    def __init__(self):
        self.scores: List[dict] = []
        self.flushed = 0

    @contextmanager
    def trace_operation(self, name, **attrs):
        yield SimpleNamespace(trace_id="trace-1", span_id="span-1")

    def record_score(self, **kwargs):
        self.scores.append(kwargs)

    def flush(self):
        self.flushed += 1


class _FakeTools:
    def build_model_tools(self, *, namespace):
        return [], {}


def _install_fakes(monkeypatch, *, model_responses: List[_ChatResp]):
    """Replace the module-level platform on claw with a fake assembly."""
    sessions = _FakeSessions()
    fake_platform = SimpleNamespace(
        models=_FakeModels(model_responses),
        sessions=sessions,
        data=_FakeData(),
        guardrails=_FakeGuardrails(),
        observability=_FakeObservability(),
        tools=_FakeTools(),
    )
    monkeypatch.setattr(claw, "platform", fake_platform)
    return fake_platform


class TestAgentLoopTerminator:
    def test_non_empty_response_when_max_iterations_exhausted(self, monkeypatch):
        """The agent loop must always produce a non-empty answer. The previous
        for...else fallback assigned ``response.content or ""`` from the last
        tool-calling response, which is almost always empty — so the user got
        a blank string when the agent ran hot."""
        # Make every iteration return tool calls so the loop never breaks
        # naturally. Then a final no-tools chat (one extra call) returns the
        # closing answer.
        tool_call_response = _ChatResp(
            content="",
            tool_calls=[
                {
                    "id": "tc-1",
                    "type": "function",
                    "function": {"name": "claw.knowledge.search", "arguments": "{}"},
                }
            ],
        )
        # Five iterations of tool calls, then one final no-tools answer.
        model_responses = [tool_call_response] * claw.MAX_AGENT_ITERATIONS + [
            _ChatResp(content="Here is the answer.", tool_calls=None)
        ]
        fake_platform = _install_fakes(monkeypatch, model_responses=model_responses)

        result = claw.claw_assistant(message="hi", user_id="u-1", session_id="")
        assert result["response"] == "Here is the answer."
        # Exactly MAX + 1 chat calls (5 in-loop + 1 closer).
        assert len(fake_platform.models.calls) == claw.MAX_AGENT_ITERATIONS + 1
        # The closer call has no tools attached so the model is forced to answer.
        assert fake_platform.models.calls[-1]["tools"] in (None, [])

    def test_normal_path_does_not_make_extra_chat_call(self, monkeypatch):
        """When the model finishes naturally (no tool calls), the agent loop
        does NOT make an extra closer call."""
        fake_platform = _install_fakes(
            monkeypatch,
            model_responses=[_ChatResp(content="Hello there.", tool_calls=None)],
        )
        result = claw.claw_assistant(message="hi", user_id="u-2", session_id="")
        assert result["response"] == "Hello there."
        assert len(fake_platform.models.calls) == 1


class TestGetPromptFallback:
    def test_falls_back_to_inline_prompt_when_lookup_fails(self, monkeypatch):
        """If the prompt isn't registered (e.g., a fresh deploy that bypassed
        run_local), assemble_context must not crash — it falls back to the
        inline CLAW_SYSTEM_PROMPT."""
        sessions = _FakeSessions()

        def boom(name):
            raise RuntimeError("prompt not registered")

        fake_platform = SimpleNamespace(
            models=SimpleNamespace(get_prompt=boom, register_prompt=lambda **_: None),
            sessions=sessions,
            data=_FakeData(),
            guardrails=_FakeGuardrails(),
            observability=_FakeObservability(),
            tools=_FakeTools(),
        )
        monkeypatch.setattr(claw, "platform", fake_platform)
        messages, _ = claw.assemble_context(user_id="u", session_id="s", message="hi")
        # The first message is the system prompt; it should be the inline default.
        assert messages[0]["role"] == "system"
        assert "Claw" in messages[0]["content"]


class TestExecuteToolDispatch:
    def test_memory_save_routes_to_sessions(self, monkeypatch):
        sessions = _FakeSessions()
        monkeypatch.setattr(
            claw,
            "platform",
            SimpleNamespace(sessions=sessions),
        )
        result = claw._execute_tool(
            "s", "u", "claw.memory.save_fact", {"key": "favorite_color", "value": "blue"}
        )
        assert result == {"status": "saved", "key": "favorite_color"}
        assert sessions.memory["u"]["favorite_color"] == "blue"

    def test_unknown_tool_returns_error(self):
        result = claw._execute_tool("s", "u", "claw.unknown.tool", {})
        assert "error" in result

    def test_calendar_event_returns_event_id(self):
        result = claw._execute_tool(
            "s",
            "u",
            "claw.calendar.create_event",
            {"title": "sync", "start_time": "2026-06-01T10:00:00+00:00"},
        )
        assert result["status"] == "scheduled"
        assert result["event_id"].startswith("evt-")

    def test_tickets_returns_ticket_id(self):
        result = claw._execute_tool("s", "u", "claw.tickets.create", {"title": "bug"})
        assert result["status"] == "open"
        assert result["ticket_id"].startswith("tkt-")


# Make pytest collect this module despite no class-based asyncio markers.
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")
