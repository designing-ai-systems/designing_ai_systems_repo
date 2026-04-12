"""Tests for Session Service domain models (book Listings 4.1, 4.2, 4.19)."""

from datetime import datetime

from services.sessions.models import Function, MemoryEntry, Message, Session, ToolCall


class TestMessage:
    def test_simple_user_message(self):
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.tool_calls is None
        assert msg.tool_call_id is None
        assert isinstance(msg.timestamp, datetime)

    def test_assistant_message_with_tool_calls(self):
        func = Function(name="check_schedule", arguments='{"date": "Tuesday"}')
        tc = ToolCall(id="call_123", type="function", function=func)
        msg = Message(role="assistant", content=None, tool_calls=[tc])
        assert msg.content is None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].function.name == "check_schedule"

    def test_tool_result_message(self):
        msg = Message(role="tool", content='{"slots": ["9am"]}', tool_call_id="call_123")
        assert msg.role == "tool"
        assert msg.tool_call_id == "call_123"

    def test_timestamp_auto_set(self):
        before = datetime.utcnow()
        msg = Message(role="user", content="test")
        after = datetime.utcnow()
        assert before <= msg.timestamp <= after


class TestSession:
    def test_session_creation(self):
        now = datetime.utcnow()
        session = Session(
            session_id="sess-1",
            user_id="user-1",
            messages=[],
            created_at=now,
            updated_at=now,
        )
        assert session.session_id == "sess-1"
        assert session.user_id == "user-1"
        assert session.messages == []


class TestMemoryEntry:
    def test_memory_entry(self):
        entry = MemoryEntry(key="allergies", value=["penicillin"], user_id="user-1")
        assert entry.key == "allergies"
        assert entry.value == ["penicillin"]
        assert entry.session_id is None
