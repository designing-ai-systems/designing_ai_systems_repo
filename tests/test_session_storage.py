"""Tests for InMemorySessionStorage (book Listing 4.8 + 4.20 interface)."""

import pytest

from services.sessions.models import Message, Session
from services.sessions.store import InMemorySessionStorage


@pytest.fixture
def storage():
    return InMemorySessionStorage()


class TestGetOrCreateSession:
    def test_create_new_session(self, storage):
        session = storage.get_or_create_session(user_id="user-1")
        assert isinstance(session, Session)
        assert session.user_id == "user-1"
        assert session.session_id  # non-empty
        assert session.messages == []

    def test_create_with_explicit_id(self, storage):
        session = storage.get_or_create_session(user_id="user-1", session_id="my-sess")
        assert session.session_id == "my-sess"

    def test_retrieve_existing_session(self, storage):
        s1 = storage.get_or_create_session(user_id="user-1", session_id="sess-1")
        s2 = storage.get_or_create_session(user_id="user-1", session_id="sess-1")
        assert s1.session_id == s2.session_id
        assert s1.created_at == s2.created_at

    def test_new_session_when_id_not_found(self, storage):
        session = storage.get_or_create_session(user_id="user-1", session_id="nonexistent")
        assert session.session_id == "nonexistent"


class TestListSessions:
    def test_list_empty(self, storage):
        result = storage.list_sessions(user_id="user-1")
        assert result == []

    def test_list_returns_user_sessions(self, storage):
        storage.get_or_create_session(user_id="user-1", session_id="s1")
        storage.get_or_create_session(user_id="user-1", session_id="s2")
        storage.get_or_create_session(user_id="user-2", session_id="s3")
        result = storage.list_sessions(user_id="user-1")
        assert len(result) == 2
        ids = {s.session_id for s in result}
        assert ids == {"s1", "s2"}


class TestMessages:
    def test_add_and_get_messages(self, storage):
        storage.get_or_create_session(user_id="user-1", session_id="sess-1")
        msgs = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there"),
        ]
        count = storage.add_messages("sess-1", msgs)
        assert count == 2

        retrieved, total = storage.get_messages("sess-1")
        assert total == 2
        assert len(retrieved) == 2
        assert retrieved[0].role == "user"
        assert retrieved[1].content == "Hi there"

    def test_get_messages_with_limit(self, storage):
        storage.get_or_create_session(user_id="user-1", session_id="sess-1")
        msgs = [Message(role="user", content=f"msg-{i}") for i in range(5)]
        storage.add_messages("sess-1", msgs)

        retrieved, total = storage.get_messages("sess-1", limit=2)
        assert len(retrieved) == 2
        assert total == 5

    def test_get_messages_with_offset(self, storage):
        storage.get_or_create_session(user_id="user-1", session_id="sess-1")
        msgs = [Message(role="user", content=f"msg-{i}") for i in range(5)]
        storage.add_messages("sess-1", msgs)

        retrieved, total = storage.get_messages("sess-1", limit=2, offset=3)
        assert len(retrieved) == 2
        assert retrieved[0].content == "msg-3"
        assert total == 5


class TestDeleteSession:
    def test_delete_existing(self, storage):
        storage.get_or_create_session(user_id="user-1", session_id="sess-1")
        assert storage.delete_session("sess-1") is True
        assert storage.list_sessions("user-1") == []

    def test_delete_nonexistent(self, storage):
        assert storage.delete_session("nope") is False


class TestMemory:
    def test_save_and_get_memory(self, storage):
        assert storage.save_memory("user-1", "allergies", ["penicillin"])
        memories = storage.get_memory("user-1")
        assert memories == {"allergies": ["penicillin"]}

    def test_get_specific_key(self, storage):
        storage.save_memory("user-1", "allergies", ["penicillin"])
        storage.save_memory("user-1", "preferred_time", "morning")
        memories = storage.get_memory("user-1", key="allergies")
        assert memories == {"allergies": ["penicillin"]}

    def test_update_existing_key(self, storage):
        storage.save_memory("user-1", "allergies", ["penicillin"])
        storage.save_memory("user-1", "allergies", ["penicillin", "latex"])
        memories = storage.get_memory("user-1")
        assert memories["allergies"] == ["penicillin", "latex"]

    def test_delete_memory(self, storage):
        storage.save_memory("user-1", "allergies", ["penicillin"])
        assert storage.delete_memory("user-1", "allergies") is True
        assert storage.get_memory("user-1") == {}

    def test_delete_nonexistent_memory(self, storage):
        assert storage.delete_memory("user-1", "nope") is False

    def test_clear_user_memory(self, storage):
        storage.save_memory("user-1", "a", 1)
        storage.save_memory("user-1", "b", 2)
        count = storage.clear_user_memory("user-1")
        assert count == 2
        assert storage.get_memory("user-1") == {}

    def test_session_scoped_memory(self, storage):
        storage.save_memory("user-1", "note", "sess-specific", session_id="sess-1")
        storage.save_memory("user-1", "global_note", "all-sessions")
        scoped = storage.get_memory("user-1", session_id="sess-1")
        assert "note" in scoped
