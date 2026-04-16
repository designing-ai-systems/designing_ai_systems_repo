"""Tests for PostgresSessionStorage against a real PostgreSQL database.

Auto-creates the 'genai_platform_test' database if PostgreSQL is running
but the database doesn't exist yet. Schema is applied automatically by
PostgresSessionStorage._create_tables().

Skips all tests when PostgreSQL is unavailable (not installed, not running,
or psycopg2 not installed) so CI / in-memory-only environments still pass.
"""

import pytest

_pg_available = False

try:
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

    try:
        _conn = psycopg2.connect("postgresql://localhost/genai_platform_test")
        _conn.close()
        _pg_available = True
    except psycopg2.OperationalError:
        # Database doesn't exist yet — try to create it
        try:
            _conn = psycopg2.connect("postgresql://localhost/postgres")
            _conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            with _conn.cursor() as cur:
                cur.execute("CREATE DATABASE genai_platform_test")
            _conn.close()
            _pg_available = True
        except Exception:
            pass
except ImportError:
    pass

pytestmark = pytest.mark.skipif(not _pg_available, reason="PostgreSQL test database unavailable")

if _pg_available:
    from services.sessions.models import Function, Message, Session, ToolCall  # noqa: E402
    from services.sessions.postgres_store import PostgresSessionStorage  # noqa: E402

TEST_DB = "postgresql://localhost/genai_platform_test"


@pytest.fixture
def storage():
    store = PostgresSessionStorage(connection_string=TEST_DB)
    yield store
    with store.conn.cursor() as cur:
        cur.execute("TRUNCATE messages, memories, sessions CASCADE")
    store.conn.commit()
    store.conn.close()


class TestGetOrCreateSession:
    def test_create_new_session(self, storage):
        session = storage.get_or_create_session(user_id="user-1")
        assert isinstance(session, Session)
        assert session.user_id == "user-1"
        assert session.session_id

    def test_create_with_explicit_id(self, storage):
        session = storage.get_or_create_session(user_id="user-1", session_id="my-sess")
        assert session.session_id == "my-sess"

    def test_retrieve_existing_session(self, storage):
        s1 = storage.get_or_create_session(user_id="user-1", session_id="sess-1")
        s2 = storage.get_or_create_session(user_id="user-1", session_id="sess-1")
        assert s1.session_id == s2.session_id

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

    def test_tool_call_messages(self, storage):
        storage.get_or_create_session(user_id="user-1", session_id="sess-1")
        tc = ToolCall(
            id="call_abc",
            type="function",
            function=Function(name="lookup", arguments='{"id": "1"}'),
        )
        msgs = [
            Message(role="assistant", content=None, tool_calls=[tc]),
            Message(role="tool", content='{"result": "ok"}', tool_call_id="call_abc"),
        ]
        storage.add_messages("sess-1", msgs)

        retrieved, total = storage.get_messages("sess-1")
        assert total == 2
        assert retrieved[0].tool_calls[0].id == "call_abc"
        assert retrieved[0].tool_calls[0].function.name == "lookup"
        assert retrieved[1].tool_call_id == "call_abc"


class TestDeleteSession:
    def test_delete_existing(self, storage):
        storage.get_or_create_session(user_id="user-1", session_id="sess-1")
        assert storage.delete_session("sess-1") is True
        assert storage.list_sessions("user-1") == []

    def test_delete_nonexistent(self, storage):
        assert storage.delete_session("nope") is False

    def test_delete_cascades_messages(self, storage):
        storage.get_or_create_session(user_id="user-1", session_id="sess-1")
        storage.add_messages("sess-1", [Message(role="user", content="hi")])
        storage.delete_session("sess-1")
        msgs, total = storage.get_messages("sess-1")
        assert total == 0


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
