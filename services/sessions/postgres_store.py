"""
PostgreSQL implementation of SessionStorage.

Uses psycopg2 and domain types exclusively.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 4.9:  PostgreSQL schema — sessions table
  - Listing 4.10: PostgreSQL schema — messages table
  - Listing 4.11: PostgresSessionStorage class + get_or_create_session
  - Listing 4.12: add_messages (batch insert in single transaction)
  - Listing 4.13: get_messages (paginated retrieval with total_count)
  - Listing 4.14: _row_to_session / _row_to_message helpers
"""

import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
from psycopg2.extras import RealDictCursor

from services.sessions.models import Function, Message, Session, ToolCall
from services.sessions.store import SessionStorage


class PostgresSessionStorage(SessionStorage):
    """PostgreSQL implementation of SessionStorage."""

    def __init__(self, connection_string: Optional[str] = None):
        if not connection_string:
            connection_string = os.getenv(
                "DB_CONNECTION_STRING",
                "postgresql://localhost/genai_platform",
            )
        self.conn = psycopg2.connect(connection_string, cursor_factory=RealDictCursor)
        self._create_tables()

    def _create_tables(self):
        """Create tables if they don't exist."""
        schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
        with open(schema_path) as f:
            sql = f.read()
        with self.conn.cursor() as cur:
            cur.execute(sql)
        self.conn.commit()

    # --- Session management ---

    def get_or_create_session(
        self, user_id: str, session_id: Optional[str] = None
    ) -> Session:
        with self.conn.cursor() as cur:
            if session_id:
                cur.execute(
                    "SELECT * FROM sessions WHERE session_id = %s", (session_id,)
                )
                row = cur.fetchone()
                if row:
                    return self._row_to_session(row)

            new_id = session_id or str(uuid.uuid4())
            now = datetime.utcnow()
            cur.execute(
                """INSERT INTO sessions (session_id, user_id, created_at, updated_at)
                   VALUES (%s, %s, %s, %s)
                   ON CONFLICT (session_id) DO UPDATE SET updated_at = EXCLUDED.updated_at
                   RETURNING *""",
                (new_id, user_id, now, now),
            )
            self.conn.commit()
            return self._row_to_session(cur.fetchone())

    def list_sessions(self, user_id: str) -> List[Session]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM sessions WHERE user_id = %s ORDER BY updated_at DESC",
                (user_id,),
            )
            return [self._row_to_session(row) for row in cur.fetchall()]

    def delete_session(self, session_id: str) -> bool:
        with self.conn.cursor() as cur:
            cur.execute(
                "DELETE FROM sessions WHERE session_id = %s", (session_id,)
            )
            self.conn.commit()
            return cur.rowcount > 0

    # --- Messages ---

    def add_messages(self, session_id: str, messages: List[Message]) -> int:
        with self.conn.cursor() as cur:
            for msg in messages:
                tool_calls_json = None
                if msg.tool_calls:
                    tool_calls_json = json.dumps([
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ])
                cur.execute(
                    """INSERT INTO messages
                       (session_id, role, content, tool_calls, tool_call_id, timestamp)
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    (
                        session_id,
                        msg.role,
                        msg.content,
                        tool_calls_json,
                        msg.tool_call_id,
                        msg.timestamp,
                    ),
                )
            cur.execute(
                "UPDATE sessions SET updated_at = %s WHERE session_id = %s",
                (datetime.utcnow(), session_id),
            )
            self.conn.commit()
        return len(messages)

    def get_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> Tuple[List[Message], int]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) as count FROM messages WHERE session_id = %s",
                (session_id,),
            )
            total = cur.fetchone()["count"]

            query = "SELECT * FROM messages WHERE session_id = %s ORDER BY id"
            params: list = [session_id]
            if limit:
                query += " LIMIT %s"
                params.append(limit)
            if offset:
                query += " OFFSET %s"
                params.append(offset)
            cur.execute(query, params)
            rows = cur.fetchall()
        return [self._row_to_message(r) for r in rows], total

    # --- Memory ---

    def save_memory(
        self,
        user_id: str,
        key: str,
        value: Any,
        session_id: Optional[str] = None,
    ) -> bool:
        now = datetime.utcnow()
        with self.conn.cursor() as cur:
            cur.execute(
                """INSERT INTO memories (user_id, key, value, session_id, created_at, updated_at)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   ON CONFLICT (user_id, key, session_id)
                   DO UPDATE SET value = EXCLUDED.value, updated_at = EXCLUDED.updated_at""",
                (user_id, key, json.dumps(value), session_id or "", now, now),
            )
            self.conn.commit()
        return True

    def get_memory(
        self,
        user_id: str,
        key: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        query = "SELECT key, value FROM memories WHERE user_id = %s"
        params: list = [user_id]
        if key:
            query += " AND key = %s"
            params.append(key)
        if session_id:
            query += " AND session_id = %s"
            params.append(session_id)
        else:
            query += " AND session_id = ''"
        with self.conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        return {r["key"]: json.loads(r["value"]) for r in rows}

    def delete_memory(
        self, user_id: str, key: str, session_id: Optional[str] = None
    ) -> bool:
        query = "DELETE FROM memories WHERE user_id = %s AND key = %s AND session_id = %s"
        params: list = [user_id, key, session_id or ""]
        with self.conn.cursor() as cur:
            cur.execute(query, params)
            self.conn.commit()
        return cur.rowcount > 0

    def clear_user_memory(self, user_id: str) -> int:
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM memories WHERE user_id = %s", (user_id,))
            self.conn.commit()
        return cur.rowcount

    # --- Row conversions ---

    def _row_to_session(self, row) -> Session:
        return Session(
            session_id=row["session_id"],
            user_id=row["user_id"],
            messages=[],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _row_to_message(self, row) -> Message:
        tool_calls = None
        tc_data = row.get("tool_calls")
        if tc_data:
            if isinstance(tc_data, str):
                tc_data = json.loads(tc_data)
            tool_calls = [
                ToolCall(
                    id=tc["id"],
                    type=tc["type"],
                    function=Function(
                        name=tc["function"]["name"],
                        arguments=tc["function"]["arguments"],
                    ),
                )
                for tc in tc_data
            ]
        return Message(
            role=row["role"],
            content=row.get("content"),
            tool_calls=tool_calls,
            tool_call_id=row.get("tool_call_id"),
            timestamp=row["timestamp"],
        )
