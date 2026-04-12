"""
Session Store - Data layer for session management.

Provides storage abstraction with domain types.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 4.8: SessionStorage ABC (session/message CRUD)
  - Listing 4.20: Memory methods added to SessionStorage ABC
"""

import os
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from services.sessions.models import Message, Session


class SessionStorage(ABC):
    """Abstract storage interface for session persistence.

    Listing 4.8: session/message CRUD methods.
    Listing 4.20: save_memory, get_memory, delete_memory, clear_user_memory.
    """

    @abstractmethod
    def get_or_create_session(self, user_id: str, session_id: Optional[str] = None) -> Session:
        """Retrieve existing session or create new one."""
        pass

    @abstractmethod
    def list_sessions(self, user_id: str) -> List[Session]:
        """Retrieve all sessions for a given user."""
        pass

    @abstractmethod
    def add_messages(self, session_id: str, messages: List[Message]) -> int:
        """Append messages to session. Returns count added."""
        pass

    @abstractmethod
    def get_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> Tuple[List[Message], int]:
        """Retrieve messages with pagination. Returns messages and total count."""
        pass

    @abstractmethod
    def delete_session(self, session_id: str) -> bool:
        """Remove session and all its messages. Returns success."""
        pass

    @abstractmethod
    def save_memory(
        self,
        user_id: str,
        key: str,
        value: Any,
        session_id: Optional[str] = None,
    ) -> bool:
        """Save a fact to memory. Updates if key exists."""
        pass

    @abstractmethod
    def get_memory(
        self,
        user_id: str,
        key: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retrieve memories for a user."""
        pass

    @abstractmethod
    def delete_memory(self, user_id: str, key: str, session_id: Optional[str] = None) -> bool:
        """Remove a fact from memory. Returns True if key existed."""
        pass

    @abstractmethod
    def clear_user_memory(self, user_id: str) -> int:
        """Remove all memories for a user. Returns count deleted."""
        pass


class InMemorySessionStorage(SessionStorage):
    """In-memory implementation of SessionStorage for development and testing."""

    def __init__(self):
        self._sessions: Dict[str, Session] = {}
        self._messages: Dict[str, List[Message]] = {}
        self._memories: Dict[tuple, Any] = {}

    def get_or_create_session(self, user_id: str, session_id: Optional[str] = None) -> Session:
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]

        new_id = session_id or str(uuid.uuid4())
        now = datetime.utcnow()
        session = Session(
            session_id=new_id,
            user_id=user_id,
            messages=[],
            created_at=now,
            updated_at=now,
        )
        self._sessions[new_id] = session
        self._messages[new_id] = []
        return session

    def list_sessions(self, user_id: str) -> List[Session]:
        return [s for s in self._sessions.values() if s.user_id == user_id]

    def add_messages(self, session_id: str, messages: List[Message]) -> int:
        if session_id not in self._messages:
            self._messages[session_id] = []
        self._messages[session_id].extend(messages)
        if session_id in self._sessions:
            self._sessions[session_id].updated_at = datetime.utcnow()
        return len(messages)

    def get_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> Tuple[List[Message], int]:
        if session_id not in self._messages:
            return [], 0
        all_msgs = self._messages[session_id]
        total = len(all_msgs)
        start = offset or 0
        end = start + limit if limit else None
        return all_msgs[start:end], total

    def delete_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            self._messages.pop(session_id, None)
            return True
        return False

    def save_memory(
        self,
        user_id: str,
        key: str,
        value: Any,
        session_id: Optional[str] = None,
    ) -> bool:
        mem_key = (user_id, key, session_id or "")
        self._memories[mem_key] = value
        return True

    def get_memory(
        self,
        user_id: str,
        key: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        results = {}
        for (mu, mk, ms), value in self._memories.items():
            if mu != user_id:
                continue
            if key and mk != key:
                continue
            if session_id and ms != session_id:
                continue
            results[mk] = value
        return results

    def delete_memory(self, user_id: str, key: str, session_id: Optional[str] = None) -> bool:
        mem_key = (user_id, key, session_id or "")
        if mem_key in self._memories:
            del self._memories[mem_key]
            return True
        return False

    def clear_user_memory(self, user_id: str) -> int:
        to_delete = [k for k in self._memories if k[0] == user_id]
        for k in to_delete:
            del self._memories[k]
        return len(to_delete)


def create_storage() -> SessionStorage:
    """Create storage instance based on environment configuration.

    Environment variables:
        SESSION_STORAGE: "memory" (default) or "postgres"
        DB_CONNECTION_STRING: PostgreSQL connection string (if using postgres)
    """
    storage_type = os.getenv("SESSION_STORAGE", "memory")
    if storage_type == "postgres":
        from services.sessions.postgres_store import PostgresSessionStorage

        return PostgresSessionStorage()
    return InMemorySessionStorage()
