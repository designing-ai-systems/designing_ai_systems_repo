"""
Session Service client.

Translates simple Python method calls into gRPC requests and returns
domain dataclasses, never exposing Protocol Buffers to the caller.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 4.16: SessionClient setup (class + __init__)
  - Listing 4.17: get_or_create SDK method
  - Listing 4.21: save_memory / get_memory SDK methods
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from proto import sessions_pb2, sessions_pb2_grpc
from services.sessions.models import Function, Message, Session, ToolCall

from .base import BaseClient


class SessionClient(BaseClient):
    """Client for Session Service."""

    def __init__(self, platform):
        super().__init__(platform, service_name="sessions")
        self._stub = sessions_pb2_grpc.SessionServiceStub(self._channel)

    # --- Session management ---

    def get_or_create(
        self, user_id: str, session_id: Optional[str] = None
    ) -> Session:
        """Get existing session or create a new one (Listing 4.17)."""
        request = sessions_pb2.GetOrCreateSessionRequest(
            user_id=user_id,
            session_id=session_id if session_id else "",
        )
        response = self._stub.GetOrCreateSession(
            request, metadata=self._metadata
        )
        s = response.session
        return Session(
            session_id=s.session_id,
            user_id=s.user_id,
            messages=[],
            created_at=datetime.fromtimestamp(s.created_at / 1000),
            updated_at=datetime.fromtimestamp(s.updated_at / 1000),
        )

    def list_sessions(self, user_id: str) -> List[Session]:
        """List all sessions for a user."""
        request = sessions_pb2.ListSessionsRequest(user_id=user_id)
        response = self._stub.ListSessions(request, metadata=self._metadata)
        return [
            Session(
                session_id=s.session_id,
                user_id=s.user_id,
                messages=[],
                created_at=datetime.fromtimestamp(s.created_at / 1000),
                updated_at=datetime.fromtimestamp(s.updated_at / 1000),
            )
            for s in response.sessions
        ]

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages."""
        request = sessions_pb2.DeleteSessionRequest(session_id=session_id)
        response = self._stub.DeleteSession(request, metadata=self._metadata)
        return response.success

    # --- Message operations ---

    def add_messages(
        self, session_id: str, messages: List[Dict[str, Any]]
    ) -> int:
        """Add messages to a session. Accepts list of dicts (Listing 4.18)."""
        proto_messages = [self._dict_to_proto(m) for m in messages]
        request = sessions_pb2.AddMessagesRequest(
            session_id=session_id, messages=proto_messages
        )
        response = self._stub.AddMessages(request, metadata=self._metadata)
        return response.message_count

    def get_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        strategy: Optional[str] = None,
    ) -> Tuple[List[Message], int]:
        """Get messages from a session. Returns (messages, total_count)."""
        request = sessions_pb2.GetMessagesRequest(session_id=session_id)
        if limit is not None:
            request.limit = limit
        if offset is not None:
            request.offset = offset
        if strategy:
            request.strategy = strategy
        response = self._stub.GetMessages(request, metadata=self._metadata)
        messages = [self._proto_to_domain(m) for m in response.messages]
        return messages, response.total_count

    # --- Memory operations (Listing 4.21) ---

    def save_memory(
        self,
        user_id: str,
        key: str,
        value: Any,
        session_id: Optional[str] = None,
    ) -> bool:
        """Save a fact to memory."""
        request = sessions_pb2.SaveMemoryRequest(
            user_id=user_id,
            key=key,
            value=json.dumps(value),
            session_id=session_id or "",
        )
        response = self._stub.SaveMemory(request, metadata=self._metadata)
        return response.success

    def get_memory(
        self,
        user_id: str,
        key: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retrieve memories. Optionally filter by key or session."""
        request = sessions_pb2.GetMemoryRequest(user_id=user_id)
        if key:
            request.key = key
        if session_id:
            request.session_id = session_id
        response = self._stub.GetMemory(request, metadata=self._metadata)
        return {k: json.loads(v) for k, v in response.memories.items()}

    def delete_memory(
        self,
        user_id: str,
        key: str,
        session_id: Optional[str] = None,
    ) -> bool:
        """Delete a memory entry."""
        request = sessions_pb2.DeleteMemoryRequest(
            user_id=user_id,
            key=key,
            session_id=session_id or "",
        )
        response = self._stub.DeleteMemory(request, metadata=self._metadata)
        return response.success

    def clear_user_memory(self, user_id: str) -> int:
        """Clear all memories for a user."""
        request = sessions_pb2.ClearUserMemoryRequest(user_id=user_id)
        response = self._stub.ClearUserMemory(
            request, metadata=self._metadata
        )
        return response.count

    # --- Conversion helpers ---

    def _dict_to_proto(self, msg_dict: Dict[str, Any]) -> sessions_pb2.Message:
        message = sessions_pb2.Message(
            role=msg_dict.get("role", "user"),
            timestamp=msg_dict.get(
                "timestamp", int(datetime.utcnow().timestamp() * 1000)
            ),
        )
        if "content" in msg_dict and msg_dict["content"] is not None:
            message.content = msg_dict["content"]
        if "tool_call_id" in msg_dict and msg_dict["tool_call_id"] is not None:
            message.tool_call_id = msg_dict["tool_call_id"]
        if "name" in msg_dict and msg_dict["name"] is not None:
            message.name = msg_dict["name"]
        if "tool_calls" in msg_dict and msg_dict["tool_calls"]:
            for tc in msg_dict["tool_calls"]:
                tool_call = sessions_pb2.ToolCall(
                    id=tc["id"],
                    type=tc.get("type", "function"),
                    function=sessions_pb2.Function(
                        name=tc["function"]["name"],
                        arguments=tc["function"]["arguments"],
                    ),
                )
                message.tool_calls.append(tool_call)
        return message

    def _proto_to_domain(self, proto_msg) -> Message:
        tool_calls = None
        if proto_msg.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    type=tc.type,
                    function=Function(
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    ),
                )
                for tc in proto_msg.tool_calls
            ]
        return Message(
            role=proto_msg.role,
            content=proto_msg.content if proto_msg.HasField("content") else None,
            tool_calls=tool_calls,
            tool_call_id=(
                proto_msg.tool_call_id
                if proto_msg.HasField("tool_call_id")
                else None
            ),
            timestamp=datetime.fromtimestamp(proto_msg.timestamp / 1000)
            if proto_msg.timestamp
            else datetime.utcnow(),
        )
