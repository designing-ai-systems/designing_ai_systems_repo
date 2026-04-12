"""
Session Service - gRPC service implementation.

Thin translation layer: receives proto requests, delegates to storage (domain types),
formats results as proto responses.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 4.15: gRPC service implementation for sessions
"""

import json

import grpc

from proto import sessions_pb2, sessions_pb2_grpc
from services.sessions.models import Function, Message, ToolCall
from services.sessions.store import SessionStorage, create_storage
from services.shared.servicer_base import BaseServicer


class SessionService(sessions_pb2_grpc.SessionServiceServicer, BaseServicer):
    def __init__(self, storage: SessionStorage = None):
        self.storage = storage or create_storage()

    def add_to_server(self, server: grpc.Server):
        sessions_pb2_grpc.add_SessionServiceServicer_to_server(self, server)

    # --- helpers: proto <-> domain conversion ---

    def _proto_msg_to_domain(self, proto_msg) -> Message:
        """Convert a proto Message to a domain Message."""
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
            tool_call_id=(proto_msg.tool_call_id if proto_msg.HasField("tool_call_id") else None),
        )

    def _domain_msg_to_proto(self, msg: Message) -> sessions_pb2.Message:
        """Convert a domain Message to a proto Message."""
        proto_msg = sessions_pb2.Message(
            role=msg.role,
            timestamp=int(msg.timestamp.timestamp() * 1000),
        )
        if msg.content is not None:
            proto_msg.content = msg.content
        if msg.tool_call_id is not None:
            proto_msg.tool_call_id = msg.tool_call_id
        if msg.tool_calls:
            for tc in msg.tool_calls:
                proto_msg.tool_calls.append(
                    sessions_pb2.ToolCall(
                        id=tc.id,
                        type=tc.type,
                        function=sessions_pb2.Function(
                            name=tc.function.name,
                            arguments=tc.function.arguments,
                        ),
                    )
                )
        return proto_msg

    # --- Session management ---

    def GetOrCreateSession(self, request, context):
        try:
            session_id = request.session_id if request.HasField("session_id") else None
            session = self.storage.get_or_create_session(
                user_id=request.user_id, session_id=session_id
            )
            return sessions_pb2.GetOrCreateSessionResponse(
                session=sessions_pb2.Session(
                    session_id=session.session_id,
                    user_id=session.user_id,
                    created_at=int(session.created_at.timestamp() * 1000),
                    updated_at=int(session.updated_at.timestamp() * 1000),
                )
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to get or create session: {e}")
            return sessions_pb2.GetOrCreateSessionResponse()

    def ListSessions(self, request, context):
        try:
            sessions = self.storage.list_sessions(user_id=request.user_id)
            proto_sessions = [
                sessions_pb2.Session(
                    session_id=s.session_id,
                    user_id=s.user_id,
                    created_at=int(s.created_at.timestamp() * 1000),
                    updated_at=int(s.updated_at.timestamp() * 1000),
                )
                for s in sessions
            ]
            return sessions_pb2.ListSessionsResponse(sessions=proto_sessions)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to list sessions: {e}")
            return sessions_pb2.ListSessionsResponse(sessions=[])

    # --- Message operations ---

    def AddMessages(self, request, context):
        try:
            domain_msgs = [self._proto_msg_to_domain(m) for m in request.messages]
            count = self.storage.add_messages(request.session_id, domain_msgs)
            return sessions_pb2.AddMessagesResponse(success=True, message_count=count)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to add messages: {e}")
            return sessions_pb2.AddMessagesResponse(success=False, message_count=0)

    def GetMessages(self, request, context):
        try:
            limit = request.limit if request.HasField("limit") else None
            offset = request.offset if request.HasField("offset") else None
            messages, total_count = self.storage.get_messages(
                session_id=request.session_id, limit=limit, offset=offset
            )
            proto_msgs = [self._domain_msg_to_proto(m) for m in messages]
            return sessions_pb2.GetMessagesResponse(messages=proto_msgs, total_count=total_count)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to get messages: {e}")
            return sessions_pb2.GetMessagesResponse(messages=[], total_count=0)

    def DeleteSession(self, request, context):
        try:
            success = self.storage.delete_session(request.session_id)
            return sessions_pb2.DeleteSessionResponse(success=success)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to delete session: {e}")
            return sessions_pb2.DeleteSessionResponse(success=False)

    # --- Memory operations ---

    def SaveMemory(self, request, context):
        try:
            session_id = request.session_id if request.HasField("session_id") else None
            success = self.storage.save_memory(
                user_id=request.user_id,
                key=request.key,
                value=json.loads(request.value),
                session_id=session_id,
            )
            return sessions_pb2.SaveMemoryResponse(success=success)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to save memory: {e}")
            return sessions_pb2.SaveMemoryResponse(success=False)

    def GetMemory(self, request, context):
        try:
            key = request.key if request.HasField("key") else None
            session_id = request.session_id if request.HasField("session_id") else None
            memories = self.storage.get_memory(
                user_id=request.user_id, key=key, session_id=session_id
            )
            memories_json = {k: json.dumps(v) for k, v in memories.items()}
            return sessions_pb2.GetMemoryResponse(memories=memories_json)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to get memory: {e}")
            return sessions_pb2.GetMemoryResponse(memories={})

    def DeleteMemory(self, request, context):
        try:
            session_id = request.session_id if request.HasField("session_id") else None
            success = self.storage.delete_memory(
                user_id=request.user_id, key=request.key, session_id=session_id
            )
            return sessions_pb2.DeleteMemoryResponse(success=success)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to delete memory: {e}")
            return sessions_pb2.DeleteMemoryResponse(success=False)

    def ClearUserMemory(self, request, context):
        try:
            count = self.storage.clear_user_memory(request.user_id)
            return sessions_pb2.ClearUserMemoryResponse(count=count)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to clear user memory: {e}")
            return sessions_pb2.ClearUserMemoryResponse(count=0)
