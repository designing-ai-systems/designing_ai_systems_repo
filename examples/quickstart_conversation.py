"""
Session + Model Service integration example.

Demonstrates end-to-end conversation with memory.
Requires OPENAI_API_KEY or ANTHROPIC_API_KEY.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 4.18: Workflow using the Session Service
  - Listing 4.22: Healthcare workflow with model-managed memory
"""

import os
import sys
import threading
import time
from http.server import HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from genai_platform import GenAIPlatform
from services.gateway.registry import ServiceRegistry
from services.gateway.servers import create_grpc_server as create_gateway_grpc
from services.gateway.servers import create_http_server
from services.models.main import load_env_file
from services.models.service import ModelService
from services.sessions.service import SessionService
from services.shared.server import create_grpc_server, get_service_port


def _start_servers():
    """Start session service, model service, and gateway."""
    HTTPServer.allow_reuse_address = True

    project_root = Path(__file__).resolve().parents[1]
    load_env_file(project_root / ".env")

    # Session service
    session_port = get_service_port("sessions")
    session_servicer = SessionService()
    session_server = create_grpc_server(
        servicer=session_servicer, port=session_port, service_name="sessions"
    )
    session_server.start()

    # Model service
    model_port = get_service_port("models")
    model_servicer = ModelService()
    model_server = create_grpc_server(
        servicer=model_servicer, port=model_port, service_name="models"
    )
    model_server.start()

    # Gateway
    registry = ServiceRegistry()
    registry.register_platform_service(
        "sessions", os.getenv("SESSIONS_SERVICE_ADDR", f"localhost:{session_port}")
    )
    registry.register_platform_service(
        "models", os.getenv("MODELS_SERVICE_ADDR", f"localhost:{model_port}")
    )

    grpc_port = int(os.getenv("GATEWAY_PORT", "50051"))
    gateway_server = create_gateway_grpc(registry, grpc_port)
    gateway_server.start()

    http_port = int(os.getenv("GATEWAY_HTTP_PORT", "8080"))
    http_server = create_http_server(registry, http_port)
    http_thread = threading.Thread(target=http_server.serve_forever, daemon=True)
    http_thread.start()

    return session_server, model_server, gateway_server, http_server


def _stop_servers(session_server, model_server, gateway_server, http_server):
    """Gracefully stop all servers."""
    http_server.shutdown()
    gateway_server.stop(grace=0)
    model_server.stop(grace=0)
    session_server.stop(grace=0)


def chat_with_history(platform, session_id: str, question: str, model: str = "gpt-4o-mini"):
    """Send a message with full conversation history."""
    messages, _ = platform.sessions.get_messages(session_id, limit=20)

    conversation = [
        {"role": "system", "content": "You are a helpful patient intake assistant."},
    ]
    for msg in messages:
        conversation.append({"role": msg.role, "content": msg.content})
    conversation.append({"role": "user", "content": question})

    answer_parts = []
    for chunk in platform.models.chat_stream(model=model, messages=conversation):
        print(chunk.token, end="", flush=True)
        answer_parts.append(chunk.token)
    answer = "".join(answer_parts)
    print()

    platform.sessions.add_messages(
        session_id,
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
    )
    return answer


if __name__ == "__main__":
    print("=" * 60)
    print("  Session + Model Service Integration Demo")
    print("  Chapters 3-4: Conversation with Memory")
    print("=" * 60)
    print("\nStarting services...")

    session_server, model_server, gateway_server, http_server = _start_servers()
    time.sleep(1)
    print("Services ready!\n")

    platform = GenAIPlatform()

    try:
        # Create session (returns Session dataclass)
        session = platform.sessions.get_or_create(user_id="patient-123")
        print(f"Session: {session.session_id}\n")

        # First question with context
        print("Q: I'm traveling from Canada for my appointment.")
        print("A: ", end="", flush=True)
        chat_with_history(
            platform,
            session.session_id,
            "I'm traveling from Canada for my appointment. What documents do I need?",
        )
        print()

        # Follow-up -- tests context retention
        print("Q: Where am I coming from?")
        print("A: ", end="", flush=True)
        answer = chat_with_history(platform, session.session_id, "Where am I coming from?")
        print()

        if "canada" in answer.lower():
            print("Context awareness verified!")
        else:
            print("Context not used -- model should have answered 'Canada'")

        # Show stored messages
        print("\n" + "-" * 60)
        stored, total = platform.sessions.get_messages(session.session_id)
        for i, msg in enumerate(stored, 1):
            preview = (msg.content or "")[:80]
            if len(msg.content or "") > 80:
                preview += "..."
            print(f"  {i}. [{msg.role.upper()}] {preview}")
        print(f"\nTotal: {total} messages stored")

        # List sessions for the user
        sessions = platform.sessions.list_sessions("patient-123")
        print(f"\nUser has {len(sessions)} session(s)")

        print("\n✓ Done!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        _stop_servers(session_server, model_server, gateway_server, http_server)
