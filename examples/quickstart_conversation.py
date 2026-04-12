"""
Session + Model Service integration example.

Demonstrates end-to-end conversation with memory.
Requires OPENAI_API_KEY or ANTHROPIC_API_KEY.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 4.18: Workflow using the Session Service
  - Listing 4.22: Healthcare workflow with model-managed memory
"""

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from genai_platform import GenAIPlatform
from services.gateway.main import main as start_gateway
from services.models.main import main as start_model_service
from services.sessions.main import main as start_session_service


def start_service_in_thread(service_func, service_name):
    def run_service():
        try:
            service_func()
        except KeyboardInterrupt:
            pass

    thread = threading.Thread(target=run_service, daemon=True, name=service_name)
    thread.start()
    return thread


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

    start_service_in_thread(start_session_service, "SessionService")
    time.sleep(1)
    start_service_in_thread(start_model_service, "ModelService")
    time.sleep(1)
    start_service_in_thread(start_gateway, "Gateway")
    time.sleep(1)
    print("Services ready!\n")

    platform = GenAIPlatform()

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

    print("\nDone! Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        sys.exit(0)
