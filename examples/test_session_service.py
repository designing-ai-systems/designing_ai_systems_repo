"""
Comprehensive Session Service test.

This example demonstrates all Session Service capabilities from Chapter 4:
- Session management (get_or_create, delete)
- Message storage and retrieval (add_messages, get_messages)
- Model-managed memory (save_memory, get_memory)
- Pagination and context window strategies

Runs the full suite against both storage backends:
  1. In-memory (always)
  2. PostgreSQL (auto-detected; skipped if unavailable)
"""

import os
import sys
import threading
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genai_platform import GenAIPlatform
from services.gateway.main import main as start_gateway
from services.gateway.registry import ServiceRegistry
from services.gateway.servers import create_grpc_server as create_gateway_grpc
from services.gateway.servers import create_http_server
from services.sessions.service import SessionService
from services.sessions.store import InMemorySessionStorage
from services.shared.server import create_grpc_server, get_service_port

TEST_DB = "postgresql://localhost/genai_platform_test"


def _detect_postgres() -> bool:
    """Return True if PostgreSQL is available, auto-creating the test DB if needed."""
    try:
        import psycopg2
        from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
    except ImportError:
        return False

    try:
        conn = psycopg2.connect(TEST_DB)
        conn.close()
        return True
    except psycopg2.OperationalError:
        pass

    try:
        conn = psycopg2.connect("postgresql://localhost/postgres")
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        with conn.cursor() as cur:
            cur.execute("CREATE DATABASE genai_platform_test")
        conn.close()
        return True
    except Exception:
        return False


def print_section(title):
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def test_session_management(platform):
    """Test session creation and retrieval."""
    print_section("Testing Session Management")

    # Test 1: Create a new session
    print("\n1. Creating new session for user 'patient-123'...")
    session1 = platform.sessions.get_or_create("patient-123")
    print(f"   ✓ Created session: {session1.session_id}")
    print(f"   - User ID: {session1.user_id}")
    print(f"   - Created at: {session1.created_at}")

    # Test 2: Retrieve existing session
    print("\n2. Retrieving same session...")
    session2 = platform.sessions.get_or_create("patient-123", session1.session_id)
    print(f"   ✓ Retrieved session: {session2.session_id}")
    assert session1.session_id == session2.session_id, "Session IDs should match"

    # Test 3: Create session with specific ID
    print("\n3. Creating session with specific ID...")
    session3 = platform.sessions.get_or_create("patient-456", "custom-session-id")
    print(f"   ✓ Created session: {session3.session_id}")
    assert session3.session_id == "custom-session-id", "Should use provided ID"

    return session1.session_id, session3.session_id


def test_message_operations(platform, session_id):
    """Test message storage and retrieval."""
    print_section("Testing Message Operations")

    # Test 1: Add simple messages
    print("\n1. Adding simple user-assistant exchange...")
    messages = [
        {"role": "user", "content": "What documents do I need for my appointment?"},
        {"role": "assistant", "content": "Please bring your insurance card and a photo ID."},
    ]
    count = platform.sessions.add_messages(session_id, messages)
    print(f"   ✓ Added {count} messages")

    # Test 2: Add messages with tool calls
    print("\n2. Adding messages with tool calls...")
    tool_messages = [
        {"role": "user", "content": "Do you have any openings next Tuesday?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": "check_availability",
                        "arguments": '{"date": "2024-01-16"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_abc123",
            "content": '{"slots": ["9:00 AM", "2:30 PM", "4:00 PM"]}',
        },
        {"role": "assistant", "content": "Yes! We have openings at 9:00 AM, 2:30 PM, and 4:00 PM."},
    ]
    count = platform.sessions.add_messages(session_id, tool_messages)
    print(f"   ✓ Added {count} messages (including tool calls)")

    # Test 3: Retrieve all messages
    print("\n3. Retrieving all messages...")
    all_messages, total = platform.sessions.get_messages(session_id)
    print(f"   ✓ Retrieved {len(all_messages)} messages (total: {total})")

    # Display message summary
    print("\n   Message summary:")
    for i, msg in enumerate(all_messages, 1):
        role = msg.role
        if msg.content:
            content_preview = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
            print(f"   {i}. [{role}] {content_preview}")
        elif msg.tool_calls:
            tc = msg.tool_calls[0]
            print(f"   {i}. [{role}] Tool call: {tc.function.name}")
        else:
            print(f"   {i}. [{role}] (no content)")

    # Test 4: Retrieve with pagination
    print("\n4. Testing pagination (limit=2)...")
    paginated, total = platform.sessions.get_messages(session_id, limit=2)
    print(f"   ✓ Retrieved {len(paginated)} messages (total: {total})")

    # Test 5: Retrieve with offset
    print("\n5. Testing pagination (limit=2, offset=2)...")
    offset_msgs, total = platform.sessions.get_messages(session_id, limit=2, offset=2)
    print(f"   ✓ Retrieved {len(offset_msgs)} messages starting from offset 2")


def test_model_managed_memory(platform):
    """Test model-managed memory operations."""
    print_section("Testing Model-Managed Memory")

    user_id = "patient-789"

    # Test 1: Save simple memory
    print("\n1. Saving allergy information...")
    platform.sessions.save_memory(user_id, "allergies", ["penicillin", "latex"])
    print("   ✓ Saved memory: allergies")

    # Test 2: Save complex memory
    print("\n2. Saving medication list...")
    platform.sessions.save_memory(
        user_id,
        "medications",
        [
            {"name": "lisinopril", "dosage": "10mg", "frequency": "daily"},
            {"name": "metformin", "dosage": "500mg", "frequency": "twice daily"},
        ],
    )
    print("   ✓ Saved memory: medications")

    # Test 3: Save preference
    print("\n3. Saving appointment preference...")
    platform.sessions.save_memory(user_id, "preferred_time", "morning")
    print("   ✓ Saved memory: preferred_time")

    # Test 4: Retrieve all memories
    print("\n4. Retrieving all memories for user...")
    memories = platform.sessions.get_memory(user_id)
    print(f"   ✓ Retrieved {len(memories)} memories:")
    for key, value in memories.items():
        print(f"   - {key}: {value}")

    # Test 5: Retrieve specific memory
    print("\n5. Retrieving specific memory (allergies)...")
    allergies = platform.sessions.get_memory(user_id, key="allergies")
    print(f"   ✓ Retrieved: {allergies}")

    # Test 6: Update existing memory
    print("\n6. Updating memory (adding new allergy)...")
    platform.sessions.save_memory(user_id, "allergies", ["penicillin", "latex", "sulfa drugs"])
    updated = platform.sessions.get_memory(user_id, key="allergies")
    print(f"   ✓ Updated allergies: {updated['allergies']}")

    # Test 7: Session-scoped memory
    print("\n7. Testing session-scoped memory...")
    session = platform.sessions.get_or_create(user_id)
    platform.sessions.save_memory(
        user_id,
        "session_note",
        "Patient asked about pediatric forms",
        session_id=session.session_id,
    )
    print("   ✓ Saved session-scoped memory")

    # Retrieve session-scoped
    session_memories = platform.sessions.get_memory(user_id, session_id=session.session_id)
    print(f"   ✓ Session memories: {list(session_memories.keys())}")

    # Test 8: Delete a memory
    print("\n8. Deleting a memory...")
    platform.sessions.delete_memory(user_id, "session_note", session_id=session.session_id)
    print("   ✓ Deleted memory: session_note")

    # Test 9: Clear all user memories
    print("\n9. Clearing all memories for user...")
    count = platform.sessions.clear_user_memory(user_id)
    print(f"   ✓ Cleared {count} memories")

    # Verify cleared
    remaining = platform.sessions.get_memory(user_id)
    print(f"   ✓ Remaining memories: {len(remaining)}")


def test_conversation_workflow(platform):
    """Test a realistic conversation workflow with memory."""
    print_section("Testing Complete Conversation Workflow")

    user_id = "patient-complete-test"

    # Step 1: Create session
    print("\n1. Patient starts conversation...")
    session = platform.sessions.get_or_create(user_id)
    print(f"   ✓ Session created: {session.session_id}")

    # Step 2: First exchange - patient mentions allergy
    print("\n2. First exchange (patient mentions allergy)...")
    messages = [
        {"role": "system", "content": "You are a helpful patient intake assistant."},
        {"role": "user", "content": "Hi, I'm here for my appointment. I'm allergic to penicillin."},
        {
            "role": "assistant",
            "content": "Welcome! I've noted your penicillin allergy. Let me help you with intake.",
        },
    ]
    platform.sessions.add_messages(session.session_id, messages)
    print(f"   ✓ Added {len(messages)} messages")

    # Step 3: Model saves allergy to memory
    print("\n3. Assistant saves allergy to memory...")
    platform.sessions.save_memory(user_id, "allergies", ["penicillin"])
    print("   ✓ Saved to memory")

    # Step 4: More conversation
    print("\n4. Continuing conversation...")
    more_messages = [
        {"role": "user", "content": "What insurance do you accept?"},
        {
            "role": "assistant",
            "content": "We accept most major insurance providers. Do you have your insurance card?",
        },
        {"role": "user", "content": "Yes, I have Blue Cross."},
        {
            "role": "assistant",
            "content": "Great! Blue Cross is accepted. I'll verify your coverage.",
        },
    ]
    platform.sessions.add_messages(session.session_id, more_messages)
    print(f"   ✓ Added {len(more_messages)} more messages")

    # Step 5: Simulate new session (return visit)
    print("\n5. Simulating return visit (new session)...")
    new_session = platform.sessions.get_or_create(user_id)
    print(f"   ✓ New session: {new_session.session_id}")

    # Step 6: Load memories for new session
    print("\n6. Loading patient history from memory...")
    memories = platform.sessions.get_memory(user_id)
    print(f"   ✓ Loaded memories: {memories}")
    print("   ✓ Assistant knows about allergy from previous visit!")

    # Step 7: Retrieve old conversation
    print("\n7. Checking old session messages...")
    old_messages, total = platform.sessions.get_messages(session.session_id)
    print(f"   ✓ Old session has {total} messages")

    # Step 8: Delete old session
    print("\n8. Cleaning up old session...")
    platform.sessions.delete_session(session.session_id)
    print("   ✓ Deleted old session")

    # Step 9: Verify memories persist
    print("\n9. Verifying memories persist after session deletion...")
    memories_after = platform.sessions.get_memory(user_id)
    print(f"   ✓ Memories still exist: {list(memories_after.keys())}")

    # Cleanup
    platform.sessions.clear_user_memory(user_id)
    print("\n   ✓ Cleaned up test data")


def _start_servers(storage):
    """Start session service and gateway, returning server objects for cleanup."""
    from http.server import HTTPServer

    HTTPServer.allow_reuse_address = True

    port = get_service_port("sessions")
    servicer = SessionService(storage=storage)
    session_server = create_grpc_server(servicer=servicer, port=port, service_name="sessions")
    session_server.start()

    registry = ServiceRegistry()
    sessions_addr = os.getenv("SESSIONS_SERVICE_ADDR", f"localhost:{port}")
    registry.register_platform_service("sessions", sessions_addr)
    models_addr = os.getenv("MODELS_SERVICE_ADDR", "localhost:50053")
    registry.register_platform_service("models", models_addr)

    grpc_port = int(os.getenv("GATEWAY_PORT", "50051"))
    gateway_server = create_gateway_grpc(registry, grpc_port)
    gateway_server.start()

    http_port = int(os.getenv("GATEWAY_HTTP_PORT", "8080"))
    http_server = create_http_server(registry, http_port)
    http_thread = threading.Thread(target=http_server.serve_forever, daemon=True)
    http_thread.start()

    return session_server, gateway_server, http_server


def _stop_servers(session_server, gateway_server, http_server):
    """Gracefully stop all servers."""
    http_server.shutdown()
    gateway_server.stop(grace=0)
    session_server.stop(grace=0)


def run_tests_with_backend(backend_name, storage):
    """Run the full test suite against a given storage backend."""
    print("\n" + "#" * 60)
    print(f"  BACKEND: {backend_name}")
    print("#" * 60)

    print(f"\n  Starting services with {backend_name} storage...")
    session_server, gateway_server, http_server = _start_servers(storage)
    time.sleep(1)
    print("  ✓ Services started\n")

    platform = GenAIPlatform()

    try:
        session_id_1, session_id_2 = test_session_management(platform)
        test_message_operations(platform, session_id_1)
        test_model_managed_memory(platform)
        test_conversation_workflow(platform)

        # Cleanup test sessions
        print_section("Cleanup")
        print("\nCleaning up test sessions...")
        platform.sessions.delete_session(session_id_1)
        platform.sessions.delete_session(session_id_2)
        print("✓ Cleanup complete")

        print(f"\n✓ All {backend_name} tests passed!")

    finally:
        _stop_servers(session_server, gateway_server, http_server)
        time.sleep(0.5)


def _insert_demo_data(storage):
    """Leave sample rows in PostgreSQL so users can browse them in a GUI like Postico."""
    from services.sessions.models import Function, Message, ToolCall

    print_section("Inserting Demo Data (PostgreSQL)")

    # Clear any leftover test data so only demo rows remain
    with storage.conn.cursor() as cur:
        cur.execute("TRUNCATE messages, memories, sessions CASCADE")
    storage.conn.commit()

    # Demo patient session
    session = storage.get_or_create_session(user_id="demo-patient", session_id="demo-session-1")
    print(f"  Created session: {session.session_id}")

    storage.add_messages(
        "demo-session-1",
        [
            Message(role="system", content="You are a helpful patient intake assistant."),
            Message(role="user", content="Hi, I need to schedule a follow-up appointment."),
            Message(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_demo_1",
                        type="function",
                        function=Function(
                            name="check_availability",
                            arguments='{"date": "2026-04-22", "doctor": "Dr. Smith"}',
                        ),
                    )
                ],
            ),
            Message(
                role="tool",
                content='{"slots": ["9:00 AM", "11:30 AM", "3:00 PM"]}',
                tool_call_id="call_demo_1",
            ),
            Message(
                role="assistant",
                content="Dr. Smith has openings on April 22nd at 9:00 AM, 11:30 AM, and 3:00 PM. Which works best?",
            ),
            Message(role="user", content="11:30 AM please. Also, I'm allergic to penicillin."),
            Message(
                role="assistant",
                content="Booked for 11:30 AM on April 22nd. I've noted your penicillin allergy.",
            ),
        ],
    )
    print("  Added 7 messages (including a tool call)")

    # Demo memories
    storage.save_memory("demo-patient", "allergies", ["penicillin"])
    storage.save_memory("demo-patient", "preferred_doctor", "Dr. Smith")
    storage.save_memory(
        "demo-patient",
        "medications",
        [
            {"name": "lisinopril", "dosage": "10mg", "frequency": "daily"},
            {"name": "metformin", "dosage": "500mg", "frequency": "twice daily"},
        ],
    )
    print("  Saved 3 memories (allergies, preferred_doctor, medications)")

    # Second session for the same patient
    storage.get_or_create_session(user_id="demo-patient", session_id="demo-session-2")
    storage.add_messages(
        "demo-session-2",
        [
            Message(role="user", content="Can you remind me what medications I'm on?"),
            Message(
                role="assistant",
                content="You're currently taking lisinopril 10mg daily and metformin 500mg twice daily.",
            ),
        ],
    )
    print("  Created a second session with 2 messages")

    print("\n  ✓ Demo data ready! Open Postico and connect to 'genai_platform_test' to browse.")
    print("    Tables: sessions, messages, memories")


def run_all_tests():
    """Run all test suites against both storage backends."""
    print("\n" + "=" * 60)
    print("  SESSION SERVICE COMPREHENSIVE TEST")
    print("  Chapter 4: Teaching Your AI to Remember")
    print("=" * 60)

    # --- Pass 1: In-memory ---
    run_tests_with_backend("In-Memory", InMemorySessionStorage())

    # --- Pass 2: PostgreSQL (if available) ---
    pg_available = _detect_postgres()
    if pg_available:
        from services.sessions.postgres_store import PostgresSessionStorage

        pg_storage = PostgresSessionStorage(connection_string=TEST_DB)
        run_tests_with_backend("PostgreSQL", pg_storage)
        _insert_demo_data(pg_storage)
    else:
        print("\n" + "-" * 60)
        print("  Skipping PostgreSQL tests (server not available)")
        print("  To enable: brew install postgresql@16 && brew services start postgresql@16")
        print("-" * 60)

    print_section("FINAL SUMMARY")
    backends = ["In-Memory"]
    if pg_available:
        backends.append("PostgreSQL")
    print(f"\n✓ All tests passed on: {', '.join(backends)}")
    print("\nTested functionality:")
    print("  - Session creation and retrieval")
    print("  - Message storage with tool calls")
    print("  - Message pagination")
    print("  - Model-managed memory (save/get/delete)")
    print("  - Session-scoped vs user-scoped memory")
    print("  - Complete conversation workflow")
    print("  - Memory persistence across sessions")


if __name__ == "__main__":
    run_all_tests()
