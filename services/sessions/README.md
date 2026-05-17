# Session Service

The Session Service manages conversation history and model-managed memory, enabling context-aware AI interactions.

## Key Components

- **SessionStorage**: Abstract storage interface with in-memory and PostgreSQL implementations
- **Message Management**: Stores conversations with tool call support
- **Model-Managed Memory**: Key-value storage for critical facts that persist across sessions
- **Context Window Support**: Pagination and strategy foundation for managing long conversations

## Quick Start

### 1. Install Dependencies
```bash
uv sync
```

### 2. Generate Protocol Buffers
```bash
uv run python proto/generate.py
```

### 3. Run Tests

Comprehensive test suite:
```bash
uv run python examples/test_session_service.py
```

Or see Session + Model integration:
```bash
uv run python examples/quickstart_conversation.py
```

## Manual Service Startup

If you want to run services separately:

**Terminal 1 - Session Service**:
```bash
uv run python -m services.sessions.main
```

**Terminal 2 - API Gateway**:
```bash
uv run python -m services.gateway.main
```

**Terminal 3 - Run tests**:
```bash
uv run python examples/test_session_service.py
```

## Features

### Session Management
- **GetOrCreateSession**: Create new sessions or resume existing ones
- **DeleteSession**: Clean up completed conversations

### Message Operations
- **AddMessages**: Store conversation turns with tool call support
- **GetMessages**: Retrieve history with pagination (limit/offset)
- Full support for user, assistant, system, and tool messages
- Tool calls with proper ID linkage between requests and results

### Model-Managed Memory
- **SaveMemory**: Store critical facts (allergies, preferences, etc.)
- **GetMemory**: Load memories for context
- **DeleteMemory**: Remove specific entries
- **ClearUserMemory**: Delete all memories for a user
- User-scoped and session-scoped storage

## Storage Backends

### In-Memory (Default)
Works out-of-the-box with no dependencies. Perfect for development and testing.

```bash
# No configuration needed - this is the default
python examples/test_session_service.py
```

### PostgreSQL (Production)
For production deployments with persistence and reliability.

**Setup:**
```bash
# Install with the postgres extra so the driver lands in the venv.
uv sync --extra postgres

# Set environment variables
export SESSION_STORAGE=postgres
export DB_CONNECTION_STRING=postgresql://localhost/genai_platform
```

**Database Setup:**
```bash
# Create database
createdb genai_platform

# Tables are auto-created on first run
```

**Schema:**
- `sessions`: Session metadata (user_id, timestamps)
- `messages`: Conversation history with tool calls (JSONB)
- `memories`: Key-value storage for model-managed facts

> **Note**: PostgreSQL testing to be added. In-memory storage is fully tested and production-ready for stateless deployments.

## Examples

### Basic Session Management
```python
from genai_platform import GenAIPlatform

platform = GenAIPlatform()

# Create/retrieve session
session = platform.sessions.get_or_create("user-123")

# Add conversation
platform.sessions.add_messages(session.session_id, [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"}
])

# Get history with pagination
messages, total = platform.sessions.get_messages(
    session.session_id,
    limit=20
)
```

### Tool Call Messages
```python
# Store complete tool call flow
messages = [
    {"role": "user", "content": "Check my appointment"},
    {
        "role": "assistant",
        "tool_calls": [{
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "check_appointment",
                "arguments": '{"user_id": "123"}'
            }
        }]
    },
    {
        "role": "tool",
        "tool_call_id": "call_123",
        "content": '{"time": "2pm", "status": "confirmed"}'
    },
    {"role": "assistant", "content": "Your appointment is at 2pm."}
]

platform.sessions.add_messages(session.session_id, messages)
```

### Model-Managed Memory
```python
# Save critical facts
platform.sessions.save_memory("patient-123", "allergies", ["penicillin"])
platform.sessions.save_memory("patient-123", "medications", [
    {"name": "lisinopril", "dosage": "10mg"}
])

# Load for next conversation
memories = platform.sessions.get_memory("patient-123")
# Returns: {"allergies": [...], "medications": [...]}

# Use in system prompt
system_prompt = f"""You are a helpful medical assistant.
Patient allergies: {memories.get('allergies', [])}
Current medications: {memories.get('medications', [])}
"""
```

### Complete Workflow
```python
# Healthcare assistant pattern
platform = GenAIPlatform()

# Get session
session = platform.sessions.get_or_create(patient_id)

# Load patient memories and recent conversation
memories = platform.sessions.get_memory(patient_id)
messages, _ = platform.sessions.get_messages(session.session_id, limit=20)

# Build context
context = build_prompt(memories, messages, new_question)

# Generate response (Chapter 3 Model Service)
response = platform.models.generate(model="gpt-4o", messages=context)

# Save conversation turn
platform.sessions.add_messages(session.session_id, [
    {"role": "user", "content": new_question},
    {"role": "assistant", "content": response.text}
])

# Model decides to save important fact
if important_fact:
    platform.sessions.save_memory(patient_id, fact_key, fact_value)
```

See `examples/quickstart_session_service.py` and `examples/test_session_service.py` for more.

## Context Window Management

The service provides foundation for Chapter 4 context window strategies:

- **Pagination**: Retrieve specific ranges of messages
- **Strategy Parameter**: Ready for truncation, summarization, hierarchical memory
- **Total Count Tracking**: Know conversation length for smart decisions

Future enhancements:
- Automatic summarization of older messages
- Hierarchical memory (facts + summary + recent)
- Retrieval-augmented search across past sessions

## Port

Default: `50052` (override with `SESSIONS_PORT` env var)

## Related Services

- **Model Service**: Generate responses (Chapter 3)
- **Data Service**: Search and retrieval (Chapter 5)
- **API Gateway**: Routes requests to services
