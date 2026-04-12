# GenAI Platform

Production-ready platform for building GenAI applications with multi-provider support. Accompanies the book [**Designing AI Systems**](https://www.manning.com/books/designing-ai-systems) (Manning).

## Features

- **Multi-provider inference**: OpenAI, Anthropic (with streaming)
- **Session management**: Conversation history and model-managed memory
- **Domain dataclasses**: Clean Python API -- never exposes Protocol Buffers
- **Model discovery**: Query capabilities, register custom models
- **Prompt registry**: Centralized system prompt management
- **Storage abstraction**: In-memory (dev) or PostgreSQL (production)
- **Service architecture**: gRPC microservices with unified API Gateway

## Setup

```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install
pip install -e ".[dev]"

# 3. Configure API keys (create .env file)
echo "OPENAI_API_KEY=your-key" > .env
echo "ANTHROPIC_API_KEY=your-key" >> .env

# 4. Generate Protocol Buffer code (only needed after changing .proto files)
python -m proto.generate

# 5. Run tests
pytest tests/ -v
```

### Optional: PostgreSQL storage

By default the Session Service uses in-memory storage. To persist sessions
across restarts, use PostgreSQL.

#### macOS (Homebrew)

```bash
# Install PostgreSQL 16
brew install postgresql@16

# Start the server (runs in the background, auto-starts on login)
brew services start postgresql@16

# Create the database
createdb genai_platform

# Apply the schema
psql genai_platform < services/sessions/schema.sql
```

> **Tip:** If `psql` / `createdb` are not on your PATH after install, add the
> Homebrew PostgreSQL bin directory:
> ```bash
> echo 'export PATH="/opt/homebrew/opt/postgresql@16/bin:$PATH"' >> ~/.zshrc
> source ~/.zshrc
> ```

#### Install the Python driver and configure

```bash
pip install -e ".[postgres]"

# Set env vars before starting the session service:
export SESSION_STORAGE=postgres
export DB_CONNECTION_STRING="postgresql://localhost/genai_platform"
```

The session service will now read and write to PostgreSQL.

## Quick Start

**Model Service (Chapter 3):**
```bash
python examples/quickstart_models.py
```

**Session + Model Integration (Chapters 3-4):**
```bash
python examples/quickstart_conversation.py
```

**Run services separately** (optional):
```bash
python -m services.sessions.main  # Terminal 1
python -m services.models.main    # Terminal 2
python -m services.gateway.main   # Terminal 3
```

## Usage

### Model Service

```python
from genai_platform import GenAIPlatform

platform = GenAIPlatform()

# Chat -- returns ChatResponse dataclass
response = platform.models.chat(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.7,
    max_tokens=150,
)
print(response.content)       # attribute access, not dict
print(response.usage.total_tokens)

# Streaming -- yields ChatChunk dataclass
for chunk in platform.models.chat_stream(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
):
    print(chunk.token, end="", flush=True)

# Model discovery -- returns list of ModelInfo
models = platform.models.list_models()
for m in models:
    print(f"{m.name} ({m.provider})")
```

### Session Service

```python
# Create session -- returns Session dataclass
session = platform.sessions.get_or_create(user_id="user-123")

# List sessions
sessions = platform.sessions.list_sessions("user-123")

# Store conversation
platform.sessions.add_messages(session.session_id, [
    {"role": "user", "content": "What documents do I need?"},
    {"role": "assistant", "content": "You'll need ID and insurance."},
])

# Retrieve history -- returns list of Message dataclasses
messages, total = platform.sessions.get_messages(session.session_id, limit=20)
for msg in messages:
    print(f"[{msg.role}] {msg.content}")

# Model-managed memory
platform.sessions.save_memory("user-123", "allergies", ["penicillin"])
memories = platform.sessions.get_memory("user-123")
```

## Supported Models

**OpenAI**: `gpt-4o`, `gpt-4o-mini`
**Anthropic**: `claude-sonnet-4-5`, `claude-opus-4-5`, `claude-haiku-4-5`

## Architecture

```
genai_platform/
├── genai_platform/          # SDK (public API)
│   ├── platform.py          # GenAIPlatform entry point
│   └── clients/             # Service clients (sessions, models)
├── proto/                   # Protocol Buffer definitions
│   ├── sessions.proto       # Session Service contract
│   └── models.proto         # Model Service contract
├── services/
│   ├── gateway/             # API Gateway (gRPC proxy)
│   ├── sessions/            # Session Service
│   │   ├── models.py        #   Domain dataclasses (Session, Message, ...)
│   │   ├── store.py         #   Storage ABC + InMemorySessionStorage
│   │   ├── postgres_store.py#   PostgreSQL implementation
│   │   ├── schema.sql       #   Database schema
│   │   └── service.py       #   gRPC servicer (proto <-> domain boundary)
│   └── models/              # Model Service
│       ├── models.py        #   Domain dataclasses (ChatResponse, ChatChunk, ...)
│       ├── service.py       #   gRPC servicer
│       └── providers/       #   Provider adapters
│           ├── base.py      #     ModelProvider ABC (domain types)
│           ├── openai_provider.py
│           └── anthropic_provider.py
├── tests/                   # Unit tests (pytest)
└── examples/                # Runnable demo scripts
```

### Book Listing Cross-Reference

Each source file maps to specific listings in [Designing AI Systems](https://www.manning.com/books/designing-ai-systems):

| File | Book Listings |
|------|--------------|
| `services/sessions/models.py` | 4.1 (Message), 4.2 (Session), 4.19 (MemoryEntry) |
| `services/sessions/store.py` | 4.8 (SessionStorage ABC), 4.20 (memory methods) |
| `services/sessions/schema.sql` | 4.9 (sessions table), 4.10 (messages table) |
| `services/sessions/postgres_store.py` | 4.11 (PostgresSessionStorage), 4.12 (add_messages), 4.13 (get_messages), 4.14 (row converters) |
| `services/sessions/service.py` | 4.15 (gRPC servicer) |
| `genai_platform/clients/sessions.py` | 4.16 (SessionClient setup), 4.17 (get_or_create), 4.21 (memory methods) |
| `proto/sessions.proto` | 4.3 (service def), 4.4 (session msgs), 4.5 (add msgs), 4.6 (get msgs), 4.7 (Message type) |
| `services/models/models.py` | 3.1-3.4 (chat types), 3.11 (RetryConfig), 3.12 (FallbackConfig), 3.13 (RoutingConfig), 3.14 (RateLimitConfig), 3.15 (CacheConfig), 3.16 (RequestMetrics) |
| `services/models/providers/base.py` | 3.8 (ModelProvider ABC) |
| `services/models/providers/anthropic_provider.py` | 3.9 (Anthropic adapter) |
| `services/models/providers/openai_provider.py` | 3.9 pattern (OpenAI adapter) |
| `proto/models.proto` | 3.5 (service def), 3.6 (ChatRequest), 3.7 (ChatResponse), 3.10 (ChatChunk) |
| `genai_platform/clients/models.py` | 3.17 (ModelClient init), 3.18 (chat method), 3.19 (chat_stream) |
| `examples/quickstart_models.py` | 3.20 (complete workflow) |
| `examples/quickstart_conversation.py` | 4.18 (session workflow), 4.22 (memory workflow) |

### Key Design Principles

1. **Domain types at the core**: Business logic uses Python dataclasses, never Protocol Buffers.
2. **Proto at the boundary**: gRPC servicers convert between proto messages and domain types.
3. **Provider abstraction**: All LLM providers implement the same `ModelProvider` ABC with domain types.
4. **Storage abstraction**: `SessionStorage` ABC with swappable backends (in-memory, PostgreSQL).
5. **SDK hides gRPC**: Clients return dataclasses to callers. Proto is an internal detail.

## Running Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

## Status

- **Model Service** (Chapter 3): OpenAI, Anthropic, streaming, prompt management, custom models
- **Session Service** (Chapter 4): Messages, pagination, model-managed memory, PostgreSQL
- **API Gateway**: gRPC proxy with service discovery
- Data Service (Chapter 5): planned
- Tool Service (Chapter 6): planned
- Guardrails Service (Chapter 7): planned
- Evaluation Service (Chapter 8): planned
