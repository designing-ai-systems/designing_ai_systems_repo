# Designing AI systems repository

Production-ready platform for building GenAI applications with multi-provider support. Accompanies the book [**Designing AI Systems**](https://www.manning.com/books/designing-ai-systems) (Manning).

## Features

- **Multi-provider inference**: OpenAI, Anthropic (with streaming)
- **Session management**: Conversation history and model-managed memory
- **Tool management**: Registration, discovery, versioning, sandboxed execution
- **Guardrails**: Input validation, output filtering, policy enforcement
- **Domain dataclasses**: Clean Python API -- never exposes Protocol Buffers
- **Model discovery**: Query capabilities, register custom models
- **Prompt registry**: Centralized system prompt management
- **Storage abstraction**: In-memory (dev) or PostgreSQL (production)
- **Service architecture**: gRPC microservices with unified API Gateway

## Requirements

- **Python 3.12+** (macOS: `brew install python@3.12`)

## Setup

```bash
# 1. Create virtual environment (Python 3.12+)
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

# 6. Lint (runs in CI on every PR)
ruff check .
ruff format --check .
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

**Tools & Guardrails (Chapter 6):**
```bash
python examples/quickstart_tools.py
```

**Run services separately** (optional):
```bash
python -m services.sessions.main    # Terminal 1
python -m services.models.main      # Terminal 2
python -m services.tools.main       # Terminal 3
python -m services.guardrails.main  # Terminal 4
python -m services.gateway.main     # Terminal 5
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

### Tool Service

```python
from genai_platform import GenAIPlatform
from services.tools.models import ToolBehavior, RateLimits

platform = GenAIPlatform()

# Register a tool with operational metadata (Listing 6.4, 6.8)
platform.tools.register(
    name="healthcare.scheduling.book_appointment",
    description="Book a patient appointment",
    behavior=ToolBehavior(is_read_only=False, requires_confirmation=True),
    rate_limits=RateLimits(requests_per_session=3),
    capabilities=["scheduling", "booking"],
    tags=["patient-facing", "hipaa-compliant"],
)

# Discover tools by capability (Listing 6.7)
tools = platform.tools.discover(capabilities=["scheduling"])
for t in tools:
    print(f"{t.name} (read_only={t.behavior.is_read_only})")

# Execute a tool (Listing 6.12)
result = platform.tools.execute(
    tool_name="healthcare.scheduling.book_appointment",
    arguments={"patient_id": "p-123", "datetime": "2026-04-15T10:00:00Z"},
)
print(result.success, result.result)
```

### Guardrails Service

```python
# Validate input (Listing 6.20, 6.21)
result = platform.guardrails.validate_input(
    content="Schedule an appointment for tomorrow",
    checks=["prompt_injection", "pii_detection"],
)
print(result["allowed"])  # True

# Filter output -- redacts PII (Listing 6.23)
result = platform.guardrails.filter_output(
    content="Patient SSN: 123-45-6789",
    filters=["pii_redaction"],
)
print(result["content"])  # "Patient SSN: [REDACTED]"

# Policy check (Listing 6.19)
result = platform.guardrails.check_policy(
    policy_name="booking-rules",
    action="book_appointment",
    context={"referral_id": "ref-123"},
)
print(result.allowed)
```

## Supported Models

**OpenAI**: `gpt-4o`, `gpt-4o-mini`
**Anthropic**: `claude-sonnet-4-5`, `claude-opus-4-5`, `claude-haiku-4-5`

## Architecture

```
genai_platform/
├── genai_platform/            # SDK (public API)
│   ├── platform.py            # GenAIPlatform entry point
│   └── clients/               # Service clients
│       ├── sessions.py        #   SessionClient
│       ├── models.py          #   ModelClient (with fallback)
│       ├── tools.py           #   ToolClient (register, discover, execute)
│       └── guardrails.py      #   GuardrailsClient (validate, filter, policy)
├── proto/                     # Protocol Buffer definitions
│   ├── sessions.proto         # Session Service contract
│   ├── models.proto           # Model Service contract
│   ├── tools.proto            # Tool Service contract
│   └── guardrails.proto       # Guardrails Service contract
├── services/
│   ├── gateway/               # API Gateway (gRPC proxy)
│   ├── sessions/              # Session Service
│   │   ├── models.py          #   Domain dataclasses (Session, Message, ...)
│   │   ├── store.py           #   Storage ABC + InMemorySessionStorage
│   │   ├── postgres_store.py  #   PostgreSQL implementation
│   │   ├── schema.sql         #   Database schema
│   │   └── service.py         #   gRPC servicer
│   ├── models/                # Model Service
│   │   ├── models.py          #   Domain dataclasses (ChatResponse, ...)
│   │   ├── service.py         #   gRPC servicer
│   │   └── providers/         #   Provider adapters (OpenAI, Anthropic)
│   ├── tools/                 # Tool Service (Chapter 6)
│   │   ├── models.py          #   Domain dataclasses (ToolDefinition, ...)
│   │   ├── store.py           #   ToolRegistry ABC + InMemoryToolRegistry
│   │   ├── credential_store.py#   CredentialStore ABC + InMemoryCredentialStore
│   │   ├── circuit_breaker.py #   CircuitBreaker (closed/open/half-open)
│   │   └── service.py         #   gRPC servicer
│   └── guardrails/            # Guardrails Service (Chapter 6)
│       ├── models.py          #   Domain dataclasses (PolicyResult, ...)
│       ├── store.py           #   PolicyStore ABC + InMemoryPolicyStore
│       └── service.py         #   gRPC servicer
├── tests/                     # Unit tests (pytest)
└── examples/                  # Runnable demo scripts
```

### Book Listing Cross-Reference

Each source file maps to specific listings in [Designing AI Systems](https://www.manning.com/books/designing-ai-systems):

| File | Book Listings |
|------|--------------|
| **Model Service (Chapter 3)** | |
| `proto/models.proto` | 3.5 (service def), 3.6 (ChatRequest), 3.7 (ChatResponse), 3.10 (ChatChunk) |
| `services/models/models.py` | 3.1-3.4 (chat types), 3.11 (RetryConfig), 3.12 (FallbackConfig), 3.13 (RoutingConfig), 3.14 (RateLimitConfig), 3.15 (CacheConfig), 3.16 (RequestMetrics) |
| `services/models/providers/base.py` | 3.8 (ModelProvider ABC) |
| `services/models/providers/anthropic_provider.py` | 3.9 (Anthropic adapter) |
| `services/models/providers/openai_provider.py` | 3.9 pattern (OpenAI adapter) |
| `genai_platform/clients/models.py` | 3.17 (ModelClient init), 3.18 (chat method), 3.19 (chat_stream) |
| `examples/quickstart_models.py` | 3.20 (complete workflow) |
| **Session Service (Chapter 4)** | |
| `proto/sessions.proto` | 4.3 (service def), 4.4 (session msgs), 4.5 (add msgs), 4.6 (get msgs), 4.7 (Message type) |
| `services/sessions/models.py` | 4.1 (Message), 4.2 (Session), 4.19 (MemoryEntry) |
| `services/sessions/store.py` | 4.8 (SessionStorage ABC), 4.20 (memory methods) |
| `services/sessions/schema.sql` | 4.9 (sessions table), 4.10 (messages table) |
| `services/sessions/postgres_store.py` | 4.11-4.14 (PostgreSQL implementation) |
| `services/sessions/service.py` | 4.15 (gRPC servicer) |
| `genai_platform/clients/sessions.py` | 4.16 (SessionClient setup), 4.17 (get_or_create), 4.21 (memory methods) |
| `examples/quickstart_conversation.py` | 4.18 (session workflow), 4.22 (memory workflow) |
| **Tool Service (Chapter 6)** | |
| `proto/tools.proto` | 6.1 (ToolService contract), 6.2 (ToolDefinition), 6.3 (ToolBehavior/RateLimits/CostMetadata), 6.7 (DiscoverToolsRequest), 6.17 (ExecutionLimits) |
| `services/tools/models.py` | 6.2 (ToolDefinition), 6.3 (ToolBehavior, RateLimits, CostMetadata), 6.15 (ToolTask), 6.17 (ExecutionLimits) |
| `services/tools/store.py` | 6.4 (registration), 6.5 (discovery by namespace), 6.7 (capability search), 6.10 (version constraints) |
| `services/tools/credential_store.py` | 6.13 (credential ref), 6.14 (CredentialStore interface) |
| `services/tools/circuit_breaker.py` | 6.18 (CircuitBreaker: closed/open/half-open) |
| `services/tools/service.py` | 6.1 (gRPC servicer), 6.4 (register), 6.7 (discover), 6.12 (execute), 6.18 (circuit breaker) |
| `genai_platform/clients/tools.py` | 6.4 (register), 6.7 (discover), 6.11 (deprecate), 6.12 (execute) |
| **Guardrails Service (Chapter 6)** | |
| `proto/guardrails.proto` | 6.19 (GuardrailsService contract) |
| `services/guardrails/models.py` | 6.19 (PolicyResult), 6.21 (GuardrailCheck), 6.23 (tiered handling) |
| `services/guardrails/store.py` | 6.21 (input config), 6.23 (tiered handlers), 6.25 (human approval gate) |
| `services/guardrails/service.py` | 6.19 (gRPC servicer), 6.20 (multi-point eval), 6.21 (input validation), 6.23 (output filtering) |
| `genai_platform/clients/guardrails.py` | 6.19 (policy check), 6.20 (validate input), 6.23 (filter output) |
| `examples/quickstart_tools.py` | 6.4, 6.7, 6.8, 6.12, 6.19, 6.20, 6.23 (end-to-end demo) |

### Key Design Principles

1. **Domain types at the core**: Business logic uses Python dataclasses, never Protocol Buffers.
2. **Proto at the boundary**: gRPC servicers convert between proto messages and domain types.
3. **Provider abstraction**: All LLM providers implement the same `ModelProvider` ABC with domain types.
4. **Storage abstraction**: `SessionStorage` / `ToolRegistry` / `PolicyStore` ABCs with swappable backends.
5. **SDK hides gRPC**: Clients return dataclasses to callers. Proto is an internal detail.
6. **Circuit breaker**: Tool execution protected by closed/open/half-open state machine (Listing 6.18).
7. **Credential isolation**: Tools reference credentials by name; secrets stored separately (Listing 6.13-6.14).

## Running Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

## Status

- **Model Service** (Chapter 3): OpenAI, Anthropic, streaming, prompt management, custom models, client-side fallback
- **Session Service** (Chapter 4): Messages, pagination, model-managed memory, PostgreSQL
- **Tool Service** (Chapter 6): Registration, discovery (namespace/capability/tags), versioning, sandboxed execution, circuit breaker, credential store
- **Guardrails Service** (Chapter 6): Input validation (prompt injection, PII), output filtering (PII redaction), policy enforcement, violation reporting
- **API Gateway**: gRPC proxy with service discovery (sessions, models, tools, guardrails)
- Data Service (Chapter 5): planned
- Evaluation Service (Chapter 8): planned
