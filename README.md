# Designing AI systems repository

Production-ready platform for building GenAI applications with multi-provider support. Accompanies the book [**Designing AI Systems**](https://www.manning.com/books/designing-ai-systems) (Manning).

## Features

- **Multi-provider inference**: OpenAI, Anthropic (with streaming)
- **Session management**: Conversation history and model-managed memory
- **Data & RAG pipeline**: Document ingestion, chunking, embedding, vector search, hybrid search
- **Tool management**: Registration, discovery, versioning, sandboxed execution
- **Guardrails**: Input validation, output filtering, policy enforcement
- **Domain dataclasses**: Clean Python API -- never exposes Protocol Buffers
- **Model discovery**: Query capabilities, register custom models
- **Prompt registry**: Centralized system prompt management
- **Storage abstraction**: In-memory (dev) or PostgreSQL + pgvector (production)
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

By default the Session Service uses in-memory storage. To persist sessions and
enable the Data Service's vector search, use PostgreSQL with the `pgvector`
extension. Both services share one database (separate tables) and one server.

#### macOS (Homebrew)

```bash
# Install PostgreSQL 17 and pgvector (required for Data Service)
brew install postgresql@17 pgvector

# Start the server (auto-starts on login)
brew services start postgresql@17

# Create the database and apply both schemas
createdb genai_platform
psql genai_platform < services/sessions/schema.sql
psql genai_platform < services/data/schema.sql
```

> **Tip:** If `psql` / `createdb` are not on your PATH, link PostgreSQL 17
> (it is keg-only by default):
> ```bash
> brew link --force postgresql@17
> ```

#### Install the Python driver and configure

```bash
pip install -e ".[postgres]"

# Set env vars before starting the services:
export SESSION_STORAGE=postgres
export VECTOR_STORE=pgvector
export DB_CONNECTION_STRING="postgresql://localhost/genai_platform"
```

Both services will now read and write to PostgreSQL. Running a local
PostgreSQL 17 with `pgvector` also enables the `test_data_comprehensive.py`
pgvector tests to run locally (mirroring how the Session Service tests run
against your local server).

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
python examples/quickstart_tools.py               # full platform end-to-end (+ model loop if OPENAI_API_KEY set)
python examples/test_tool_service.py              # tool service: register / discover / HTTP exec + credential injection / async / circuit breaker
python examples/test_tool_service.py --mcp        # same, plus a live MCP call to https://mcp.deepwiki.com/mcp
python examples/test_guardrails_service.py        # guardrails: input validation, output filtering, policy check, violation reporting
python examples/quickstart_mcp.py                 # platform registers DeepWiki (public MCP server) and runs real MCP calls
```

### Local development with Docker (recommended)

`docker compose up` brings up the seven platform services + a
pgvector-enabled Postgres on a shared network, in one command:

```bash
# (one-time) build the per-service images
docker compose build

# bring everything up
docker compose up -d

# in a separate terminal, deploy + test a workflow
genai-platform deploy examples/quickstart_workflow.py
curl -X POST http://localhost:8080/patient-assistant \
     -H 'Content-Type: application/json' \
     -d '{"question":"What documents do I need?","patient_id":"p-1"}'
```

The gateway is the only service whose ports are mapped to the host
(`8080` for external HTTP, `50051` for the SDK's gRPC channel).
Inter-service traffic is private to the compose network.

The Workflow Service mounts the host's docker socket so
`genai-platform deploy` can launch new workflow containers onto the same
compose network. The gateway then reaches each workflow container by its
container name — no host port mapping needed for individual workflows.

#### Run services separately, without Docker

If you can't (or don't want to) use Docker, run each platform service
in its own terminal:

```bash
python -m services.sessions.main    # Terminal 1
python -m services.models.main      # Terminal 2
python -m services.data.main        # Terminal 3
python -m services.tools.main       # Terminal 4
python -m services.guardrails.main  # Terminal 5
python -m services.workflow.main    # Terminal 6
python -m services.gateway.main     # Terminal 7
```

In this mode, `genai-platform deploy` falls back to host-port mode: each
new workflow container gets a free host port and the gateway reaches it
at `localhost:<port>`.

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

### Data Service

```python
from genai_platform import GenAIPlatform
from services.data.models import IndexConfig

platform = GenAIPlatform()

# Create an index
config = IndexConfig(name="company-docs", chunking_strategy="fixed", chunk_size=512)
index = platform.data.create_index(config, owner="team-a")

# Ingest a document (async -- returns an IngestJob)
job = platform.data.ingest("company-docs", "handbook.txt", b"...", metadata={"dept": "hr"})

# Poll for completion
status = platform.data.get_ingest_status(job.job_id)
print(status.status)  # "queued" → "processing" → "completed"

# Semantic search
results = platform.data.search("company-docs", query="vacation policy", top_k=5)
for r in results:
    print(f"[{r.score:.2f}] {r.text[:100]}")

# Hybrid search (vector + keyword via Reciprocal Rank Fusion)
results = platform.data.hybrid_search("company-docs", query="vacation policy")

# Register a custom parser (dynamic code loading)
platform.data.register_parser("custom-fmt", my_parser_instance)
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

### Workflow Service (Chapter 8)

Decorate a function with `@workflow`, run `genai-platform deploy`, and the
function becomes an HTTP service routed by the gateway. Three response
modes; the SDK's `platform.workflows.call(...)` auto-detects which one a
child workflow uses, so parent code never branches on it.

```python
from genai_platform import GenAIPlatform, workflow

# Listings 8.1, 8.2, 8.4: a sync workflow.
@workflow(
    name="patient_intake_assistant",
    api_path="/patient-assistant",
    response_mode="sync",
    min_replicas=1,
    max_replicas=10,
    target_cpu_percent=70,
    cpu="500m",
    memory="512Mi",
    timeout_seconds=15,
)
def handle(question: str, patient_id: str) -> dict:
    platform = GenAIPlatform()
    # ... orchestrate sessions / models / data / guardrails ...
    return {"patient_id": patient_id, "answer": "..."}
```

**Deploy locally** (chapter-8 demo flow):

```bash
# 1. Bring up the platform services
docker compose up -d         # (lands in commit 4 — for now: python -m services.X.main)

# 2. Build the image, register, run the container, register the route.
genai-platform deploy examples/quickstart_workflow.py

# 3. Hit the workflow through the gateway.
curl -X POST http://localhost:8080/patient-assistant \
     -H 'Content-Type: application/json' \
     -d '{"question": "What documents do I need?", "patient_id": "p-1"}'
```

`genai-platform deploy` writes `Dockerfile`, `Deployment.yaml`,
`HorizontalPodAutoscaler.yaml`, and `Service.yaml` to
`build/<workflow-name>/`. The Kubernetes manifests are
**ready-to-apply artifacts** — the CLI does not invoke `kubectl`. To
deploy to a real cluster (EKS / GKE / minikube / ...):

```bash
# Push the image to your registry, then:
kubectl apply -f build/patient_intake_assistant/
```

**Composition** (Listings 8.15-8.18):

```python
@workflow(name="research_assistant", api_path="/research-assistant", response_mode="sync")
def parent(topic: str) -> dict:
    platform = GenAIPlatform()
    papers, news = platform.workflows.call_parallel([
        ("/papers", {"topic": topic}),
        ("/news", {"topic": topic}),
    ])
    return {"papers": papers, "news": news}
```

**Async + polling** (Listings 8.7, 8.10, 8.11):

```python
@workflow(name="deep_researcher", api_path="/research", response_mode="async")
def deep_research(topic: str, depth: int = 3) -> dict:
    platform = GenAIPlatform()
    platform.workflows.update_job_progress(message="phase 1/3: gathering")
    # ... long-running work, with checkpoint() between phases ...
    return {"summary": "..."}

# Client side:
# POST /research → 202 + {"job_id": "...", "status_url": "/jobs/..."}
# GET /jobs/{id} → eventually {"job": {"status": "succeeded", "result_json": ...}}
```

Runnable end-to-end demos:
- `examples/quickstart_workflow.py` — sync (Listings 8.1, 8.4)
- `examples/quickstart_workflow_stream.py` — streaming SSE (Listings 8.5, 8.6)
- `examples/quickstart_workflow_async.py` — async + checkpoint + polling (Listings 8.7, 8.8, 8.10, 8.11)
- `examples/quickstart_workflow_compose.py` — `call_parallel` across children (Listings 8.15-8.18)

**Prerequisite for chapter-8 demos:** Docker installed (the deploy CLI
calls `docker build` and the Workflow Service runs containers via
`docker run`).

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
│       ├── data.py            #   DataClient (indexes, ingest, search)
│       ├── tools.py           #   ToolClient (register, discover, execute)
│       └── guardrails.py      #   GuardrailsClient (validate, filter, policy)
├── proto/                     # Protocol Buffer definitions
│   ├── sessions.proto         # Session Service contract
│   ├── models.proto           # Model Service contract
│   ├── data.proto             # Data Service contract
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
│   ├── data/                  # Data Service (Chapter 5)
│   │   ├── models.py          #   Domain dataclasses (Index, Chunk, SearchResult, ...)
│   │   ├── parsers.py         #   DocumentParser ABC + PlainText, Markdown parsers
│   │   ├── chunking.py        #   ChunkingStrategy ABC + Fixed, Recursive, StructureAware
│   │   ├── embedding.py       #   EmbeddingGenerator (wraps Model Service)
│   │   ├── store.py           #   VectorStore ABC + InMemoryVectorStore
│   │   ├── pgvector_store.py  #   PostgreSQL + pgvector implementation
│   │   ├── schema.sql         #   Database schema (pgvector, full-text search)
│   │   ├── search.py          #   SearchOrchestrator + Reciprocal Rank Fusion
│   │   ├── pipeline.py        #   IngestionPipeline (parse → chunk → embed → store)
│   │   └── service.py         #   gRPC servicer (proto <-> domain boundary)
│   ├── tools/                 # Tool Service (Chapter 6, grpc.aio)
│   │   ├── models.py          #   Domain dataclasses (ToolDefinition, ...)
│   │   ├── store.py           #   ToolRegistry ABC + InMemoryToolRegistry
│   │   ├── credential_store.py#   CredentialStore ABC + InMemoryCredentialStore
│   │   ├── circuit_breaker.py #   CircuitBreaker (closed/open/half-open)
│   │   └── service.py         #   gRPC servicer
│   └── guardrails/            # Guardrails Service (Chapter 6, grpc.aio)
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
| **Data Service (Chapter 5)** | |
| `services/data/models.py` | 5.1 (IndexConfig), 5.3 (Index), 5.4 (DocumentSection, ExtractedDocument), 5.7 (DocumentMetadata), 5.8 (Chunk), 5.15 (IngestJob), 5.17 (SearchResult) |
| `services/data/parsers.py` | 5.4 (DocumentParser ABC), 5.5 (format detection) |
| `services/data/chunking.py` | 5.9 (ChunkingStrategy ABC), 5.11 (FixedSizeChunking) |
| `services/data/embedding.py` | 5.12 (EmbeddingGenerator) |
| `services/data/store.py` | 5.16 (VectorStore write ops), 5.17 (VectorStore search), 5.21 (keyword_search) |
| `services/data/pgvector_store.py` | 5.19 (PgvectorStore search) |
| `services/data/schema.sql` | 5.18 (pgvector schema), 5.22 (full-text search column) |
| `services/data/search.py` | 5.20 (search orchestration), 5.23 (hybrid search + RRF) |
| `services/data/pipeline.py` | 5.5 (format routing), 5.13 (document ingestion) |
| `services/data/service.py` | 5.2 (index management), 5.14 (document management), 5.15 (async ingest) |
| `proto/data.proto` | 5.24 (gRPC contract) |
| `genai_platform/clients/data.py` | 5.25 (DataClient SDK wrapper) |
| **Tool Service (Chapter 6)** | |
| `proto/tools.proto` | 6.1 (ToolService contract), 6.2 (ToolDefinition), 6.3 (ToolBehavior/RateLimits/CostMetadata), 6.7 (DiscoverToolsRequest), 6.17 (ExecutionLimits) |
| `services/tools/models.py` | 6.2 (ToolDefinition), 6.3 (ToolBehavior, RateLimits, CostMetadata), 6.15 (ToolTask), 6.17 (ExecutionLimits) |
| `services/tools/store.py` | 6.4 (registration), 6.5 (discovery by namespace), 6.7 (capability search), 6.10 (version constraints) |
| `services/tools/credential_store.py` | 6.13 (credential ref), 6.14 (CredentialStore interface) |
| `services/tools/circuit_breaker.py` | 6.18 (CircuitBreaker: closed/open/half-open) |
| `services/tools/service.py` | 6.1 (gRPC servicer), 6.4 (register), 6.7 (discover), 6.12 (execute), 6.18 (circuit breaker) |
| `genai_platform/clients/tools.py` | 6.4 (register), 6.7 (discover), 6.12 (execute) |
| **Guardrails Service (Chapter 6)** | |
| `proto/guardrails.proto` | 6.19 (GuardrailsService contract) |
| `services/guardrails/models.py` | 6.19 (PolicyResult), 6.21 (GuardrailCheck), 6.23 (tiered handling) |
| `services/guardrails/store.py` | 6.21 (input config), 6.23 (tiered handlers), 6.25 (human approval gate) |
| `services/guardrails/service.py` | 6.19 (gRPC servicer), 6.20 (multi-point eval), 6.21 (input validation), 6.23 (output filtering) |
| `genai_platform/clients/guardrails.py` | 6.19 (policy check), 6.20 (validate input), 6.23 (filter output) |
| `examples/quickstart_tools.py` | 6.4, 6.7, 6.8, 6.12–6.14 (execute + seeded CredentialStore), 6.19, 6.20, 6.23 (end-to-end demo) |
| **Workflow Service (Chapter 8)** | |
| `genai_platform/workflow.py` | 8.2 (`@workflow` decorator) |
| `examples/quickstart_workflow.py` | 8.1 (sync workflow example) |
| `examples/quickstart_workflow_stream.py` | 8.5 (streaming workflow yielding tokens) |
| `examples/quickstart_workflow_async.py` | 8.7 (async deep-research workflow), 8.11 (progress + checkpointing) |
| `examples/quickstart_workflow_compose.py` | 8.15 (parent calling child), 8.16 (parallel calls) |
| `genai_platform/runtime/server.py` | 8.3 (`find_workflow` + `build_app`), 8.4 (sync handler), 8.6 (SSE/stream handler), 8.8 (async handler), 8.12 (uvicorn entrypoint with workers) |
| `genai_platform/clients/workflow.py` | 8.11 (job-progress + checkpoint helpers), 8.17 (`call_parallel` with ThreadPoolExecutor), 8.18 (`call()` response-mode routing), 8.19 (HTTP retry for workflow→workflow), 8.20 (`_poll_until_complete` + `_consume_stream`) |
| `genai_platform/grpc_retry.py` | 8.13 (gRPC `RetryInterceptor`) |
| `genai_platform/clients/base.py` | 8.14 (attaching retry interceptor to every SDK channel) |
| `services/workflow/schema.sql` | 8.9 (jobs table) |
| `services/gateway/http_handler.py` | 8.10 (`/jobs/{id}` polling endpoint, proxied to `WorkflowService.GetJobStatus`) |
| `proto/workflow.proto` | 8.21 (WorkflowService gRPC contract), 8.22 (registry messages — WorkflowSpec, ScalingConfig, ResourceConfig), 8.23 (deployment messages — WorkflowDeployment, deploy/rollback requests) |
| `genai_platform/cli/deploy.py` | 8.24 (generated Dockerfile in `generate_dockerfile`); also generates the K8s Deployment / HorizontalPodAutoscaler / Service manifests in `build/<workflow-name>/` |
| `genai_platform/cli/docker_runner.py` | local-Docker demo runner used by `genai-platform deploy` (production replaces this with the Workflow Service's K8s API client) |

### Key Design Principles

1. **Domain types at the core**: Business logic uses Python dataclasses, never Protocol Buffers.
2. **Proto at the boundary**: gRPC servicers convert between proto messages and domain types.
3. **Provider abstraction**: All LLM providers implement the same `ModelProvider` ABC with domain types.
4. **Storage abstraction**: `SessionStorage`, `VectorStore`, `ToolRegistry`, and `PolicyStore` ABCs with swappable backends (in-memory, PostgreSQL/pgvector).
5. **SDK hides gRPC**: Clients return dataclasses to callers. Proto is an internal detail.
6. **Dynamic extensibility**: Custom parsers and chunking strategies can be registered at runtime via the SDK (source code is uploaded over gRPC and loaded by the Data Service).
7. **Circuit breaker**: Tool execution protected by closed/open/half-open state machine (Listing 6.18).
8. **Credential isolation**: Tools reference credentials by name; secrets stored separately (Listing 6.13-6.14).

## Production deployment

The architecture splits cleanly into **shared platform services** the
platform team operates once per organization, and **per-app workflows**
that application teams deploy individually. Both halves are
container-first, so the same Docker images that run under
`docker compose up` locally are what an adopter pushes to a registry
and references from Kubernetes manifests.

**Platform services (operate once):** `docker/sessions.Dockerfile`,
`docker/models.Dockerfile`, `docker/data.Dockerfile`,
`docker/tools.Dockerfile`, `docker/guardrails.Dockerfile`,
`docker/gateway.Dockerfile`, `docker/workflow.Dockerfile`. Tag them for
your registry, push, and apply your own Deployment / Service manifests
(or the docker-compose stack as-is, for small deployments).

**Workflows (one per AI app):** `genai-platform deploy <file>` writes
Kubernetes manifests alongside the Docker artifacts:

```
build/<workflow-name>/
├── Dockerfile                       # built and (optionally) pushed
├── Deployment.yaml                  # populated from @workflow scaling fields
├── HorizontalPodAutoscaler.yaml     # populated from min_/max_replicas / target_cpu_percent
└── Service.yaml                     # exposes the workflow's port internally
```

Push the image to your registry, then `kubectl apply -f
build/<workflow-name>/`. The CLI never invokes `kubectl` itself — the
manifests are starter artifacts ready for any cluster.

Why per-service Deployments (not sidecars)? Because stateful shared
services lose their purpose when copied per-workflow. Sessions Service
exists to share conversation state across workflows; sidecar = each
workflow gets its own private session store. Same logic for Data Service
(vector index), Tools Service (registry), Models Service (rate-limit
pooling). The split between *shared platform services* and *per-workflow
containers* is deliberate.

## Running Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

## Status

- **Model Service** (Chapter 3): OpenAI, Anthropic, streaming, prompt management, custom models, client-side fallback
- **Session Service** (Chapter 4): Messages, pagination, model-managed memory, PostgreSQL
- **Data Service** (Chapter 5): Document ingestion, chunking, vector/hybrid search, pgvector, dynamic parser registration
- **Tool Service** (Chapter 6): Registration, discovery (namespace/capability/tags), versioning, HTTP execution with credential injection (api_key / bearer / oauth2 / basic) and per-tool timeout + response-size limits, async execution with task polling (Listing 6.16), MCP server registration over streamable HTTP with per-server policy overrides (Listing 6.18), circuit breaker, credential store
- **Guardrails Service** (Chapter 6): Input validation (prompt injection, PII), output filtering (PII redaction), policy enforcement, violation reporting
- **API Gateway**: gRPC proxy with service discovery (sessions, models, data, tools, guardrails, workflow); sync client to backends (tools/guardrails use grpc.aio servers, compatible at the wire level); HTTP forwarding to workflow containers (sync / SSE / 202+poll); `/jobs/{id}` proxy to Workflow Service; route table re-hydrates from `WorkflowService.ListRoutes` on startup
- **Workflow Service** (Chapter 8): `@workflow` decorator, FastAPI runtime server (sync / stream / async handlers), gRPC RetryInterceptor, workflow composition (`call`/`call_parallel`), `genai-platform deploy` CLI that builds Docker images and generates Kubernetes manifests
- **Local platform stack**: `docker compose up` brings up Postgres + all seven services on a shared network; per-service Dockerfiles in `docker/` are the same artifacts a platform team would push to a registry for K8s
- Observability & Experimentation (Chapter 7): planned -- traces, spans, structured logging, experimentation
- AI Assistant (Chapter 9): planned -- agent loop, memory, knowledge, tools, safety, observability
