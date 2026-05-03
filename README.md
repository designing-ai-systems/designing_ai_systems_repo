# Designing AI systems repository

Production-ready platform for building GenAI applications with multi-provider support. Accompanies the book [**Designing AI Systems**](https://www.manning.com/books/designing-ai-systems) (Manning).

## Features

- **Multi-provider inference**: OpenAI, Anthropic (with streaming)
- **Session management**: Conversation history and model-managed memory
- **Data & RAG pipeline**: Document ingestion, chunking, embedding, vector search, hybrid search
- **Tool management**: Registration, discovery, versioning, sandboxed execution
- **Guardrails**: Input validation, output filtering, policy enforcement
- **Observability**: Distributed traces, generations, structured logs, metrics, scores, cost & budget tracking
- **Experimentation**: Versioned prompt/model/config targets, offline evaluation, online scoring rules, A/B tests, annotation queues
- **Domain dataclasses**: Clean Python API -- never exposes Protocol Buffers
- **Model discovery**: Query capabilities, register custom models
- **Prompt registry**: Centralized system prompt management
- **Storage abstraction**: In-memory (dev) or PostgreSQL + pgvector (production)
- **Service architecture**: gRPC microservices with unified API Gateway

## Requirements

- **[`uv`](https://docs.astral.sh/uv/)** — the only prerequisite. It manages the Python interpreter, the virtual environment, and all dependencies. Install with:
  - macOS: `brew install uv`
  - Linux / macOS: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - Windows: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`

## Setup

```bash
# 1. Install dependencies (uv fetches Python 3.12, creates .venv, syncs from uv.lock)
uv sync

# 2. Configure API keys
cat > .env <<'EOF'
OPENAI_API_KEY=your-key
ANTHROPIC_API_KEY=your-key
EOF

# 3. Generate Protocol Buffer code (only needed after editing .proto files)
uv run python -m proto.generate

# 4. Run tests
uv run pytest tests/ -v

# 5. Lint (runs in CI on every PR)
uv run ruff check .
uv run ruff format --check .
```

That's it. `uv run <cmd>` executes inside the project's virtual environment without needing to activate it. If you prefer activating:

```bash
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows PowerShell
```

> **Prompt tip:** if your shell prompt shows double parentheses like
> `((.venv) )` after activating, add `export VIRTUAL_ENV_DISABLE_PROMPT=1`
> to your shell rc file (`~/.zshrc` / `~/.bashrc`). Works around a known
> bug in some Python `venv` activate templates.

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
uv sync --extra postgres

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
uv run python examples/quickstart_models.py
```

**Session + Model Integration (Chapters 3-4):**
```bash
uv run python examples/quickstart_conversation.py
```

**Tools & Guardrails (Chapter 6):**
```bash
uv run python examples/quickstart_tools.py            # full platform end-to-end (+ model loop if OPENAI_API_KEY set)
uv run python examples/test_tool_service.py           # tool service: register / discover / HTTP exec + credential injection / async / circuit breaker
uv run python examples/test_tool_service.py --mcp     # same, plus a live MCP call to https://mcp.deepwiki.com/mcp
uv run python examples/test_guardrails_service.py     # guardrails: input validation, output filtering, policy check, violation reporting
uv run python examples/quickstart_mcp.py              # platform registers DeepWiki (public MCP server) and runs real MCP calls
```

**Observability & Experimentation (Chapter 7):**
```bash
uv run python examples/quickstart_observability.py    # custom trace_operation + cost drill-down (Listings 7.10, 7.13)
uv run python examples/test_observability_service.py  # spans/generations, scores, logs, metrics, percentiles, budgets, health
uv run python examples/quickstart_experiments.py      # register a target, run offline eval, A/B test on production traffic
uv run python examples/test_experiments_service.py    # full improvement loop: targets, datasets, scoring rules, experiments, annotation
```

### Local development with Docker (recommended)

`docker compose up` brings up the nine platform services + a
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
uv run python -m services.sessions.main       # Terminal 1
uv run python -m services.models.main         # Terminal 2
uv run python -m services.data.main           # Terminal 3
uv run python -m services.tools.main          # Terminal 4
uv run python -m services.guardrails.main     # Terminal 5
uv run python -m services.workflow.main       # Terminal 6
uv run python -m services.observability.main  # Terminal 7
uv run python -m services.experiments.main    # Terminal 8 (set OBSERVABILITY_SERVICE_ADDR=localhost:50059)
uv run python -m services.gateway.main        # Terminal 9
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

### Observability Service (Chapter 7)

```python
from datetime import datetime, timedelta, timezone

from genai_platform import GenAIPlatform

platform = GenAIPlatform()

# Custom span around an operation that wraps platform calls (Listing 7.10).
# Child platform calls inherit the trace by reusing `ctx`.
with platform.observability.trace_operation(
    "rerank_pipeline",
    workflow_id="patient-intake",
    user_id="user-123",
) as ctx:
    # ... your custom work here ...
    pass

# Read a full trace back as a domain dataclass (Listing 7.5).
# The SDK flushes the in-process buffer before reading.
trace = platform.observability.get_trace(ctx.trace_id)
for span in trace.spans:
    print(f"  {span.service}.{span.operation} [{span.status}] {span.duration_ms:.1f}ms")
for gen in trace.generations:
    print(
        f"  {gen.model}  in={gen.prompt_tokens} "
        f"out={gen.completion_tokens} ${gen.cost_usd:.4f}"
    )

# Score a generation after the fact -- automated scorer, LLM-as-judge, or human (Listing 7.11)
platform.observability.record_score(
    trace_id=ctx.trace_id,
    name="helpfulness",
    value=0.85,
    source="MODEL_JUDGE",
    generation_id=trace.generations[0].span.span_id if trace.generations else "",
)

# Cost drill-down by team -> by model (Listing 7.13)
now = datetime.now(timezone.utc)
report = platform.observability.get_cost_report(
    start_time=now - timedelta(days=30),
    end_time=now,
    group_by=["team"],
)
print(f"  total ${report.total_cost_usd:.2f}")
for bucket in report.buckets:
    print(f"   - team={bucket.dimensions.get('team', '')}: ${bucket.cost_usd:.2f}")

# Budget alerts -- thresholds fire as the spend crosses each level (Listing 7.13)
platform.observability.set_budget_alert(
    name="engineering-monthly",
    scope_type="team",
    scope_value="engineering",
    limit_usd=10000.0,
    period="monthly",
    thresholds=[0.5, 0.8, 1.0],
    notification_channels=["slack:#platform-alerts"],
)
status = platform.observability.get_budget_status("engineering-monthly")
print(f"  spent ${status.current_spend_usd:.2f} of ${status.alert.limit_usd:.2f}")
```

### Experimentation Service (Chapter 7)

```python
from genai_platform import GenAIPlatform

platform = GenAIPlatform()

# Register a versioned target -- prompt, model config, or retrieval config (Listing 7.16)
target = platform.experiments.register_target(
    name="patient-intake-v2",
    version=3,
    target_type="PROMPT",
    author="ada",
    change_description="add insurance confirmation step",
    metadata={"ticket": "PROD-1247"},
)

# Curated evaluation dataset (Listing 7.17)
platform.experiments.create_dataset(
    name="intake-bench",
    test_cases=[
        {
            "id": "tc-001",
            "input_query": "What documents do I need for my appointment?",
            "ideal_response": "Bring photo ID and your insurance card.",
            "key_elements": ["photo ID", "insurance card"],
            "tags": ["intake", "documents"],
        },
    ],
    metadata={"owner": "team-a"},
)

# Offline eval -- streamed: each yield is a CreateEvaluationProgress message (Listing 7.16)
final = None
for progress in platform.experiments.run_evaluation(
    dataset_name="intake-bench",
    targets=[f"{target.name}:{target.version}"],
    metrics=["key_elements"],
    repeats_per_case=1,
):
    final = progress
if final and final.results.target_results:
    tr = final.results.target_results[0]
    print(f"  {tr.target_id}  overall_score={tr.overall_score:.2f}")

# Online scoring rule -- continuous quality monitor on production traffic (Listing 7.18)
platform.experiments.create_scoring_rule(
    name="intake-quality-monitor",
    workflow_id="patient-intake",
    sample_rate=0.1,
    scorers=[{"name": "key_elements", "type": "key_elements", "required_elements": ["insurance"]}],
    alert_on={"key_elements": {"below": 0.8, "window": "1h"}},
)

# A/B test in production (Listing 7.20)
platform.experiments.create_experiment(
    name="intake-prompt-ab",
    workflow_id="patient-intake",
    variants=[
        {
            "name": "control",
            "traffic_allocation": 0.5,
            "prompt_variant": {"prompt_name": "patient-intake-v2", "version": 2},
        },
        {
            "name": "treatment",
            "traffic_allocation": 0.5,
            "prompt_variant": {"prompt_name": "patient-intake-v2", "version": 3},
        },
    ],
    success_metrics=["resolved"],
    minimum_sample_size=100,
)
assignment = platform.experiments.assign_variant(
    experiment_name="intake-prompt-ab",
    assignment_key="user-123",
)
# ... run the workflow with the assigned variant, then record the outcome ...
platform.experiments.record_outcome(
    experiment_name="intake-prompt-ab",
    assignment_id=assignment.assignment_id,
    outcomes={"resolved": 1.0},
)

results = platform.experiments.get_experiment_results("intake-prompt-ab")
for c in results.comparisons:
    print(
        f"  {c.metric_name}: winner={c.winner} "
        f"effect={c.effect_size:.3f} p={c.p_value:.3f}"
    )
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
│   ├── observability/         # Observability Service (Chapter 7, grpc.aio)
│   │   ├── models.py          #   Domain dataclasses (Span, Generation, Score, ...)
│   │   ├── metrics.py         #   PlatformMetrics constants (Listing 7.4)
│   │   ├── store.py           #   ObservabilityStore ABC + InMemoryObservabilityStore
│   │   └── service.py         #   gRPC servicer (14 RPCs, Listing 7.1)
│   ├── experiments/           # Experimentation Service (Chapter 7, grpc.aio)
│   │   ├── models.py          #   Domain dataclasses (ExperimentTarget, Dataset, Variant, ...)
│   │   ├── scorers.py         #   Scorer ABC + key-element / LLM-judge / retrieval (Listing 7.12)
│   │   ├── evaluation.py      #   Offline evaluation pipeline
│   │   ├── ab_testing.py      #   Consistent-hash assignment + Welch's t-test
│   │   ├── store.py           #   ExperimentStore ABC + InMemoryExperimentStore
│   │   └── service.py         #   gRPC servicer (20 RPCs, Listing 7.14)
│   └── shared/
│       ├── traced_service.py  #   TracedService base + trace_operation/trace_generation (Listing 7.6)
│       └── observability_client.py  # Buffered async client (Listing 7.9)
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
| **Observability & Experimentation (Chapter 7)** | |
| `proto/observability.proto` | 7.1 (ObservabilityService contract), 7.2 (LogEvent / IngestLogs / QueryLogs), 7.5 (Span / Generation / Trace), 7.11 (Score) |
| `proto/experiments.proto` | 7.14 (ExperimentationService contract), 7.15 (ExperimentTarget / Dataset / Experiment / VariantAssignment / AnnotationQueue) |
| `services/shared/traced_service.py` | 7.6 (TracedService base, `trace_operation` and `trace_generation` context managers) |
| `services/shared/observability_client.py` | 7.9 (buffered async telemetry client with periodic flush + push-back on RPC failure) |
| `services/observability/models.py` | 7.5 (Span / Generation / Trace), 7.11 (Score), 7.13 (CostReport / BudgetAlert / BudgetStatus) |
| `services/observability/metrics.py` | 7.4 (`PlatformMetrics` standardized counter / histogram names) |
| `services/observability/store.py` | 7.1 storage primitives -- p50/p95/p99 percentiles, cost aggregations, budget projections, service-health window |
| `services/observability/service.py` | 7.1 gRPC servicer (Logs, Metrics, Spans, Generations, Traces, Scores, Cost, Budgets, Health) |
| `services/experiments/scorers.py` | 7.12 (`Scorer` ABC + KeyElement / LLMJudge / RetrievalRelevance scorers) |
| `services/experiments/evaluation.py` | 7.16 (offline evaluation pipeline: target × dataset × scorers → EvaluationSummary) |
| `services/experiments/ab_testing.py` | 7.20 (consistent-hash variant assignment + Welch's t-test on outcomes) |
| `services/experiments/store.py` | 7.14 storage primitives -- targets, datasets, scoring rules, experiments, assignments, outcomes, annotation queues |
| `services/experiments/service.py` | 7.14 gRPC servicer (target lifecycle, datasets, offline + online eval, A/B tests, annotation) |
| `services/models/service.py` | 7.3 (`_log_fallback_triggered`), 7.7 (Chat wraps provider call in `trace_generation`), 7.8 (per-request metrics) |
| `services/models/metrics_publisher.py` | 7.8 (`ModelServiceMetricsPublisher` — emits `model.requests`, `model.latency`, `model.tokens`, `model.cost`) |
| `genai_platform/clients/observability.py` | 7.10 (`platform.observability.trace_operation`), 7.13 (cost reports + budget alerts), full query/ingest surface |
| `genai_platform/clients/experiments.py` | 7.16, 7.17, 7.18, 7.19, 7.20 (full experimentation SDK: targets, datasets, evaluations, scoring rules, A/B tests, annotation) |
| `examples/quickstart_observability.py` | 7.10 (custom span around a workflow op), 7.13 (cost drill-down) |
| `examples/test_observability_service.py` | 7.5, 7.6, 7.11, 7.2, 7.4, 7.13 (end-to-end traces, scores, logs, metrics, cost, budgets) |
| `examples/quickstart_experiments.py` | 7.16, 7.17, 7.18, 7.20 (target → dataset → offline eval → scoring rule → A/B test) |
| `examples/test_experiments_service.py` | 7.16-7.20 (full improvement loop: targets, datasets, scoring rules, experiments, annotation queues) |
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
`docker/gateway.Dockerfile`, `docker/workflow.Dockerfile`,
`docker/observability.Dockerfile`, `docker/experiments.Dockerfile`. Tag
them for your registry, push, and apply your own Deployment / Service
manifests (or the docker-compose stack as-is, for small deployments).

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
uv run pytest tests/ -v
```

## Status

- **Model Service** (Chapter 3): OpenAI, Anthropic, streaming, prompt management, custom models, client-side fallback
- **Session Service** (Chapter 4): Messages, pagination, model-managed memory, PostgreSQL
- **Data Service** (Chapter 5): Document ingestion, chunking, vector/hybrid search, pgvector, dynamic parser registration
- **Tool Service** (Chapter 6): Registration, discovery (namespace/capability/tags), versioning, HTTP execution with credential injection (api_key / bearer / oauth2 / basic) and per-tool timeout + response-size limits, async execution with task polling (Listing 6.16), MCP server registration over streamable HTTP with per-server policy overrides (Listing 6.18), circuit breaker, credential store
- **Guardrails Service** (Chapter 6): Input validation (prompt injection, PII), output filtering (PII redaction), policy enforcement, violation reporting
- **API Gateway**: gRPC proxy with service discovery (sessions, models, data, tools, guardrails, workflow, observability, experiments); sync client to backends (tools/guardrails/observability/experiments use grpc.aio servers, compatible at the wire level); HTTP forwarding to workflow containers (sync / SSE / 202+poll); `/jobs/{id}` proxy to Workflow Service; route table re-hydrates from `WorkflowService.ListRoutes` on startup
- **Workflow Service** (Chapter 8): `@workflow` decorator, FastAPI runtime server (sync / stream / async handlers), gRPC RetryInterceptor, workflow composition (`call`/`call_parallel`), `genai-platform deploy` CLI that builds Docker images and generates Kubernetes manifests
- **Observability Service** (Chapter 7): structured logs, metrics with p50/p95/p99 percentiles, distributed traces, generations (LLM-specific spans with token counts and cost), scores (numeric / categorical / boolean), cost attribution & drill-down, budget alerts with projections, service-health windows; in-process buffered client (`services/shared/observability_client.py`) keeps the request path off the wire
- **Experimentation Service** (Chapter 7): versioned targets (prompt / model / config / composite) with full lifecycle, curated + production-derived datasets, offline evaluation pipeline with key-element / LLM-judge / retrieval scorers, online sampling rules, A/B experiments with consistent-hash assignment and Welch's t-test, human annotation queues
- **Local platform stack**: `docker compose up` brings up Postgres + all nine services on a shared network; per-service Dockerfiles in `docker/` are the same artifacts a platform team would push to a registry for K8s
- AI Assistant (Chapter 9): planned -- agent loop, memory, knowledge, tools, safety, observability
