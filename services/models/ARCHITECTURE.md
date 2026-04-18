# Model Service Architecture

## Overview

The Model Service provides a unified interface to AI models with two core principles:

1. **Provider adapters auto-discover built-in models** — zero configuration for commercial APIs
2. **Separate ABCs for separate capabilities** — LLM inference (`ModelProvider`) and embedding (`EmbeddingProvider`) are distinct abstractions in distinct folders

This aligns with Chapters 3 and 5 of the book.

## Core Principle

**Zero Config for Common Case:**
```bash
# Just set API keys - models and embeddings work automatically
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

**Flexible When Needed:**
```python
platform.models.register_model(name="my-llama", endpoint="http://localhost:8000/v1", adapter_type="openai")
```

## Architecture

### 1. Chat Provider Adapters (`providers/`)

**Purpose**: LLM inference — next-token prediction via `chat()` and `chat_stream()`.

Each adapter:
- Implements the `ModelProvider` interface
- Auto-discovers its supported models via `get_supported_models()`
- Translates between platform domain types and provider API

**Available Adapters:**
- `OpenAIProvider` — OpenAI Chat Completions API (gpt-4o, gpt-4o-mini)
- `AnthropicProvider` — Anthropic Messages API (claude-sonnet-4-5, claude-haiku-4-5, claude-opus-4-5)
- Future: `VLLMProvider` — vLLM/OpenAI-compatible format

**ABC (`providers/base.py`):**
```python
class ModelProvider(ABC):
    def chat(model, messages, config, ...) -> ChatResponse
    def chat_stream(model, messages, config, ...) -> Iterator[ChatChunk]
    def get_supported_models() -> List[ModelInfo]
```

### 2. Embedding Provider Adapters (`embedding_providers/`)

**Purpose**: Text-to-vector embedding via `embed()`. Separate from chat because it is a fundamentally different capability with different inputs, outputs, and provider landscape.

Each adapter:
- Implements the `EmbeddingProvider` interface
- Auto-discovers its supported embedding models via `get_supported_embedding_models()`

**Available Adapters:**
- `OpenAIEmbeddingProvider` — OpenAI Embeddings API (text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002)
- `HuggingFaceEmbeddingProvider` — sentence-transformers for local embedding (any HuggingFace model, e.g. all-MiniLM-L6-v2). Optional dependency.

**ABC (`embedding_providers/base.py`):**
```python
class EmbeddingProvider(ABC):
    def embed(texts: List[str], model: str) -> EmbeddingResponse
    def get_supported_embedding_models() -> List[ModelInfo]
```

**Why separate ABCs and folders:**
- Anthropic has no embedding API — a combined ABC would force a broken stub
- HuggingFace sentence-transformers has no chat API — same problem in reverse
- OpenAI has both capabilities but they're different API surfaces (`chat.completions.create` vs `embeddings.create`)
- Each folder grows independently as new providers are added

### 3. Domain Models (`models.py`)

Python dataclasses that form the clean boundary between gRPC protobuf and internal logic:

- `ChatMessage`, `ChatConfig`, `ChatResponse`, `ChatChunk` — chat types
- `EmbeddingResponse` — embedding result (list of vectors + metadata)
- `ModelInfo`, `ModelCapability` — model discovery
- `TokenUsage` — token accounting
- `FallbackConfig`, `RetryConfig`, `RoutingConfig` — operational configs

The servicer converts proto → domain at the boundary; all internal logic uses domain types.

### 4. Model Registry (`store.py`)

**Purpose**: Stores explicit model registrations (custom + overrides).

**Use Cases:**
- Register custom self-hosted models
- Override built-in model endpoints (e.g., Azure OpenAI proxy)

**NOT for:** Built-in commercial models (those are auto-discovered).

### 5. Prompt Registry (`store.py`)

**Purpose**: Version and manage system prompts. Standard registry pattern.

### 6. Model Service Servicer (`service.py`)

**Main orchestrator** with two parallel provider registries:

```python
class ModelService(ModelServiceServicer, BaseServicer):
    self._providers: Dict[str, ModelProvider]              # chat
    self._embedding_providers: Dict[str, EmbeddingProvider]  # embedding
```

Handles all gRPC RPCs: `Chat`, `ChatStream`, `Embed`, `ListModels`, `ListEmbeddingModels`, prompt management, and model registry operations.

### 7. SDK Client (`genai_platform/clients/models.py`)

`ModelClient` provides high-level Python methods that hide gRPC details:

```python
# Chat
response = platform.models.chat(model="gpt-4o", messages=[...])

# Streaming
for chunk in platform.models.chat_stream(model="gpt-4o", messages=[...]):
    print(chunk.token, end="")

# Embedding
resp = platform.models.embed(texts=["hello", "world"], model="text-embedding-3-small")

# Discovery
models = platform.models.list_models()
embedding_models = platform.models.list_embedding_models()
```

Supports client-side fallback chains via `FallbackConfig`.

### 8. gRPC Contract (`proto/models.proto`)

Defines all RPCs on the `ModelService`:

| RPC | Purpose |
|---|---|
| `Chat` | Synchronous chat completion |
| `ChatStream` | Streaming chat completion |
| `Embed` | Text embedding |
| `ListModels` | Discover available chat models |
| `ListEmbeddingModels` | Discover available embedding models |
| `RegisterPrompt` / `GetPrompt` / `ListPrompts` | Prompt management |
| `RegisterModel` / `ListRegisteredModels` / `GetModelStatus` | Custom model registry |

### 9. Gateway Integration

The Model Service is accessed through the API Gateway (`services/gateway/`). The gateway's `ModelServiceProxy` in `grpc_proxy.py` forwards all RPCs to the Model Service backend. Clients never connect to the Model Service directly — they connect to the gateway and use `x-target-service: models` metadata for routing.

## Resolution Flows

### Chat Resolution

When a `Chat` or `ChatStream` request comes in for model "X":

```
1. Check ModelRegistry (explicit registrations)
   - Custom models, overrides of built-in models
   
2. Check chat provider adapters (auto-discovered)
   - OpenAI models (if OPENAI_API_KEY set)
   - Anthropic models (if ANTHROPIC_API_KEY set)
   
3. Return NOT_FOUND if no match
```

### Embedding Resolution

When an `Embed` request comes in for model "X":

```
1. Check embedding provider adapters (auto-discovered)
   - OpenAI embedding models (if OPENAI_API_KEY set)
   - HuggingFace models (if HF_EMBEDDING_MODELS set)
   
2. Return NOT_FOUND if no match
```

## Visual Architecture

```
┌───────────────────────────────────────────────────────────┐
│                    Model Service                          │
├───────────────────────────────────────────────────────────┤
│                                                           │
│  Chat Providers (ModelProvider ABC):                      │
│  ┌────────────────────────┐                               │
│  │ • OpenAIProvider       │ → chat(), chat_stream()       │
│  │ • AnthropicProvider    │ → chat(), chat_stream()       │
│  │ • (Future: VLLM...)    │                               │
│  └────────────────────────┘                               │
│                                                           │
│  Embedding Providers (EmbeddingProvider ABC):              │
│  ┌────────────────────────┐                               │
│  │ • OpenAIEmbedding      │ → embed() (text-embedding-3)  │
│  │ • HuggingFaceEmbedding │ → embed() (sentence-transformers) │
│  └────────────────────────┘                               │
│                                                           │
│  ┌─────────────────┐  ┌─────────────────┐                 │
│  │  ModelRegistry  │  │ PromptRegistry  │                 │
│  │  (Custom models)│  │ (Versioning)    │                 │
│  └─────────────────┘  └─────────────────┘                 │
└───────────────────────────────────────────────────────────┘
          │                          │
          │  SDK (ModelClient)       │
          │  via API Gateway         │
          │                          │
          ├─→ api.openai.com         │  (chat + embeddings)
          ├─→ api.anthropic.com      │  (chat only)
          ├─→ HuggingFace local      │  (embeddings only)
          └─→ localhost:8000 (custom)│
```

## Folder Structure

```
services/models/
├── __init__.py
├── ARCHITECTURE.md
├── README.md
├── main.py                          # gRPC server entry point
├── models.py                        # Domain dataclasses
├── service.py                       # gRPC servicer (orchestrator)
├── store.py                         # ModelRegistry + PromptRegistry
├── providers/                       # LLM inference
│   ├── __init__.py
│   ├── base.py                      # ModelProvider ABC
│   ├── openai_provider.py
│   └── anthropic_provider.py
└── embedding_providers/             # Text-to-vector
    ├── __init__.py
    ├── base.py                      # EmbeddingProvider ABC
    ├── openai_provider.py
    └── huggingface_provider.py
```

## Configuration

### Environment Variables

**OpenAI (chat + embeddings):**
```bash
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1  # Optional override
```

**Anthropic (chat only):**
```bash
ANTHROPIC_API_KEY=sk-ant-...
```

**HuggingFace (embeddings only, optional dependency):**
```bash
HF_EMBEDDING_MODELS=all-MiniLM-L6-v2,all-mpnet-base-v2
# Requires: pip install sentence-transformers
```

**Auto-discovery:** If key/config is present, the corresponding provider initializes and reports its models.

### Using the SDK

```python
from genai_platform import GenAIPlatform

platform = GenAIPlatform()

# Chat
response = platform.models.chat(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)

# Chat with fallback
from services.models.models import FallbackConfig
response = platform.models.chat(
    model="claude-sonnet-4-5",
    messages=[{"role": "user", "content": "Hello!"}],
    fallback_config=FallbackConfig(providers=["gpt-4o"]),
)

# Embedding
resp = platform.models.embed(
    texts=["The quick brown fox", "jumped over the lazy dog"],
    model="text-embedding-3-small",
)
# resp.embeddings -> [[0.1, 0.2, ...], [0.3, 0.4, ...]]

# Register custom model
platform.models.register_model(
    name="my-llama-70b",
    endpoint="http://localhost:8000/v1",
    adapter_type="openai",
    provider="Internal Llama",
)
```

## Key Design Decisions

### 1. Separate ABCs for Chat and Embedding

**Why:** They are fundamentally different capabilities with different provider landscapes. A combined ABC forces broken stubs on providers that only support one capability.

### 2. Separate Folders (`providers/` vs `embedding_providers/`)

**Why:** Clean naming (both have `base.py`, `openai_provider.py`), independent growth, mirrors the two registries in `service.py`.

### 3. Auto-Discovery from Adapters

**Why:** Zero configuration for 99% of users. Provider adapters know which models they support.

### 4. Registry Overrides Auto-Discovered

**Why:** Power users can customize without breaking defaults. Resolution checks registry first.

### 5. Client-Side Fallback

**Why:** Resilience against provider outages. The SDK's `ModelClient` handles fallback chains transparently — if the primary model fails, it tries the next provider in the chain.

### 6. HuggingFace as Optional Dependency

**Why:** `sentence-transformers` pulls ~2GB of PyTorch. Making it optional keeps the base install lightweight while allowing local embedding when needed.

## Adding New Providers

### New Chat Provider

1. Create adapter in `providers/`:
```python
class GoogleProvider(ModelProvider):
    def get_supported_models(self):
        return [ModelInfo(name="gemini-pro", ...)]
    def chat(...) -> ChatResponse: ...
    def chat_stream(...) -> Iterator[ChatChunk]: ...
```

2. Register in `ModelService._initialize_providers()`:
```python
google_key = os.getenv("GOOGLE_API_KEY")
if google_key:
    self._providers["google"] = GoogleProvider(api_key=google_key)
```

### New Embedding Provider

1. Create adapter in `embedding_providers/`:
```python
class CohereEmbeddingProvider(EmbeddingProvider):
    def get_supported_embedding_models(self):
        return [ModelInfo(name="embed-english-v3.0", ...)]
    def embed(texts, model) -> EmbeddingResponse: ...
```

2. Register in `ModelService._initialize_embedding_providers()`:
```python
cohere_key = os.getenv("COHERE_API_KEY")
if cohere_key:
    self._embedding_providers["cohere"] = CohereEmbeddingProvider(api_key=cohere_key)
```

## Testing

```bash
# Run all Model Service tests
python -m pytest tests/test_model_*.py -v

# Quick integration test (auto-starts services)
python examples/quickstart_models.py
```

## Future Enhancements

1. **Custom Model Inference**: Route inference requests to registered custom endpoints.
2. **Health Checking**: Implement actual health checks for registered models.
3. **Response Caching**: Chapter 3 caching strategies via `CacheConfig`.
4. **vLLM Provider**: Dedicated adapter for vLLM-specific features.
