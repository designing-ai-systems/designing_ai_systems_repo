# Model Service

The Model Service provides unified access to AI model providers (OpenAI, Anthropic, etc.) and custom self-hosted models.

## Architecture

See [ARCHITECTURE.md](./ARCHITECTURE.md) for detailed design documentation.

**Key Components**:
- **ProviderRegistry**: Manages commercial providers (OpenAI, Anthropic)
- **ModelRegistry**: Stores custom/self-hosted models only
- **PromptRegistry**: System prompt management with versioning
- **Provider Adapters**: Translate between platform and provider APIs

## Quick Start

### 1. Install Dependencies
```bash
uv sync
```

### 2. Generate Protocol Buffers
```bash
uv run python proto/generate.py
```

### 3. Configure API Keys

Create `.env` file in project root:
```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

Or export them:
```bash
export OPENAI_API_KEY=your_key
export ANTHROPIC_API_KEY=your_key
```

### 4. Run Complete Test

The easiest way to test everything:
```bash
uv run python examples/quickstart_models.py
```

This automatically starts all services (Session, Model, Gateway) and runs basic tests.

## Manual Service Startup

If you want to run services separately:

**Terminal 1 - Session Service**:
```bash
uv run python -m services.sessions.main
```

**Terminal 2 - Model Service**:
```bash
uv run python -m services.models.main
```

**Terminal 3 - API Gateway**:
```bash
uv run python -m services.gateway.main
```

**Terminal 4 - Run tests**:
```bash
uv run python examples/test_model_service.py
```

## Features

### Core Inference
- **Chat**: Synchronous chat completion
- **ChatStream**: Streaming chat with progressive tokens

### Discovery
- **ListModels**: Get all available models (commercial + custom)
- **GetModelCapabilities**: Query specific model capabilities

### Prompt Management
- **RegisterPrompt**: Store prompts with versioning
- **GetPrompt**: Retrieve by name/version
- **ListPrompts**: List all registered prompts

### Custom Models
- **RegisterModel**: Register self-hosted models
- **ListRegisteredModels**: View custom models
- **GetModelStatus**: Check model health

## Supported Providers

### Commercial (Built-in)
- **OpenAI**: gpt-4o, gpt-4o-mini
- **Anthropic**: claude-3-5-sonnet, claude-3-5-haiku

Providers are automatically configured when API keys are present.

### Custom Models
Register self-hosted models via `register_model`:
```python
platform.models.register_model(
    name="my-llama-70b",
    endpoint="http://localhost:8000/v1",
    adapter_type="vllm",  # Which adapter: "openai", "anthropic", "vllm"
    context_window=8192,
    supports_vision=False,
    supports_tools=True,
    provider="Internal Llama"  # Optional display name
)
```

## Examples

### Basic Chat
```python
from genai_platform import GenAIPlatform

platform = GenAIPlatform()

response = platform.models.chat(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.7,
    max_tokens=150
)
print(response['text'])
```

### Streaming
```python
for token in platform.models.chat_stream(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Count to 5"}],
    temperature=0.7
):
    print(token, end="", flush=True)
```

See `examples/quickstart_models.py` and `examples/test_model_service.py` for more.

## Port

Default: `50053` (override with `MODELS_PORT` env var)
