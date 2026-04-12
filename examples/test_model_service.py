"""
Complete Model Service test - demonstrates all operations.

Tests:
- ListModels (discovery)
- GetModelCapabilities
- Chat with OpenAI
- Chat with Anthropic
- ChatStream (streaming)
- Prompt management (register, get, list)
- Custom model registration

Automatically starts all required services.
"""

import sys
import threading
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genai_platform import GenAIPlatform
from services.gateway.main import main as start_gateway
from services.models.main import main as start_model_service


def start_service_in_thread(service_func, service_name):
    """Start a service in a background thread."""

    def run_service():
        try:
            service_func()
        except KeyboardInterrupt:
            pass

    thread = threading.Thread(target=run_service, daemon=True, name=service_name)
    thread.start()
    return thread


def test_discovery(platform: GenAIPlatform):
    """Test model discovery operations."""
    print("\n" + "=" * 60)
    print("TEST: Model Discovery")
    print("=" * 60)

    # List all available models
    models = platform.models.list_models()
    print(f"\nAvailable models: {len(models)}")
    for model in models:
        print(f"  - {model.name} ({model.provider})")
        print(f"    Context: {model.capabilities.context_window}")
        print(f"    Vision: {model.capabilities.supports_vision}")
        print(f"    Tools: {model.capabilities.supports_tools}")

    # Get capabilities for specific model
    if models:
        model_name = models[0].name
        caps = platform.models.get_model_capabilities(model_name)
        print(f"\nCapabilities for {model_name}:")
        print(f"  Context window: {caps.context_window}")
        print(f"  Vision: {caps.supports_vision}")
        print(f"  Tools: {caps.supports_tools}")


def test_chat(platform: GenAIPlatform, model: str, prompt: str):
    """Test basic chat completion."""
    print(f"\n[Chat] Model: {model}")
    print(f"Prompt: {prompt}")

    response = platform.models.chat(
        model=model, messages=[{"role": "user", "content": prompt}], temperature=0.7, max_tokens=150
    )

    print(f"Response: {response.content}")
    if response.usage:
        print(
            f"Tokens: {response.usage.total_tokens} "
            f"(prompt: {response.usage.prompt_tokens}, "
            f"completion: {response.usage.completion_tokens})"
        )
    print(f"Finish reason: {response.finish_reason}")


def test_streaming(platform: GenAIPlatform, model: str):
    """Test streaming chat completion."""
    print(f"\n[Streaming] Model: {model}")
    print("Streaming response: ", end="", flush=True)

    for chunk in platform.models.chat_stream(
        model=model,
        messages=[{"role": "user", "content": "Count from 1 to 5, one number per line."}],
        temperature=0.7,
        max_tokens=50,
    ):
        print(chunk.token, end="", flush=True)

    print()


def test_prompts(platform: GenAIPlatform):
    """Test prompt management."""
    print("\n" + "=" * 60)
    print("TEST: Prompt Management")
    print("=" * 60)

    # Register a prompt
    result = platform.models.register_prompt(
        name="patient-intake",
        content="You are a helpful patient intake assistant for a medical clinic.",
        author="test@example.com",
        tags=["production", "medical"],
    )
    print(f"\nRegistered prompt: {result['name']} v{result['version']}")

    # Get the prompt
    prompt = platform.models.get_prompt("patient-intake")
    print(f"Retrieved: {prompt['name']} v{prompt['version']}")
    print(f"Content: {prompt['content'][:50]}...")

    # List all prompts
    prompts = platform.models.list_prompts()
    print(f"\nAll prompts: {len(prompts)}")
    for p in prompts:
        print(f"  - {p['name']} v{p['version']}")


def test_custom_models(platform: GenAIPlatform):
    """Test custom model registration."""
    print("\n" + "=" * 60)
    print("TEST: Custom Model Registration")
    print("=" * 60)

    # Register a custom model
    result = platform.models.register_model(
        name="my-llama-70b",
        endpoint="http://localhost:8000/v1",
        adapter_type="openai-compatible",
        context_window=8192,
        supports_vision=False,
        supports_tools=True,
        health_check="/health",
    )
    print(f"\nRegistered: {result['name']}")
    print(f"Status: {result['status']}")

    # List registered models
    models = platform.models.list_registered_models()
    print(f"\nCustom models: {len(models)}")
    for model in models:
        print(f"  - {model['name']} ({model['adapter_type']})")
        print(f"    Endpoint: {model['endpoint']}")

    # Get model status
    status = platform.models.get_model_status("my-llama-70b")
    print(f"\nStatus for {status['name']}:")
    print(f"  Status: {status['status']}")
    print(f"  Endpoint: {status['endpoint']}")


def main():
    print("=" * 60)
    print("Model Service Comprehensive Test")
    print("=" * 60)
    print("\nStarting services...")

    # Start services
    start_service_in_thread(start_model_service, "ModelService")
    time.sleep(2)
    start_service_in_thread(start_gateway, "Gateway")
    time.sleep(2)

    print("Services started!\n")

    # Initialize platform
    platform = GenAIPlatform()

    try:
        # Discovery tests
        test_discovery(platform)

        # Chat tests
        print("\n" + "=" * 60)
        print("TEST: Chat Completions")
        print("=" * 60)
        test_chat(platform, "gpt-4o", "What is the capital of France?")
        test_chat(platform, "claude-sonnet-4-5", "What is 2+2?")

        # Streaming test
        print("\n" + "=" * 60)
        print("TEST: Streaming")
        print("=" * 60)
        test_streaming(platform, "gpt-4o")

        # Prompt management tests
        test_prompts(platform)

        # Custom model tests
        test_custom_models(platform)

        print("\n" + "=" * 60)
        print("✓ All tests completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()

    print("\nPress Ctrl+C to stop services...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        sys.exit(0)


if __name__ == "__main__":
    main()
