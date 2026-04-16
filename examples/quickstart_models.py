"""
Quick start example for Model Service.

Demonstrates chat, streaming, model discovery, and **client-side fallback**.
Primary model: claude-sonnet-4-5 (Anthropic) with gpt-4o (OpenAI) as fallback.

Requires ANTHROPIC_API_KEY and/or OPENAI_API_KEY in .env file.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 3.12: FallbackConfig dataclass
  - Listing 3.18: ModelClient.chat() with fallback_config
  - Listing 3.20: Complete workflow example using Model Service
"""

import os
import sys
import threading
import time
from http.server import HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from genai_platform import GenAIPlatform
from services.gateway.registry import ServiceRegistry
from services.gateway.servers import create_grpc_server as create_gateway_grpc
from services.gateway.servers import create_http_server
from services.models.main import load_env_file
from services.models.models import FallbackConfig, RetryConfig
from services.models.service import ModelService
from services.shared.server import create_grpc_server, get_service_port


def _start_servers():
    """Start model service and gateway, returning server objects for cleanup."""
    HTTPServer.allow_reuse_address = True

    project_root = Path(__file__).resolve().parents[1]
    load_env_file(project_root / ".env")

    port = get_service_port("models")
    servicer = ModelService()
    model_server = create_grpc_server(servicer=servicer, port=port, service_name="models")
    model_server.start()

    registry = ServiceRegistry()
    models_addr = os.getenv("MODELS_SERVICE_ADDR", f"localhost:{port}")
    registry.register_platform_service("models", models_addr)

    grpc_port = int(os.getenv("GATEWAY_PORT", "50051"))
    gateway_server = create_gateway_grpc(registry, grpc_port)
    gateway_server.start()

    http_port = int(os.getenv("GATEWAY_HTTP_PORT", "8080"))
    http_server = create_http_server(registry, http_port)
    http_thread = threading.Thread(target=http_server.serve_forever, daemon=True)
    http_thread.start()

    return model_server, gateway_server, http_server


def _stop_servers(model_server, gateway_server, http_server):
    """Gracefully stop all servers."""
    http_server.shutdown()
    gateway_server.stop(grace=0)
    model_server.stop(grace=0)


def main():
    print("Starting services...")
    model_server, gateway_server, http_server = _start_servers()
    time.sleep(1)
    print("Services ready!\n")

    platform = GenAIPlatform()

    try:
        # --- Example 1: Model discovery ---
        available = platform.models.list_models()
        providers = {m.provider for m in available}

        print("=" * 50)
        print("Available models")
        print("=" * 50)
        for m in available:
            print(f"  {m.name} ({m.provider}) - context: {m.capabilities.context_window}")
        print()

        if not providers:
            print("No providers configured. Set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env")
            return

        # --- Example 2: Test both providers if both keys exist ---
        if "anthropic" in providers and "openai" in providers:
            print("=" * 50)
            print("Testing both providers (both API keys detected)")
            print("=" * 50)

            test_question = "Say 'Hello' in one word."

            # Test Anthropic
            print("\n[Anthropic]")
            resp = platform.models.chat(
                model="claude-sonnet-4-5",
                messages=[{"role": "user", "content": test_question}],
                max_tokens=10,
            )
            print(f"  Response: {resp.content}")
            print(f"  Model: {resp.model}")

            # Test OpenAI
            print("\n[OpenAI]")
            resp = platform.models.chat(
                model="gpt-4o",
                messages=[{"role": "user", "content": test_question}],
                max_tokens=10,
            )
            print(f"  Response: {resp.content}")
            print(f"  Model: {resp.model}")
            print("\n✓ Both providers working!\n")
        else:
            names = ", ".join(providers)
            print(f"Only {names} configured. Set both keys to test multi-provider support.\n")

        # --- Configure fallback: Anthropic primary, OpenAI fallback ---
        fallback = FallbackConfig(
            providers=["gpt-4o"],
            retry_config=RetryConfig(max_retries=1),
        )
        primary_model = "claude-sonnet-4-5"

        # --- Example 3: Chat with fallback ---
        print("=" * 50)
        print(f"Chat with fallback (primary={primary_model}, fallback=gpt-4o)")
        print("=" * 50)
        question = "Explain quantum computing in one sentence."
        print(f"Q: {question}")

        response = platform.models.chat(
            model=primary_model,
            messages=[{"role": "user", "content": question}],
            temperature=0.7,
            max_tokens=100,
            fallback_config=fallback,
        )

        print(f"A: {response.content}")
        print(f"Served by: {response.provider} / {response.model}")
        if response.usage:
            print(f"Tokens: {response.usage.total_tokens}")
        print()

        # --- Example 4: Streaming with fallback ---
        print("=" * 50)
        print(f"Streaming with fallback (primary={primary_model}, fallback=gpt-4o)")
        print("=" * 50)
        question = "Count from 1 to 5."
        print(f"Q: {question}")
        print("A: ", end="", flush=True)

        last_model = None
        for chunk in platform.models.chat_stream(
            model=primary_model,
            messages=[{"role": "user", "content": question}],
            max_tokens=50,
            fallback_config=fallback,
        ):
            print(chunk.token, end="", flush=True)
            last_model = chunk.model

        print(f"\nServed by: {last_model}")
        print()

        print("\n✓ Done!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        _stop_servers(model_server, gateway_server, http_server)


if __name__ == "__main__":
    main()
