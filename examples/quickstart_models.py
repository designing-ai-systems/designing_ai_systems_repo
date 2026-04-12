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

import sys
import time
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from genai_platform import GenAIPlatform
from services.models.models import FallbackConfig, RetryConfig
from services.models.main import main as start_model_service
from services.gateway.main import main as start_gateway


def start_service_in_thread(service_func, service_name):
    def run_service():
        try:
            service_func()
        except KeyboardInterrupt:
            pass
    thread = threading.Thread(target=run_service, daemon=True, name=service_name)
    thread.start()
    return thread


def main():
    print("Starting services...")
    start_service_in_thread(start_model_service, "ModelService")
    time.sleep(1)
    start_service_in_thread(start_gateway, "Gateway")
    time.sleep(1)
    print("Services ready!\n")

    platform = GenAIPlatform()

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
        sys.exit(1)

    # --- Configure fallback: Anthropic primary, OpenAI fallback ---
    fallback = FallbackConfig(
        providers=["gpt-4o"],
        retry_config=RetryConfig(max_retries=1),
    )
    primary_model = "claude-sonnet-4-5"

    # --- Example 2: Chat with fallback ---
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

    # --- Example 3: Streaming with fallback ---
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

    print("\nDone! Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        sys.exit(0)


if __name__ == "__main__":
    main()
