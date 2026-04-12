"""
Quick start example for Model Service.

Demonstrates basic chat, streaming, and model discovery.
Requires OPENAI_API_KEY and/or ANTHROPIC_API_KEY in .env file.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 3.20: Complete workflow example using Model Service
"""

import sys
import time
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from genai_platform import GenAIPlatform
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

    # --- Example 1: Simple chat ---
    print("=" * 50)
    print("Chat - OpenAI (gpt-4o)")
    print("=" * 50)
    question = "Explain quantum computing in one sentence."
    print(f"Q: {question}")

    response = platform.models.chat(
        model="gpt-4o",
        messages=[{"role": "user", "content": question}],
        temperature=0.7,
        max_tokens=100,
    )

    print(f"A: {response.content}")
    if response.usage:
        print(f"Tokens: {response.usage.total_tokens}")
    print()

    # --- Example 2: Streaming ---
    print("=" * 50)
    print("Streaming - OpenAI (gpt-4o)")
    print("=" * 50)
    question = "Count from 1 to 5."
    print(f"Q: {question}")
    print("A: ", end="", flush=True)

    for chunk in platform.models.chat_stream(
        model="gpt-4o",
        messages=[{"role": "user", "content": question}],
        max_tokens=50,
    ):
        print(chunk.token, end="", flush=True)

    print("\n")

    # --- Example 3: Model discovery ---
    print("=" * 50)
    print("Available models")
    print("=" * 50)
    models = platform.models.list_models()
    for m in models:
        print(f"  {m.name} ({m.provider}) - context: {m.capabilities.context_window}")

    print("\n\nDone! Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        sys.exit(0)


if __name__ == "__main__":
    main()
