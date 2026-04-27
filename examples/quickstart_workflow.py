"""
Sync workflow quickstart — patient intake assistant (Listing 8.1).

Runs the canonical sync workflow end-to-end against a locally-booted
platform: Workflow Service + gateway + runtime server. Once the gateway
is up, the script POSTs to the workflow's api_path and prints the JSON
response.

In commit 3, the local-bootstrap part is replaced by `genai-platform
deploy examples/quickstart_workflow.py`, which builds a Docker image,
runs it, and registers the route with the gateway automatically.

Book: "Designing AI Systems"
  - Listing 8.1 (the @workflow function)
  - Listing 8.4 (the sync handler that serves it)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from examples._workflow_demo_harness import gateway_http_url, local_platform  # noqa: E402
from genai_platform import workflow  # noqa: E402


@workflow(
    name="patient_intake_assistant",
    api_path="/patient-assistant",
    response_mode="sync",
    timeout_seconds=10,
)
def handle_patient_question(question: str, patient_id: str) -> dict:
    # In production this would orchestrate Sessions + Models + Data + Guardrails.
    # For the demo we keep the function pure so it works without LLM credentials.
    return {
        "patient_id": patient_id,
        "answer": (
            f"Thanks for your question, {patient_id}. Here's a placeholder answer to: {question!r}"
        ),
    }


def main() -> None:
    with local_platform(handle_patient_question):
        print(f"POST {gateway_http_url()}/patient-assistant")
        r = httpx.post(
            f"{gateway_http_url()}/patient-assistant",
            json={"question": "What documents do I need?", "patient_id": "p-123"},
            timeout=15,
        )
        print(f"  status: {r.status_code}")
        print(f"  body:   {r.json()}")


if __name__ == "__main__":
    main()
