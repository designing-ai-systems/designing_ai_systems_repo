"""
Sync workflow quickstart — patient intake assistant (Listing 8.1).

This file is dual-purpose:

- ``genai-platform deploy examples/quickstart_workflow.py`` packages it
  into a container and runs it inside the platform. The container's
  Python interpreter only executes module-level code, so module-level
  imports must be limited to the SDK (no ``examples/`` package).
- ``python examples/quickstart_workflow.py`` (run locally) boots the
  full platform in-process via the demo harness and exercises the
  workflow end-to-end. Harness-only imports therefore live inside
  ``main()`` rather than at module scope.

Book: "Designing AI Systems"
  - Listing 8.1 (the @workflow function)
  - Listing 8.4 (the sync handler that serves it)
"""

from genai_platform import workflow


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
    # Harness imports happen here, not at module scope, so the deployed
    # container can `import workflow` without needing `examples/`.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    import httpx

    from examples._workflow_demo_harness import gateway_http_url, local_platform

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
