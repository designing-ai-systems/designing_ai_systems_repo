"""
Streaming workflow quickstart (Listing 8.5).

Same workflow shape as the sync demo, but the function is a generator
yielding chunks. The runtime server emits each chunk as a Server-Sent
Event (`data: {json}\\n\\n`); the gateway streams the chunks through to
the client without buffering.

Module-level imports are limited to the SDK so this file can be
``genai-platform deploy``'d into a container; harness-only imports live
inside ``main()``.

Book: "Designing AI Systems"
  - Listing 8.5 (streaming workflow)
  - Listing 8.6 (SSE handler)
"""

from genai_platform import workflow


@workflow(
    name="patient_intake_stream",
    api_path="/patient-stream",
    response_mode="stream",
    timeout_seconds=5,
)
def stream_patient_answer(question: str, patient_id: str):
    yield {"phase": "started", "patient_id": patient_id}
    for token in ("Thanks", " for", " your", " question.", " Goodbye!"):
        yield {"token": token}
    yield {"phase": "complete"}


def main() -> None:
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    import httpx

    from examples._workflow_demo_harness import gateway_http_url, local_platform

    with local_platform(stream_patient_answer):
        print(f"POST (streaming) {gateway_http_url()}/patient-stream")
        with httpx.stream(
            "POST",
            f"{gateway_http_url()}/patient-stream",
            json={"question": "What docs?", "patient_id": "p-1"},
            timeout=15,
        ) as response:
            for line in response.iter_lines():
                if line.startswith("data: "):
                    print(f"  chunk: {json.loads(line[len('data: ') :])}")


if __name__ == "__main__":
    main()
