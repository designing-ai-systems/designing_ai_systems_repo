"""
Async workflow quickstart — deep researcher (Listings 8.7, 8.11).

The async workflow returns 202 immediately with a job_id; the script
polls `/jobs/{id}` through the gateway until the job reaches a terminal
state (`succeeded` / `failed` / `cancelled`).

The workflow itself uses `platform.workflows.update_job_progress(...)` and
`save_checkpoint(...)` to surface progress and persist mid-flight state.
The gateway proxies the polling endpoint to `WorkflowService.GetJobStatus`
over gRPC.

Book: "Designing AI Systems"
  - Listing 8.7 (async deep researcher)
  - Listing 8.8 (async handler that creates the job)
  - Listing 8.10 (`/jobs/{id}` polling endpoint)
  - Listing 8.11 (progress + checkpointing)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from examples._workflow_demo_harness import gateway_http_url, local_platform  # noqa: E402
from genai_platform import GenAIPlatform, workflow  # noqa: E402


@workflow(
    name="deep_researcher",
    api_path="/research",
    response_mode="async",
    timeout_seconds=60,
)
def deep_research(topic: str, depth: int = 3) -> dict:
    platform = GenAIPlatform(
        gateway_url=__import__("os").environ.get("GENAI_GATEWAY_URL", "localhost:50151")
    )
    platform.workflows.update_job_progress(message=f"Phase 1/3: gathering sources for {topic!r}")
    time.sleep(0.3)
    platform.workflows.save_checkpoint('{"phase": "sources_gathered"}')
    platform.workflows.update_job_progress(message="Phase 2/3: analyzing")
    time.sleep(0.3)
    platform.workflows.update_job_progress(message="Phase 3/3: drafting report")
    return {
        "topic": topic,
        "depth": depth,
        "summary": f"Found N sources about {topic}. Drafted a {depth}-page report.",
    }


def main() -> None:
    with local_platform(deep_research):
        print(f"POST {gateway_http_url()}/research")
        r = httpx.post(
            f"{gateway_http_url()}/research",
            json={"topic": "vector databases", "depth": 5},
            timeout=15,
        )
        print(f"  status: {r.status_code}")
        body = r.json()
        print(f"  body:   {body}")
        job_id = body["job_id"]

        deadline = time.time() + 30
        while time.time() < deadline:
            poll = httpx.get(f"{gateway_http_url()}/jobs/{job_id}", timeout=5).json()
            status = poll["job"]["status"]
            progress = poll["job"].get("progress_message", "")
            print(f"  poll → status={status} progress={progress!r}")
            if status in ("succeeded", "failed", "cancelled"):
                if status == "succeeded":
                    print(f"  result: {poll['job']['result_json']}")
                else:
                    print(f"  error:  {poll['job']['error']}")
                return
            time.sleep(0.3)
        raise SystemExit("timed out waiting for async workflow")


if __name__ == "__main__":
    main()
