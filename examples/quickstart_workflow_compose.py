"""
Workflow composition quickstart (Listings 8.15–8.17).

A "research assistant" parent calls two child workflows in parallel
through `platform.workflows.call_parallel(...)`. The parent never knows
or cares about the children's response_mode — `call()` auto-detects from
the response shape (200 JSON / 200 SSE / 202+poll) and returns a Python
value either way.

This demo runs the children on a small FastAPI server and points the
parent's composition HTTP client directly at them. In production each
child is its own deploy-CLI'd container; the parent's `call_parallel`
line is unchanged. The unit tests at `tests/test_workflow_compose.py`
exercise the gateway-routed path with MockTransport.

Book: "Designing AI Systems"
  - Listing 8.15 (parent calling a child)
  - Listing 8.16 (parallel calls)
  - Listing 8.17 (`call_parallel` with ThreadPoolExecutor)
  - Listing 8.18 (response-mode auto-detection)
"""

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402

from genai_platform import GenAIPlatform, workflow  # noqa: E402

CHILDREN_PORT = 8210


def _children_app() -> FastAPI:
    app = FastAPI()

    @app.post("/papers")
    def papers(payload: dict) -> dict:
        topic = payload.get("topic", "")
        return {"topic": topic, "papers": [f"paper-{i}-{topic}" for i in range(3)]}

    @app.post("/news")
    def news(payload: dict) -> dict:
        topic = payload.get("topic", "")
        return {"topic": topic, "news": [f"news-{i}-{topic}" for i in range(2)]}

    return app


@workflow(
    name="research_assistant",
    api_path="/research-assistant",
    response_mode="sync",
    timeout_seconds=15,
)
def research_assistant(topic: str) -> dict:
    """Parent: fan out to two children in parallel and aggregate results."""
    platform = GenAIPlatform(gateway_url="localhost:50151")
    # Point composition client straight at the children for the demo. The
    # real path goes parent → gateway HTTP → child container, exercised by
    # the unit tests in tests/test_workflow_compose.py.
    platform.workflows._http_client = httpx.Client(base_url=f"http://127.0.0.1:{CHILDREN_PORT}")

    papers, news = platform.workflows.call_parallel(
        [
            ("/papers", {"topic": topic}),
            ("/news", {"topic": topic}),
        ]
    )
    return {
        "topic": topic,
        "paper_count": len(papers["papers"]),
        "news_count": len(news["news"]),
        "papers": papers,
        "news": news,
    }


def main() -> None:
    config = uvicorn.Config(
        _children_app(), host="127.0.0.1", port=CHILDREN_PORT, log_level="error"
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(50):
        if server.started:
            break
        time.sleep(0.05)

    try:
        result = research_assistant(topic="vector dbs")
        print("Parent result:")
        print(f"  topic:        {result['topic']}")
        print(f"  paper_count:  {result['paper_count']}")
        print(f"  news_count:   {result['news_count']}")
        print(f"  papers:       {result['papers']['papers']}")
        print(f"  news:         {result['news']['news']}")
    finally:
        server.should_exit = True
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
