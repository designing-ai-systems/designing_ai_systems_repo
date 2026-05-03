"""
SDK runtime server (data plane).

Turns a @workflow-decorated function into a FastAPI HTTP service. The
container's CMD is ``python -m genai_platform.runtime.server <module>``;
``main()`` reads ``WORKFLOW_NAME`` from the environment, imports the
module, finds the matching decorated function, builds an app for it, and
hands it to uvicorn.

Three handler shapes, picked by the decorator's ``response_mode``:

- **sync** (Listing 8.4): runs the function in a worker thread with a
  per-request timeout, returns the function's return value as JSON.
- **stream** (Listing 8.6): function yields chunks; we emit each chunk as
  a Server-Sent Event (``data: {json}\\n\\n``) over a chunked HTTP body.
- **async** (Listing 8.8): creates a job in the Workflow Service via
  ``platform.workflows.create_job``, fires the function in a background
  task, returns ``202 Accepted`` with the ``job_id`` immediately. The
  background task calls ``complete_job`` / ``fail_job`` when done.

Health probes (``/health/live``, ``/health/ready``) are intentionally
shallow — they only check that the process is up. Deeper checks would
couple every workflow's health to its dependencies and cause cascading
failures (Chapter 8, "Health endpoints").
"""

import asyncio
import importlib
import json
import logging
import os
import sys
from contextvars import ContextVar
from typing import Any, Callable, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger(__name__)

# Listing 8.11. Set per-request by the runtime so workflow code can call
# `platform.workflows.update_job_progress(message)` without plumbing the
# job id manually.
_trace_var: ContextVar[Optional[str]] = ContextVar("workflow_trace_id", default=None)


def find_workflow(module_path: str, name: Optional[str] = None) -> Callable:
    """Import ``module_path`` and return the matching @workflow function.

    If ``name`` is given, look for a function whose ``_workflow_metadata['name']``
    matches; otherwise expect exactly one decorated function in the module.

    Raises ``LookupError`` if zero or multiple workflows match.
    """
    module = importlib.import_module(module_path)
    candidates = [
        attr
        for attr in vars(module).values()
        if callable(attr) and hasattr(attr, "_workflow_metadata")
    ]
    if name is not None:
        candidates = [c for c in candidates if c._workflow_metadata["name"] == name]
        if not candidates:
            raise LookupError(f"No @workflow function named {name!r} in module {module_path!r}")
        if len(candidates) > 1:
            raise LookupError(
                f"Multiple @workflow functions named {name!r} in module {module_path!r}"
            )
        return candidates[0]
    if not candidates:
        raise LookupError(f"No @workflow function found in module {module_path!r}")
    if len(candidates) > 1:
        names = sorted(c._workflow_metadata["name"] for c in candidates)
        raise LookupError(
            f"Multiple @workflow functions in {module_path!r}: {names}; "
            "set WORKFLOW_NAME to choose one"
        )
    return candidates[0]


def build_app(workflow_func: Callable, platform: Any = None) -> FastAPI:
    """Build a FastAPI app that serves a single @workflow function.

    Args:
        workflow_func: A function decorated with ``@workflow``. Its
            ``_workflow_metadata`` drives the handler choice and
            reliability settings.
        platform: A ``GenAIPlatform`` instance. Required for
            ``response_mode="async"`` (the handler calls
            ``platform.workflows.create_job/complete_job/fail_job``).
            Tests for sync/stream may pass ``None``.
    """
    metadata = workflow_func._workflow_metadata
    response_mode = metadata["response_mode"]

    app = FastAPI()

    @app.get("/health/live")
    def live() -> dict:
        return {"status": "ok"}

    @app.get("/health/ready")
    def ready() -> dict:
        return {"status": "ok"}

    if response_mode == "sync":
        app.post(metadata["api_path"])(_create_sync_handler(workflow_func, metadata))
    elif response_mode == "stream":
        app.post(metadata["api_path"])(_create_stream_handler(workflow_func, metadata))
    elif response_mode == "async":
        app.post(metadata["api_path"])(_create_async_handler(workflow_func, metadata, platform))
    else:
        raise ValueError(f"Unknown response_mode {response_mode!r}")

    return app


def _create_sync_handler(workflow_func: Callable, metadata: dict) -> Callable:
    """Listing 8.4: run the function in a worker thread under a per-request timeout."""
    timeout = metadata.get("timeout_seconds", 30)

    async def handler(request: Request):
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse({"error": "request body must be a JSON object"}, status_code=400)

        trace_token = _trace_var.set(request.headers.get("x-trace-id"))
        try:
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(workflow_func, **body),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                return JSONResponse(
                    {"error": f"workflow timed out after {timeout}s"},
                    status_code=504,
                )
            except TypeError as e:
                # Bad call shape (missing/unexpected kwargs) — surface as 422.
                return JSONResponse({"error": str(e)}, status_code=422)
            except Exception as e:  # noqa: BLE001 — surface anything else as 500
                logger.exception("workflow %s raised", metadata["name"])
                return JSONResponse({"error": str(e)}, status_code=500)
            return JSONResponse(result if result is not None else {})
        finally:
            _trace_var.reset(trace_token)

    return handler


def _create_stream_handler(workflow_func: Callable, metadata: dict) -> Callable:
    """Listing 8.6: SSE stream of chunks yielded by the function."""
    timeout = metadata.get("timeout_seconds", 30)

    async def handler(request: Request):
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse({"error": "request body must be a JSON object"}, status_code=400)

        async def event_stream():
            loop = asyncio.get_running_loop()
            sentinel = object()
            try:
                gen_iter = await loop.run_in_executor(None, lambda: iter(workflow_func(**body)))
            except Exception as e:  # noqa: BLE001
                yield _sse_event({"error": str(e)})
                return
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        loop.run_in_executor(None, next, gen_iter, sentinel),
                        timeout=timeout,
                    )
                except asyncio.TimeoutError:
                    yield _sse_event({"error": f"chunk timed out after {timeout}s"})
                    return
                except Exception as e:  # noqa: BLE001
                    yield _sse_event({"error": str(e)})
                    return
                if chunk is sentinel:
                    return
                yield _sse_event(chunk)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return handler


def _create_async_handler(workflow_func: Callable, metadata: dict, platform: Any) -> Callable:
    """Listing 8.8: create a job, return 202+job_id, run the function in background."""
    if platform is None:
        raise ValueError(
            "response_mode='async' requires a platform instance "
            "(passed to build_app for create_job/complete_job/fail_job)"
        )

    # Imported lazily so non-async build_app() callers don't pull in the
    # gRPC channel just to serve sync requests.
    from genai_platform.clients.workflow import job_id_var

    workflow_name = metadata["name"]

    async def handler(request: Request):
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse({"error": "request body must be a JSON object"}, status_code=400)

        job_id = platform.workflows.create_job(
            workflow_id=workflow_name,
            input_json=json.dumps(body),
        )

        async def run_in_background() -> None:
            token = job_id_var.set(job_id)
            try:
                try:
                    result = await asyncio.to_thread(workflow_func, **body)
                except Exception as e:  # noqa: BLE001
                    logger.exception("async workflow %s failed", workflow_name)
                    platform.workflows.fail_job(job_id, str(e))
                    return
                platform.workflows.complete_job(
                    job_id, json.dumps(result if result is not None else {})
                )
            finally:
                job_id_var.reset(token)

        # Fire-and-forget. The Workflow Service owns durability — the gateway's
        # /jobs/{job_id} proxy will surface the eventual outcome to clients.
        task = asyncio.create_task(run_in_background())
        # Stash the task on the app state so tests can await it deterministically.
        request.app.state.last_async_task = task

        return JSONResponse(
            {"job_id": job_id, "status": "pending", "status_url": f"/jobs/{job_id}"},
            status_code=202,
        )

    return handler


def _sse_event(payload: Any) -> bytes:
    """Encode a payload as a Server-Sent Event line: ``data: {json}\\n\\n``."""
    return f"data: {json.dumps(payload)}\n\n".encode("utf-8")


def main() -> None:
    """Container entrypoint.

    Reads ``WORKFLOW_NAME`` and ``WORKFLOW_MODULE`` (or accepts the module
    as a CLI arg) from the environment, imports the module, finds the
    matching @workflow function, and starts uvicorn.
    """
    import uvicorn

    from genai_platform import GenAIPlatform

    if len(sys.argv) >= 2:
        module_path = sys.argv[1]
    else:
        module_path = os.environ.get("WORKFLOW_MODULE")
        if not module_path:
            print(
                "Usage: python -m genai_platform.runtime.server <module>\n"
                "       (or set WORKFLOW_MODULE / WORKFLOW_NAME env vars)",
                file=sys.stderr,
            )
            sys.exit(2)

    workflow_name = os.environ.get("WORKFLOW_NAME")
    func = find_workflow(module_path, workflow_name)

    gateway_url = os.environ.get("GENAI_GATEWAY_URL", "localhost:50051")
    platform = GenAIPlatform(gateway_url=gateway_url)
    app = build_app(func, platform=platform)

    host = os.environ.get("RUNTIME_HOST", "0.0.0.0")
    port = int(os.environ.get("RUNTIME_PORT", "8000"))
    workers = int(os.environ.get("RUNTIME_WORKERS", "1"))  # Listing 8.12 uses 4

    uvicorn.run(app, host=host, port=port, workers=workers)


if __name__ == "__main__":
    main()
