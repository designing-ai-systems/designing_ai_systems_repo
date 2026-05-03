"""
Run Claw locally against an existing platform stack.

Requires the platform services to already be reachable (typically via
``docker compose up -d``). The script is idempotent: it registers the
prompt and tools, ensures the ``company-knowledge`` index exists, ingests
the two bundled Markdown docs, and then either runs a one-shot message
from ``--message`` or drops into a small REPL.

Usage::

    uv run python -m claw.run_local --message "What's our vacation policy?" \\
        --user-id sarah
    uv run python -m claw.run_local --user-id sarah   # interactive
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import grpc

from services.data.models import IndexConfig

from .claw import CLAW_KNOWLEDGE_INDEX, claw_assistant, platform, register_claw_assets

KB_DIR = Path(__file__).parent / "knowledge_base"


def ensure_knowledge_base() -> None:
    """Create the index if missing and (re)ingest the bundled Markdown files."""
    try:
        platform.data.get_index(CLAW_KNOWLEDGE_INDEX)
    except grpc.RpcError:
        platform.data.create_index(
            IndexConfig(
                name=CLAW_KNOWLEDGE_INDEX,
                embedding_model="text-embedding-3-small",
                chunking_strategy="recursive",
                chunk_size=512,
                chunk_overlap=50,
                metadata_schema={"department": "string", "doc_type": "string"},
            ),
            owner="claw",
        )
        print(f"  created index {CLAW_KNOWLEDGE_INDEX!r}")

    for path in sorted(KB_DIR.glob("*.md")):
        job = platform.data.ingest(
            index_name=CLAW_KNOWLEDGE_INDEX,
            filename=path.name,
            content=path.read_bytes(),
            content_type="text/markdown",
            metadata={"department": "engineering", "doc_type": "handbook"},
            document_id=path.stem,
        )
        # Poll until the ingest finishes so a follow-up search can find the doc.
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            status = platform.data.get_ingest_status(job.job_id)
            if status.status in {"completed", "failed"}:
                print(f"  ingested {path.name}: {status.status}")
                break
            time.sleep(0.25)
        else:
            print(f"  ingest of {path.name} did not complete in 30s; continuing")


def run_one(message: str, user_id: str, session_id: str | None) -> dict:
    print(f"\n> [{user_id}] {message}")
    result = claw_assistant(message=message, user_id=user_id, session_id=session_id or "")
    print(json.dumps(result, indent=2))
    return result


def repl(user_id: str) -> None:
    print("\nClaw REPL — empty line to exit.\n")
    session_id = ""
    while True:
        try:
            message = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not message:
            return
        result = run_one(message, user_id, session_id)
        session_id = result.get("session_id", session_id)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Claw locally against the platform stack.")
    parser.add_argument("--message", help="One-shot message; omit for an interactive REPL.")
    parser.add_argument("--user-id", default="sarah", help="User ID (default: sarah).")
    parser.add_argument("--session-id", default="", help="Existing session ID (optional).")
    args = parser.parse_args()

    print("Setting up Claw assets (prompt, tools, knowledge base)...")
    register_claw_assets()
    ensure_knowledge_base()

    if args.message:
        run_one(args.message, args.user_id, args.session_id)
    else:
        repl(args.user_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
