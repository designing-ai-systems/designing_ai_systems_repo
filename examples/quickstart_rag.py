"""
Retrieval-Augmented Generation (RAG) example.

Demonstrates the full RAG pipeline through the SDK:
  1. Create an index via the Data Service
  2. Ingest a fictional document (chunking + embedding handled server-side)
  3. Search for relevant context via semantic search
  4. Generate an answer using the Model Service with retrieved context

**Requires:**
  - OPENAI_API_KEY in .env (used for both embeddings and chat)
  - PostgreSQL + pgvector locally (see README "Optional: PostgreSQL storage")

The script runs against ``VECTOR_STORE=pgvector`` so indexes, documents, and
ingest-job records persist to Postgres — re-running the script cleans up any
prior ``globotech`` index first so it's safely idempotent.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Chapter 5: Data Service (ingestion, search, RAG)
  - Chapter 3: Model Service (chat, embeddings)
"""

import os
import sys
import tempfile
import threading
import time
from pathlib import Path

os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "0")

sys.path.insert(0, str(Path(__file__).parent.parent))

# Default to pgvector persistence so every run writes to the same Postgres
# database. Override in .env if you want in-memory for a quick smoke test.
os.environ.setdefault("VECTOR_STORE", "pgvector")
os.environ.setdefault("DB_CONNECTION_STRING", "postgresql://localhost/genai_platform")

from genai_platform import GenAIPlatform
from services.data.models import IndexConfig
from services.data.service import DataService
from services.data.store import create_vector_store
from services.gateway.registry import ServiceRegistry
from services.gateway.servers import create_grpc_server as create_gateway_grpc
from services.gateway.servers import create_http_server
from services.models.main import load_env_file
from services.models.service import ModelService
from services.shared.server import create_grpc_server, get_service_port

# ---------------------------------------------------------------------------
# Fictional internal document -- every fact here is made up.
# An LLM cannot answer these questions without this context.
# ---------------------------------------------------------------------------
GLOBOTECH_HANDBOOK = """
GloboTech Industries Employee Handbook (Internal — Confidential)
Effective Date: March 1, 2026

Section 1: Paid Time Off
All full-time employees receive exactly 27 days of paid vacation per year.
Unused vacation days roll over, but the maximum carryover is 8 days.
Employees in the Zurich office receive an additional 3 floating holidays.

Section 2: Remote Work
Employees may work remotely up to 3 days per week. Wednesday is a
mandatory in-office day for all teams. Remote work from outside the
employee's home country requires written approval from the VP of People
Operations, Priya Chandrasekaran, at least 14 business days in advance.

Section 3: Expense Policy
The daily meal allowance during business travel is $67 USD. Flights
over 6 hours qualify for business class. Employees must submit
expense reports within 11 calendar days of trip completion using the
internal tool "SpendTrack 4.0".

Section 4: Annual Bonus
The annual bonus target is 14% of base salary for individual contributors
and 19% for managers. Bonuses are paid in the March payroll cycle.
The bonus multiplier is determined by the "Quasar Score", a proprietary
performance rating on a scale of 0-150.

Section 5: Pet Policy
GloboTech allows dogs under 25 pounds in the Austin and Portland offices
on Tuesdays and Thursdays only. All pets must be registered with Facilities
using form GT-PET-2026. The Zurich and Singapore offices do not permit pets.
""".strip()


def _openai_embed_fn():
    """Return an embed_fn(texts, model) that calls OpenAI directly.

    The Data Service's default `embed_fn` is a placeholder that returns zero
    vectors, which silently breaks semantic retrieval. Wire a real one here
    against the OpenAI SDK so ingestion produces meaningful embeddings.
    """
    from openai import OpenAI

    client = OpenAI()

    def _embed(texts, model):
        resp = client.embeddings.create(model=model, input=list(texts))
        return [d.embedding for d in resp.data]

    return _embed


def _start_servers():
    """Start Data Service, Model Service, and Gateway. Returns (servicer, servers)."""
    from http.server import HTTPServer

    HTTPServer.allow_reuse_address = True

    # Load .env for API keys
    project_root = Path(__file__).resolve().parents[1]
    load_env_file(project_root / ".env")

    # Data Service — explicit backing store + real embedding fn so the
    # pipeline uses persistent pgvector storage and OpenAI embeddings.
    data_port = get_service_port("data")
    data_servicer = DataService(
        vector_store=create_vector_store(),
        embed_fn=_openai_embed_fn(),
    )
    data_server = create_grpc_server(servicer=data_servicer, port=data_port, service_name="data")
    data_server.start()

    # Model Service
    models_port = get_service_port("models")
    model_servicer = ModelService()
    model_server = create_grpc_server(
        servicer=model_servicer, port=models_port, service_name="models"
    )
    model_server.start()

    # Gateway
    registry = ServiceRegistry()
    registry.register_platform_service(
        "data", os.getenv("DATA_SERVICE_ADDR", f"localhost:{data_port}")
    )
    registry.register_platform_service(
        "models", os.getenv("MODELS_SERVICE_ADDR", f"localhost:{models_port}")
    )

    grpc_port = int(os.getenv("GATEWAY_PORT", "50051"))
    gateway_server = create_gateway_grpc(registry, grpc_port)
    gateway_server.start()

    http_port = int(os.getenv("GATEWAY_HTTP_PORT", "8080"))
    http_server = create_http_server(registry, http_port)
    http_thread = threading.Thread(target=http_server.serve_forever, daemon=True)
    http_thread.start()

    return data_servicer, data_server, model_server, gateway_server, http_server


def _stop_servers(data_servicer, data_server, model_server, gateway_server, http_server):
    """Gracefully stop all servers and drain the worker pool, freeing ports."""
    data_servicer.close(timeout=5)
    http_server.shutdown()
    gateway_server.stop(grace=0)
    model_server.stop(grace=0)
    data_server.stop(grace=0)


def wait_for_ingest(platform, job_id: str, timeout: float = 30.0):
    """Poll until ingestion completes or times out."""
    start = time.time()
    while time.time() - start < timeout:
        job = platform.data.get_ingest_status(job_id)
        if job.status == "completed":
            return job
        if job.status == "failed":
            raise RuntimeError(f"Ingestion failed: {job.error}")
        time.sleep(0.5)
    raise TimeoutError("Ingestion timed out")


def main():
    print("=" * 60)
    print("  RAG Example — Data + Model Service (Chapters 3 & 5)")
    print("  Requires: OPENAI_API_KEY")
    print("=" * 60)

    print("\nStarting services...")
    data_servicer, data_server, model_server, gateway_server, http_server = _start_servers()
    time.sleep(1)
    print(f"Services ready (vector store: {os.environ['VECTOR_STORE']})!\n")

    handbook_path = None
    try:
        platform = GenAIPlatform()

        # --- 1. Create index ---
        # With pgvector the index persists across runs; drop any prior one so
        # the script is idempotent.
        try:
            platform.data.delete_index("globotech")
            print("[0] Removed prior 'globotech' index (idempotent setup).")
        except Exception:
            pass

        print("[1] Creating index 'globotech'...")
        config = IndexConfig(
            name="globotech",
            embedding_model="text-embedding-3-small",
            embedding_dimensions=1536,
            chunking_strategy="recursive",
            chunk_size=150,
            chunk_overlap=20,
        )
        index = platform.data.create_index(config=config, owner="demo")
        print(f"    Index created: {index.name}")

        # --- 2. Ingest the handbook ---
        # Write to a temp file and read bytes back, mirroring real usage
        # where you'd ingest an actual file from disk.
        handbook_path = Path(tempfile.mkdtemp()) / "globotech_handbook.txt"
        handbook_path.write_text(GLOBOTECH_HANDBOOK)

        print("\n[2] Ingesting GloboTech Employee Handbook...")
        print(f"    Temp file: {handbook_path}")
        job = platform.data.ingest(
            index_name="globotech",
            filename=handbook_path.name,
            content=handbook_path.read_bytes(),
            metadata={"source": "handbook", "version": "2026-03"},
        )
        result = wait_for_ingest(platform, job.job_id)
        print(f"    Document ID : {result.document_id}")
        print(f"    Status      : {result.status}")

        # --- 3. Ask questions using RAG ---
        questions = [
            "How many vacation days do employees get?",
            "What is the daily meal allowance during travel?",
            "Which day is mandatory in-office?",
            "What is the pet policy in the Austin office?",
            "Who approves international remote work requests?",
        ]

        print("\n[3] Retrieval-Augmented Q&A")
        print("-" * 60)

        for q in questions:
            results = platform.data.search(
                index_name="globotech",
                query=q,
                top_k=3,
            )

            context = "\n---\n".join(r.text for r in results)

            print(f"\n  Q: {q}")
            print(f"  Retrieved {len(results)} chunks")

            response = platform.models.chat(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Answer the question using ONLY the context below. "
                            "If the context doesn't contain the answer, say so.\n\n"
                            f"Context:\n{context}"
                        ),
                    },
                    {"role": "user", "content": q},
                ],
                temperature=0.0,
            )
            print(f"  A: {response.content}")

        # --- 4. Show embedding models available ---
        print("\n" + "-" * 60)
        embedding_models = platform.models.list_embedding_models()
        print(f"Available embedding models: {[m.name for m in embedding_models]}")

        print("\nDone!")

    finally:
        if handbook_path and handbook_path.exists():
            handbook_path.unlink(missing_ok=True)
            handbook_path.parent.rmdir()
        _stop_servers(data_servicer, data_server, model_server, gateway_server, http_server)
        time.sleep(0.5)


if __name__ == "__main__":
    main()
