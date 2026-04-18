"""
Comprehensive Data Service test.

Exercises every Data Service capability from Chapter 5 end-to-end through the
SDK (platform.data.*) against both storage backends:
  1. In-memory (always runs)
  2. pgvector (auto-detected; skipped if PostgreSQL + pgvector unavailable)

Categories covered:
  A. Index management       (create / get / list / duplicate / delete)
  B. Async ingestion        (job lifecycle, metadata, counts, NOT_FOUND, replace)
  C. Document management    (list / get / delete / idempotent delete)
  D. Search / retrieval     (semantic, metadata filter, threshold, hybrid, empty)
  E. Durable queue          (pgvector only — inspects ingest_jobs directly)
  F. RAG workflow           (real generation via Model Service)

**Requires:**
  - OPENAI_API_KEY in .env (used for both embeddings and chat)
  - PostgreSQL + pgvector locally (see README) for the pgvector pass

After the pgvector pass the script leaves a ``demo-globotech`` index behind
so you can browse ``data_indexes`` / ``documents`` / ``chunks`` /
``ingest_jobs`` in Postico.
"""

import os
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "0")

sys.path.insert(0, str(Path(__file__).parent.parent))

from genai_platform import GenAIPlatform
from services.data.models import IndexConfig
from services.data.service import DataService
from services.data.store import InMemoryVectorStore
from services.gateway.registry import ServiceRegistry
from services.gateway.servers import create_grpc_server as create_gateway_grpc
from services.gateway.servers import create_http_server
from services.models.main import load_env_file
from services.models.service import ModelService
from services.shared.server import create_grpc_server, get_service_port

TEST_DB = os.environ.get("DB_TEST_URL", "postgresql://localhost/genai_platform_test")
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMS = 1536

# Fictional content so the LLM cannot answer these questions without RAG.
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


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def print_section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def _detect_pgvector() -> bool:
    """Return True if PostgreSQL + pgvector are reachable (and auto-create the test DB)."""
    try:
        import psycopg2
        from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
    except ImportError:
        return False
    try:
        conn = psycopg2.connect(TEST_DB)
        conn.close()
    except psycopg2.OperationalError:
        try:
            base = psycopg2.connect("postgresql://localhost/postgres")
            base.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            with base.cursor() as cur:
                cur.execute("CREATE DATABASE genai_platform_test")
            base.close()
        except Exception:
            return False
    try:
        conn = psycopg2.connect(TEST_DB)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.close()
        return True
    except Exception:
        return False


def _openai_embed_fn():
    """embed_fn(texts, model) -> List[List[float]] via the OpenAI SDK."""
    from openai import OpenAI

    client = OpenAI()

    def _embed(texts, model):
        resp = client.embeddings.create(model=model, input=list(texts))
        return [d.embedding for d in resp.data]

    return _embed


def _wait_for_job(platform, job_id: str, timeout: float = 60.0):
    """Poll get_ingest_status until completed/failed/timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = platform.data.get_ingest_status(job_id)
        if job.status == "completed":
            return job
        if job.status == "failed":
            raise RuntimeError(f"Ingestion failed: {job.error}")
        time.sleep(0.2)
    raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")


# ---------------------------------------------------------------------------
# Test categories
# ---------------------------------------------------------------------------


def test_index_management(platform):
    print_section("A. Index Management")

    print("\n1. Creating index 'team-eng-docs'...")
    cfg = IndexConfig(
        name="team-eng-docs",
        embedding_model=EMBEDDING_MODEL,
        embedding_dimensions=EMBEDDING_DIMS,
        chunking_strategy="recursive",
        chunk_size=200,
        chunk_overlap=20,
    )
    idx = platform.data.create_index(config=cfg, owner="team-eng")
    print(f"   ✓ Created: {idx.name} (owner={idx.owner})")
    assert idx.name == "team-eng-docs"
    assert idx.owner == "team-eng"

    print("\n2. get_index round-trip...")
    got = platform.data.get_index("team-eng-docs")
    print(f"   ✓ Retrieved: {got.name}, dims={got.config.embedding_dimensions}")
    assert got.config.embedding_model == EMBEDDING_MODEL

    print("\n3. list_indexes...")
    listed = platform.data.list_indexes()
    names = [i.name for i in listed]
    print(f"   ✓ Indexes: {names}")
    assert "team-eng-docs" in names

    print("\n4. Creating duplicate 'team-eng-docs' → expect ALREADY_EXISTS...")
    try:
        platform.data.create_index(config=cfg, owner="team-eng")
    except Exception as e:
        print(f"   ✓ Rejected: {type(e).__name__}")
    else:
        raise AssertionError("Duplicate CreateIndex should have raised")

    print("\n5. Creating + deleting 'tmp-idx' to verify cascade path...")
    platform.data.create_index(
        config=IndexConfig(
            name="tmp-idx",
            embedding_model=EMBEDDING_MODEL,
            embedding_dimensions=EMBEDDING_DIMS,
            chunk_size=100,
        ),
        owner="ephemeral",
    )
    ok = platform.data.delete_index("tmp-idx")
    assert ok is True
    try:
        platform.data.get_index("tmp-idx")
    except Exception as e:
        print(f"   ✓ Deleted 'tmp-idx' and confirmed it's gone ({type(e).__name__})")
    else:
        raise AssertionError("get_index on deleted index should have raised NOT_FOUND")


def test_document_ingestion(platform):
    print_section("B. Document Ingestion (async job lifecycle)")

    print("\n1. Ingest one document, poll until completed...")
    job = platform.data.ingest(
        index_name="team-eng-docs",
        filename="handbook.txt",
        content=GLOBOTECH_HANDBOOK.encode("utf-8"),
        metadata={"source": "handbook", "dept": "eng"},
        document_id="doc-handbook",
    )
    print(f"   → job_id={job.job_id[:8]}… status={job.status}")
    completed = _wait_for_job(platform, job.job_id)
    print(f"   ✓ Completed. document_id={completed.document_id}")

    print("\n2. Verify custom metadata preserved on retrieved chunks...")
    hits = platform.data.search(
        index_name="team-eng-docs",
        query="vacation policy",
        top_k=1,
    )
    assert hits, "expected at least one chunk"
    assert hits[0].metadata.get("dept") == "eng"
    print(f"   ✓ metadata on chunk: {hits[0].metadata}")

    print("\n3. Ingest two more docs, verify counts on the index...")
    for i, fname in enumerate(["memo-a.txt", "memo-b.txt"], start=1):
        j = platform.data.ingest(
            index_name="team-eng-docs",
            filename=fname,
            content=f"Internal memo {i} about office pet policy and Wednesday meetings.".encode(),
            metadata={"source": "memo"},
            document_id=f"memo-{i}",
        )
        _wait_for_job(platform, j.job_id)
    got = platform.data.get_index("team-eng-docs")
    print(f"   ✓ index.document_count={got.document_count}, total_chunks={got.total_chunks}")
    assert got.document_count == 3
    assert got.total_chunks >= 3

    print("\n4. Ingest into non-existent index → expect NOT_FOUND...")
    try:
        platform.data.ingest(
            index_name="no-such-idx",
            filename="x.txt",
            content=b"anything",
        )
    except Exception as e:
        print(f"   ✓ Rejected: {type(e).__name__}")
    else:
        raise AssertionError("Ingest into missing index should have failed")

    print("\n5. Replace-on-reingest: same document_id, verify chunks replaced...")
    before = platform.data.get_index("team-eng-docs").total_chunks
    j = platform.data.ingest(
        index_name="team-eng-docs",
        filename="handbook-v2.txt",
        content=GLOBOTECH_HANDBOOK.encode("utf-8") + b"\n\nAppendix A: Updated.",
        metadata={"source": "handbook", "dept": "eng", "version": "2026-04"},
        document_id="doc-handbook",  # same id → replace
    )
    _wait_for_job(platform, j.job_id)
    after = platform.data.get_index("team-eng-docs").total_chunks
    docs = platform.data.list_documents("team-eng-docs")
    doc_ids = [d.document_id for d in docs]
    print(
        f"   ✓ chunks before={before}, after={after}; doc-handbook appears once: "
        f"{doc_ids.count('doc-handbook') == 1}"
    )
    assert doc_ids.count("doc-handbook") == 1, "re-ingest should not duplicate the document row"


def test_document_management(platform):
    print_section("C. Document Management")

    print("\n1. list_documents...")
    docs = platform.data.list_documents("team-eng-docs")
    print(f"   ✓ {len(docs)} documents: {[d.document_id for d in docs]}")
    assert len(docs) >= 3

    print("\n2. get_document (specific id)...")
    got = platform.data.get_document("team-eng-docs", "memo-1")
    print(f"   ✓ filename={got.filename}, chunks={got.chunk_count}")
    assert got.document_id == "memo-1"

    print("\n3. delete_document: remove 'memo-2', verify chunks gone from search...")
    platform.data.delete_document("team-eng-docs", "memo-2")
    remaining_ids = [d.document_id for d in platform.data.list_documents("team-eng-docs")]
    assert "memo-2" not in remaining_ids
    hits = platform.data.search("team-eng-docs", "memo-2", top_k=10)
    assert all(h.document_id != "memo-2" for h in hits)
    print(f"   ✓ memo-2 removed; remaining docs: {remaining_ids}")

    print("\n4. Delete non-existent document → idempotent success...")
    ok = platform.data.delete_document("team-eng-docs", "ghost-doc")
    print(f"   ✓ delete returned {ok} with no error")


def test_search(platform, is_pgvector: bool):
    print_section("D. Search / Retrieval")

    print("\n1. Basic semantic search top_k=3...")
    hits = platform.data.search(
        index_name="team-eng-docs",
        query="How many vacation days do employees get?",
        top_k=3,
    )
    print(f"   ✓ {len(hits)} hits; top score={hits[0].score:.3f}")
    assert len(hits) >= 1
    assert hits[0].score > 0.2, f"top score {hits[0].score} suspiciously low (embeddings wired?)"

    print("\n2. Metadata filter {'dept': 'eng'} on a chunk's custom metadata...")
    hits = platform.data.search(
        index_name="team-eng-docs",
        query="pet policy",
        top_k=5,
        metadata_filters={"dept": "eng"},
    )
    print(f"   ✓ {len(hits)} hits; dept values: {set(h.metadata.get('dept') for h in hits)}")
    assert all(h.metadata.get("dept") == "eng" for h in hits)

    print("\n3. score_threshold drops weak matches...")
    hits = platform.data.search(
        index_name="team-eng-docs",
        query="completely unrelated topic like astrophysics",
        top_k=5,
        score_threshold=0.9,
    )
    print(f"   ✓ high-threshold search returned {len(hits)} hits")
    assert all(h.score >= 0.9 for h in hits)

    if is_pgvector:
        print("\n4. Hybrid search (vector + keyword via RRF) — pgvector only...")
        hits = platform.data.hybrid_search(
            index_name="team-eng-docs",
            query="Wednesday office",
            top_k=3,
        )
        print(f"   ✓ hybrid search returned {len(hits)} hits")
        assert len(hits) >= 1
    else:
        print("\n4. Hybrid search — skipped (in-memory backend does not support it)")

    print("\n5. Search on empty index returns []...")
    platform.data.create_index(
        config=IndexConfig(
            name="empty-idx",
            embedding_model=EMBEDDING_MODEL,
            embedding_dimensions=EMBEDDING_DIMS,
            chunk_size=100,
        ),
        owner="nobody",
    )
    hits = platform.data.search(index_name="empty-idx", query="anything", top_k=5)
    assert hits == []
    print("   ✓ empty index yielded 0 results")
    platform.data.delete_index("empty-idx")


def test_durable_queue(platform, test_db: str):
    """Inspects ingest_jobs directly — pgvector-only."""
    import psycopg2
    from psycopg2.extras import RealDictCursor

    print_section("E. Durable Queue (ingest_jobs inspection)")

    def fetch_rows(job_id):
        with psycopg2.connect(test_db, cursor_factory=RealDictCursor) as c:
            with c.cursor() as cur:
                cur.execute(
                    "SELECT job_id, status, attempt_count, claimed_by, document_id "
                    "FROM ingest_jobs WHERE job_id = %s",
                    (job_id,),
                )
                return cur.fetchone()

    print("\n1. Submit a job; immediately inspect the queue row...")
    job = platform.data.ingest(
        index_name="team-eng-docs",
        filename="queue-check.txt",
        content=b"A tiny document about queue semantics.",
        document_id="queue-doc",
    )
    initial = fetch_rows(job.job_id)
    print(f"   ✓ row exists: status={initial['status']}, attempt_count={initial['attempt_count']}")
    assert initial["status"] in ("queued", "processing", "completed")
    assert initial["attempt_count"] in (0, 1)

    print("\n2. Wait for completion; confirm queue columns reflect the transition...")
    _wait_for_job(platform, job.job_id)
    final = fetch_rows(job.job_id)
    print(
        f"   ✓ final: status={final['status']}, attempt_count={final['attempt_count']}, "
        f"claimed_by={final['claimed_by']}, document_id={final['document_id']}"
    )
    assert final["status"] == "completed"
    assert final["attempt_count"] >= 1
    assert final["claimed_by"] is None, "completed jobs should have claimed_by cleared"

    print("\n3. Resubmitting the same payload produces a distinct job_id...")
    job2 = platform.data.ingest(
        index_name="team-eng-docs",
        filename="queue-check.txt",
        content=b"A tiny document about queue semantics.",
        document_id="queue-doc",
    )
    _wait_for_job(platform, job2.job_id)
    assert job2.job_id != job.job_id
    print(f"   ✓ first={job.job_id[:8]}… second={job2.job_id[:8]}…")


def test_rag_workflow(platform):
    print_section("F. End-to-End RAG Workflow")

    print("\n1. Creating dedicated 'rag-globotech' index...")
    try:
        platform.data.delete_index("rag-globotech")
    except Exception:
        pass
    platform.data.create_index(
        config=IndexConfig(
            name="rag-globotech",
            embedding_model=EMBEDDING_MODEL,
            embedding_dimensions=EMBEDDING_DIMS,
            chunking_strategy="recursive",
            chunk_size=150,
            chunk_overlap=20,
        ),
        owner="rag-demo",
    )

    print("\n2. Ingesting handbook...")
    job = platform.data.ingest(
        index_name="rag-globotech",
        filename="globotech_handbook.txt",
        content=GLOBOTECH_HANDBOOK.encode("utf-8"),
        metadata={"source": "handbook"},
    )
    result = _wait_for_job(platform, job.job_id)
    print(f"   ✓ ingested; document_id={result.document_id}")

    questions = [
        ("How many vacation days do full-time employees get?", "27"),
        ("What is the daily meal allowance during travel?", "67"),
        ("Which day is mandatory in-office?", "wednesday"),
        ("What is the pet policy in the Austin office?", "austin"),
        ("Who approves international remote work requests?", "priya"),
    ]
    print("\n3. Retrieval-augmented Q&A (5 questions)...")
    hits_ok = 0
    for q, expected in questions:
        hits = platform.data.search(index_name="rag-globotech", query=q, top_k=3)
        context = "\n---\n".join(h.text for h in hits)
        resp = platform.models.chat(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"Answer using ONLY the context. If missing, say so.\n\nContext:\n{context}"
                    ),
                },
                {"role": "user", "content": q},
            ],
            temperature=0.0,
        )
        ok = expected.lower() in resp.content.lower()
        hits_ok += int(ok)
        marker = "✓" if ok else "✗"
        print(f"   {marker}  Q: {q}")
        print(f"       A: {resp.content[:100]}")
    print(f"\n   RAG answered {hits_ok}/{len(questions)} correctly")
    assert hits_ok >= 3, "RAG should hit at least 3/5 factual questions"

    platform.data.delete_index("rag-globotech")
    print("   ✓ cleaned up 'rag-globotech'")


# ---------------------------------------------------------------------------
# Service lifecycle
# ---------------------------------------------------------------------------


def _start_servers(vector_store):
    from http.server import HTTPServer

    HTTPServer.allow_reuse_address = True

    project_root = Path(__file__).resolve().parents[1]
    load_env_file(project_root / ".env")

    data_port = get_service_port("data")
    data_servicer = DataService(vector_store=vector_store, embed_fn=_openai_embed_fn())
    data_server = create_grpc_server(servicer=data_servicer, port=data_port, service_name="data")
    data_server.start()

    models_port = get_service_port("models")
    model_servicer = ModelService()
    model_server = create_grpc_server(
        servicer=model_servicer, port=models_port, service_name="models"
    )
    model_server.start()

    registry = ServiceRegistry()
    registry.register_platform_service("data", f"localhost:{data_port}")
    registry.register_platform_service("models", f"localhost:{models_port}")

    grpc_port = int(os.getenv("GATEWAY_PORT", "50051"))
    gateway_server = create_gateway_grpc(registry, grpc_port)
    gateway_server.start()

    http_port = int(os.getenv("GATEWAY_HTTP_PORT", "8080"))
    http_server = create_http_server(registry, http_port)
    http_thread = threading.Thread(target=http_server.serve_forever, daemon=True)
    http_thread.start()

    return data_servicer, data_server, model_server, gateway_server, http_server


def _stop_servers(data_servicer, data_server, model_server, gateway_server, http_server):
    data_servicer.close(timeout=5)
    http_server.shutdown()
    gateway_server.stop(grace=0)
    model_server.stop(grace=0)
    data_server.stop(grace=0)


def run_pass(backend_name: str, vector_store, *, is_pgvector: bool, test_db: str = ""):
    print("\n" + "#" * 60)
    print(f"  BACKEND: {backend_name}")
    print("#" * 60)

    print(f"\n  Starting services (backend={backend_name})...")
    servers = _start_servers(vector_store)
    time.sleep(1)
    print("  ✓ Services started")

    platform = GenAIPlatform()
    try:
        test_index_management(platform)
        test_document_ingestion(platform)
        test_document_management(platform)
        test_search(platform, is_pgvector=is_pgvector)
        if is_pgvector:
            test_durable_queue(platform, test_db)
        test_rag_workflow(platform)

        print_section("Cleanup")
        try:
            platform.data.delete_index("team-eng-docs")
            print("  ✓ Cleaned up 'team-eng-docs'")
        except Exception as e:
            print(f"  (note) cleanup issue: {e}")

        print(f"\n✓ All {backend_name} tests passed!")
    finally:
        _stop_servers(*servers)
        time.sleep(0.5)


def _insert_demo_data(test_db: str):
    """Leave a 'demo-globotech' index in pgvector for manual browsing in Postico."""
    from services.data.pgvector_store import PgvectorStore

    print_section("Demo data for Postico (pgvector)")

    # Use a dedicated DataService against the same pgvector DB.
    store = PgvectorStore(connection_string=test_db)
    # Clear any prior demo rows so only one clean set remains.
    with store.conn.cursor() as cur:
        cur.execute("DELETE FROM data_indexes WHERE name IN ('demo-globotech')")
        cur.execute("DELETE FROM documents WHERE index_name IN ('demo-globotech')")
        cur.execute("DELETE FROM chunks WHERE index_name IN ('demo-globotech')")
        cur.execute("DELETE FROM ingest_jobs WHERE index_name IN ('demo-globotech')")
    store.conn.commit()

    svc = DataService(vector_store=store, embed_fn=_openai_embed_fn())
    try:
        from unittest.mock import MagicMock

        from proto import data_pb2

        ctx = MagicMock()
        ctx.invocation_metadata.return_value = [("x-target-service", "data")]

        svc.CreateIndex(
            data_pb2.CreateIndexRequest(
                config=data_pb2.IndexConfig(
                    name="demo-globotech",
                    embedding_model=EMBEDDING_MODEL,
                    embedding_dimensions=EMBEDDING_DIMS,
                    chunking_strategy="recursive",
                    chunk_size=150,
                    chunk_overlap=20,
                ),
                owner="demo",
            ),
            ctx,
        )
        handbook_job = svc.IngestDocument(
            data_pb2.IngestDocumentRequest(
                index_name="demo-globotech",
                filename="globotech_handbook.txt",
                content=GLOBOTECH_HANDBOOK.encode("utf-8"),
                metadata={"source": "handbook", "version": "2026-03"},
                document_id="demo-handbook",
            ),
            ctx,
        )
        policy_job = svc.IngestDocument(
            data_pb2.IngestDocumentRequest(
                index_name="demo-globotech",
                filename="policy_update_q2.txt",
                content=(
                    b"Q2 2026 Policy Update: effective April 15, the expense report window "
                    b"shortens from 11 to 7 calendar days. All employees must confirm via "
                    b"SpendTrack 4.0."
                ),
                metadata={"source": "policy", "version": "2026-04", "confidential": "true"},
                document_id="demo-policy-q2",
            ),
            ctx,
        )

        for j in (handbook_job, policy_job):
            for _ in range(150):
                got = svc.GetIngestJob(data_pb2.GetIngestJobRequest(job_id=j.job_id), ctx)
                if got.status in ("completed", "failed"):
                    break
                time.sleep(0.2)
        print("  ✓ Demo index 'demo-globotech' has 2 documents and a policy row.")
        print("  ✓ Browse: data_indexes, documents, chunks, ingest_jobs in Postico.")
    finally:
        svc.close(timeout=5)
        store.conn.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("  DATA SERVICE COMPREHENSIVE TEST")
    print("  Chapter 5: ingestion, storage, retrieval")
    print("=" * 60)

    # Load .env early so OPENAI_API_KEY is available for imports that need it.
    load_env_file(Path(__file__).resolve().parents[1] / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        print("\nERROR: OPENAI_API_KEY not set. This test requires real embeddings.")
        print("       Add OPENAI_API_KEY to .env or export it before running.")
        sys.exit(1)

    # --- Pass 1: In-memory ---
    run_pass(
        "In-Memory",
        InMemoryVectorStore(),
        is_pgvector=False,
    )

    # --- Pass 2: pgvector (if available) ---
    if _detect_pgvector():
        # Fresh slate for the test DB so counts/assertions are deterministic.
        import psycopg2

        from services.data.pgvector_store import PgvectorStore

        with psycopg2.connect(TEST_DB) as c:
            with c.cursor() as cur:
                cur.execute("TRUNCATE chunks, documents, ingest_jobs, data_indexes CASCADE")
            c.commit()

        run_pass(
            "pgvector",
            PgvectorStore(connection_string=TEST_DB),
            is_pgvector=True,
            test_db=TEST_DB,
        )
        _insert_demo_data(TEST_DB)
    else:
        print("\n" + "-" * 60)
        print("  Skipping pgvector pass (PostgreSQL + pgvector not available)")
        print("  See README 'Optional: PostgreSQL storage' to enable.")
        print("-" * 60)

    print_section("FINAL SUMMARY")
    print("\nCategories exercised:")
    print("  A. Index management       — create/get/list/duplicate/delete")
    print("  B. Async ingestion        — job lifecycle / metadata / counts / replace")
    print("  C. Document management    — list/get/delete/idempotent")
    print("  D. Search                 — semantic / metadata / threshold / hybrid / empty")
    print("  E. Durable queue          — pgvector only (inspects ingest_jobs)")
    print("  F. RAG workflow           — real retrieval + generation on GloboTech handbook")


if __name__ == "__main__":
    main()
