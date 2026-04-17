"""Comprehensive Data Service storage tests for both InMemory and PgvectorStore.

Runs the same test suite against both backends using a parameterized fixture.
PgvectorStore tests run only when PostgreSQL is reachable **and** the pgvector
extension is available (``CREATE EXTENSION vector`` succeeds). Session-style
Postgres tests do not require pgvector; these tests do, because PgvectorStore
uses the ``vector`` type.

**Why tests might skip locally**

- **Connection**: Default URL is ``postgresql://localhost/genai_platform_test``
  (current OS user, no password). If auth differs, set ``DB_TEST_URL`` to match
  your install (same as ``tests/test_postgres_storage.py``).
- **Database missing**: Same auto-create logic as session tests: we try to
  ``CREATE DATABASE genai_platform_test`` via the ``postgres`` database when
  the target DB does not exist.
- **pgvector not installed**: Stock PostgreSQL (Homebrew, Postgres.app, etc.)
  does not include pgvector. Install the extension for your Postgres version
  (e.g. ``brew install pgvector`` on macOS), then in the test database run
  ``CREATE EXTENSION vector;`` once as a superuser, or rely on this module’s
  attempt when permissions allow.

**Run with your local server** — export DB_TEST_URL if needed, e.g.:

export DB_TEST_URL="postgresql://USER:PASSWORD@localhost:5432/genai_platform_test"
pytest tests/test_data_comprehensive.py -v
"""

import os
import threading
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from services.data.models import (
    Chunk,
    DocumentMetadata,
    Index,
    IndexConfig,
    IngestJob,
    JobPayload,
    SearchResult,
)
from services.data.store import InMemoryVectorStore

_DEFAULT_TEST_DB = "postgresql://localhost/genai_platform_test"
TEST_DB = os.environ.get("DB_TEST_URL", _DEFAULT_TEST_DB)

_pgvector_available = False
_PGVECTOR_SKIP_REASON = (
    "PostgreSQL + pgvector tests skipped: set DB_TEST_URL if needed, ensure "
    "database exists or can be created, and install pgvector "
    "(CREATE EXTENSION vector must succeed). See module docstring."
)

try:
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

    _pg_connected = False
    try:
        _conn = psycopg2.connect(TEST_DB)
        _conn.close()
        _pg_connected = True
    except psycopg2.OperationalError:
        try:
            _base_url = TEST_DB.rsplit("/", 1)[0] + "/postgres"
            _conn = psycopg2.connect(_base_url)
            _conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            with _conn.cursor() as cur:
                cur.execute("CREATE DATABASE genai_platform_test")
            _conn.close()
            _pg_connected = True
        except Exception as e:
            _PGVECTOR_SKIP_REASON = (
                f"PostgreSQL not reachable or DB could not be created ({TEST_DB!r}): {e}. "
                "Set DB_TEST_URL to your connection string (see test_postgres_storage.py)."
            )

    if _pg_connected:
        _conn = psycopg2.connect(TEST_DB)
        _conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        with _conn.cursor() as cur:
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            except Exception as e:
                _pgvector_available = False
                _PGVECTOR_SKIP_REASON = (
                    "PostgreSQL is reachable but pgvector is not available "
                    f"(CREATE EXTENSION vector failed: {e}). "
                    "Install pgvector for your Postgres version and ensure the role "
                    "can create extensions, or run CREATE EXTENSION vector once as superuser."
                )
            else:
                _pgvector_available = True
        _conn.close()
except ImportError:
    _PGVECTOR_SKIP_REASON = (
        "psycopg2 not installed; install optional dependency: pip install psycopg2-binary"
    )


def _make_chunks(n: int, prefix: str = "chunk") -> list[Chunk]:
    return [
        Chunk(text=f"{prefix} {i}", start_offset=i * 10, end_offset=(i + 1) * 10) for i in range(n)
    ]


def _unit_vec(dim: int, idx: int) -> list[float]:
    """Unit vector with 1.0 at position idx."""
    vec = [0.0] * dim
    vec[idx % dim] = 1.0
    return vec


# ---------------------------------------------------------------------------
# Parameterized fixture: same tests run against both backends
# ---------------------------------------------------------------------------


_PG_TRUNCATE = "TRUNCATE chunks, documents, ingest_jobs, data_indexes CASCADE"


@pytest.fixture(params=["memory", "pgvector"])
def store(request):
    if request.param == "memory":
        yield InMemoryVectorStore()
    else:
        if not _pgvector_available:
            pytest.skip(_PGVECTOR_SKIP_REASON)
        from services.data.pgvector_store import PgvectorStore

        s = PgvectorStore(connection_string=TEST_DB)
        with s.conn.cursor() as cur:
            cur.execute(_PG_TRUNCATE)
        s.conn.commit()
        yield s
        with s.conn.cursor() as cur:
            cur.execute(_PG_TRUNCATE)
        s.conn.commit()
        s.conn.close()


@pytest.fixture
def pg_store():
    """PgvectorStore-only fixture for keyword search tests."""
    if not _pgvector_available:
        pytest.skip(_PGVECTOR_SKIP_REASON)
    from services.data.pgvector_store import PgvectorStore

    s = PgvectorStore(connection_string=TEST_DB)
    with s.conn.cursor() as cur:
        cur.execute(_PG_TRUNCATE)
    s.conn.commit()
    yield s
    with s.conn.cursor() as cur:
        cur.execute(_PG_TRUNCATE)
    s.conn.commit()
    s.conn.close()


# ===========================================================================
# Insert tests
# ===========================================================================


class TestInsert:
    def test_insert_returns_count(self, store):
        chunks = _make_chunks(3)
        embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        count = store.insert("idx", "doc-1", chunks, embeddings, {"dept": "eng"})
        assert count == 3

    def test_insert_single_chunk(self, store):
        chunks = _make_chunks(1)
        count = store.insert("idx", "doc-1", chunks, [[1.0, 0.0]], {})
        assert count == 1

    def test_insert_multiple_documents(self, store):
        c1 = _make_chunks(2, prefix="alpha")
        c2 = _make_chunks(3, prefix="beta")
        store.insert("idx", "doc-1", c1, [[1.0, 0.0]] * 2, {})
        store.insert("idx", "doc-2", c2, [[0.0, 1.0]] * 3, {})
        results = store.search("idx", [1.0, 0.0], top_k=10)
        assert len(results) == 5

    def test_insert_preserves_metadata(self, store):
        chunks = _make_chunks(1)
        store.insert("idx", "doc-1", chunks, [[1.0, 0.0]], {"dept": "eng", "team": "ml"})
        results = store.search("idx", [1.0, 0.0], top_k=1)
        assert results[0].metadata["dept"] == "eng"
        assert results[0].metadata["team"] == "ml"


# ===========================================================================
# Search tests
# ===========================================================================


class TestSearch:
    def test_basic_search(self, store):
        chunks = _make_chunks(3)
        embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        store.insert("idx", "doc-1", chunks, embeddings, {})
        results = store.search("idx", [1.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].text == "chunk 0"
        assert results[0].score > 0.9

    def test_top_k_limits_results(self, store):
        chunks = _make_chunks(10)
        embeddings = [_unit_vec(10, i) for i in range(10)]
        store.insert("idx", "doc-1", chunks, embeddings, {})
        results = store.search("idx", _unit_vec(10, 0), top_k=3)
        assert len(results) == 3

    def test_results_sorted_by_score(self, store):
        chunks = _make_chunks(3)
        embeddings = [[1.0, 0.0], [0.7, 0.7], [0.0, 1.0]]
        store.insert("idx", "doc-1", chunks, embeddings, {})
        results = store.search("idx", [1.0, 0.0], top_k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_score_threshold(self, store):
        chunks = _make_chunks(3)
        embeddings = [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]
        store.insert("idx", "doc-1", chunks, embeddings, {})
        results = store.search("idx", [1.0, 0.0], top_k=10, score_threshold=0.8)
        for r in results:
            assert r.score >= 0.8

    def test_metadata_filter(self, store):
        c1 = _make_chunks(2, prefix="eng")
        c2 = _make_chunks(2, prefix="legal")
        store.insert("idx", "doc-1", c1, [[1.0, 0.0]] * 2, {"dept": "eng"})
        store.insert("idx", "doc-2", c2, [[0.9, 0.1]] * 2, {"dept": "legal"})
        results = store.search("idx", [1.0, 0.0], top_k=10, metadata_filters={"dept": "legal"})
        assert len(results) == 2
        assert all(r.metadata["dept"] == "legal" for r in results)

    def test_search_empty_index(self, store):
        results = store.search("nonexistent", [1.0, 0.0], top_k=5)
        assert results == []

    def test_index_isolation(self, store):
        chunks = _make_chunks(2)
        store.insert("idx-a", "doc-1", chunks, [[1.0, 0.0]] * 2, {})
        store.insert("idx-b", "doc-2", chunks, [[0.0, 1.0]] * 2, {})
        results = store.search("idx-a", [1.0, 0.0], top_k=10)
        assert len(results) == 2
        assert all(r.document_id == "doc-1" for r in results)

    def test_search_returns_correct_document_id(self, store):
        chunks = _make_chunks(1)
        store.insert("idx", "my-doc-42", chunks, [[1.0, 0.0]], {})
        results = store.search("idx", [1.0, 0.0], top_k=1)
        assert results[0].document_id == "my-doc-42"

    def test_search_returns_chunk_id(self, store):
        chunks = _make_chunks(1)
        store.insert("idx", "doc-1", chunks, [[1.0, 0.0]], {})
        results = store.search("idx", [1.0, 0.0], top_k=1)
        assert results[0].chunk_id  # non-empty UUID string


# ===========================================================================
# Delete tests
# ===========================================================================


class TestDelete:
    def test_delete_by_document(self, store):
        store.insert("idx", "doc-1", _make_chunks(3), [[1.0, 0.0]] * 3, {})
        store.insert("idx", "doc-2", _make_chunks(2), [[0.0, 1.0]] * 2, {})
        deleted = store.delete_by_document("idx", "doc-1")
        assert deleted == 3
        results = store.search("idx", [1.0, 0.0], top_k=10)
        assert len(results) == 2

    def test_delete_by_document_nonexistent(self, store):
        deleted = store.delete_by_document("idx", "doc-999")
        assert deleted == 0

    def test_delete_index(self, store):
        store.insert("idx", "doc-1", _make_chunks(3), [[1.0, 0.0]] * 3, {})
        store.insert("idx", "doc-2", _make_chunks(2), [[0.0, 1.0]] * 2, {})
        deleted = store.delete_index("idx")
        assert deleted == 5
        results = store.search("idx", [1.0, 0.0], top_k=10)
        assert results == []

    def test_delete_index_nonexistent(self, store):
        deleted = store.delete_index("no-such-index")
        assert deleted == 0

    def test_delete_does_not_affect_other_indexes(self, store):
        store.insert("idx-a", "doc-1", _make_chunks(2), [[1.0, 0.0]] * 2, {})
        store.insert("idx-b", "doc-2", _make_chunks(3), [[0.0, 1.0]] * 3, {})
        store.delete_index("idx-a")
        results = store.search("idx-b", [0.0, 1.0], top_k=10)
        assert len(results) == 3


# ===========================================================================
# Keyword search (pgvector only)
# ===========================================================================


class TestKeywordSearch:
    def test_inmemory_raises(self):
        store = InMemoryVectorStore()
        with pytest.raises(NotImplementedError):
            store.keyword_search("idx", "query")

    def test_basic_keyword_search(self, pg_store):
        chunks = [
            Chunk(
                text="The quick brown fox jumps over the lazy dog",
                start_offset=0,
                end_offset=44,
            ),
            Chunk(
                text="PostgreSQL is a powerful relational database",
                start_offset=0,
                end_offset=45,
            ),
            Chunk(
                text="Foxes are clever animals found in forests",
                start_offset=0,
                end_offset=41,
            ),
        ]
        embeddings = [[1.0, 0.0, 0.0]] * 3
        pg_store.insert("idx", "doc-1", chunks, embeddings, {})
        results = pg_store.keyword_search("idx", "fox", top_k=5)
        assert len(results) >= 1
        assert any("fox" in r.text.lower() for r in results)

    def test_keyword_search_no_match(self, pg_store):
        chunks = [
            Chunk(text="Hello world", start_offset=0, end_offset=11),
        ]
        pg_store.insert("idx", "doc-1", chunks, [[1.0, 0.0]], {})
        results = pg_store.keyword_search("idx", "xylophone", top_k=5)
        assert results == []

    def test_keyword_search_metadata_filter(self, pg_store):
        chunks_eng = [
            Chunk(
                text="Machine learning models improve predictions",
                start_offset=0,
                end_offset=43,
            ),
        ]
        chunks_legal = [
            Chunk(
                text="Machine learning compliance requires audits",
                start_offset=0,
                end_offset=43,
            ),
        ]
        pg_store.insert("idx", "doc-1", chunks_eng, [[1.0, 0.0]], {"dept": "eng"})
        pg_store.insert("idx", "doc-2", chunks_legal, [[0.0, 1.0]], {"dept": "legal"})
        results = pg_store.keyword_search(
            "idx", "machine learning", top_k=10, metadata_filters={"dept": "legal"}
        )
        assert len(results) == 1
        assert results[0].metadata["dept"] == "legal"


# ===========================================================================
# End-to-end ingestion flow (both backends)
# ===========================================================================


class TestEndToEndIngestion:
    """Tests the full pipeline: DataService -> VectorStore -> Search."""

    def test_ingest_and_search(self, store):
        from proto import data_pb2
        from services.data.service import DataService

        def fake_embed(texts, model):
            return [[float(len(t) % 10)] * 4 for t in texts]

        svc = DataService(vector_store=store, embed_fn=fake_embed)
        try:
            ctx = MagicMock()
            ctx.invocation_metadata.return_value = [("x-target-service", "data")]

            config = data_pb2.IndexConfig(
                name="e2e-idx",
                chunk_size=20,
                chunk_overlap=0,
                embedding_model="fake",
                embedding_dimensions=4,
            )
            svc.CreateIndex(data_pb2.CreateIndexRequest(config=config), ctx)

            resp = svc.IngestDocument(
                data_pb2.IngestDocumentRequest(
                    index_name="e2e-idx",
                    filename="handbook.txt",
                    content=b"The company handbook covers travel policies and pet policies.",
                    document_id="doc-handbook",
                    metadata={"source": "hr"},
                ),
                ctx,
            )
            assert resp.job_id

            for _ in range(50):
                job = svc.GetIngestJob(data_pb2.GetIngestJobRequest(job_id=resp.job_id), ctx)
                if job.status in ("completed", "failed"):
                    break
                time.sleep(0.1)
            assert job.status == "completed", job.error

            search_resp = svc.Search(
                data_pb2.SearchRequest(index_name="e2e-idx", query="travel", top_k=5),
                ctx,
            )
            assert len(search_resp.results) >= 1
        finally:
            svc.close(timeout=5)

    def test_ingest_multiple_docs_and_delete(self, store):
        from proto import data_pb2
        from services.data.service import DataService

        def fake_embed(texts, model):
            return [[1.0, 0.0, 0.0, 0.0]] * len(texts)

        svc = DataService(vector_store=store, embed_fn=fake_embed)
        try:
            ctx = MagicMock()
            ctx.invocation_metadata.return_value = [("x-target-service", "data")]

            config = data_pb2.IndexConfig(
                name="multi-idx",
                chunk_size=50,
                chunk_overlap=0,
                embedding_model="fake",
                embedding_dimensions=4,
            )
            svc.CreateIndex(data_pb2.CreateIndexRequest(config=config), ctx)

            for doc_id, content in [("d1", b"First document."), ("d2", b"Second document.")]:
                resp = svc.IngestDocument(
                    data_pb2.IngestDocumentRequest(
                        index_name="multi-idx",
                        filename=f"{doc_id}.txt",
                        content=content,
                        document_id=doc_id,
                    ),
                    ctx,
                )
                for _ in range(50):
                    job = svc.GetIngestJob(data_pb2.GetIngestJobRequest(job_id=resp.job_id), ctx)
                    if job.status in ("completed", "failed"):
                        break
                    time.sleep(0.1)
                assert job.status == "completed", job.error

            docs = svc.ListDocuments(data_pb2.ListDocumentsRequest(index_name="multi-idx"), ctx)
            assert len(docs.documents) == 2

            svc.DeleteDocument(
                data_pb2.DeleteDocumentRequest(index_name="multi-idx", document_id="d1"),
                ctx,
            )
            docs = svc.ListDocuments(data_pb2.ListDocumentsRequest(index_name="multi-idx"), ctx)
            assert len(docs.documents) == 1
            assert docs.documents[0].document_id == "d2"
        finally:
            svc.close(timeout=5)


# ===========================================================================
# Index metadata (both backends)
# ===========================================================================


def _make_index(name: str = "idx-a", owner: str = "team-a") -> Index:
    return Index(name=name, config=IndexConfig(name=name), owner=owner)


class TestIndexMetadata:
    def test_create_and_get(self, store):
        idx = _make_index("alpha")
        store.create_index(idx)
        got = store.get_index("alpha")
        assert got is not None
        assert got.name == "alpha"
        assert got.owner == "team-a"
        assert got.config.embedding_model == "text-embedding-3-small"

    def test_get_missing_returns_none(self, store):
        assert store.get_index("nope") is None

    def test_list_returns_created(self, store):
        store.create_index(_make_index("a"))
        store.create_index(_make_index("b"))
        names = {i.name for i in store.list_indexes()}
        assert {"a", "b"}.issubset(names)

    def test_create_duplicate_raises(self, store):
        store.create_index(_make_index("dup"))
        with pytest.raises(ValueError):
            store.create_index(_make_index("dup"))

    def test_increment_index_stats(self, store):
        store.create_index(_make_index("counted"))
        ts = datetime.utcnow()
        store.increment_index_stats(
            "counted", documents_delta=7, chunks_delta=42, last_ingested_at=ts
        )
        got = store.get_index("counted")
        assert got.document_count == 7
        assert got.total_chunks == 42
        assert got.last_ingested_at is not None
        # Second increment must stack atomically (exercises the SQL-level
        # `document_count = document_count + %s` path in pgvector).
        store.increment_index_stats(
            "counted", documents_delta=3, chunks_delta=10, last_ingested_at=ts
        )
        got = store.get_index("counted")
        assert got.document_count == 10
        assert got.total_chunks == 52

    def test_delete_index_removes_metadata(self, store):
        store.create_index(_make_index("gone"))
        store.delete_index("gone")
        assert store.get_index("gone") is None


# ===========================================================================
# Document metadata (both backends)
# ===========================================================================


def _make_doc(index_name: str, doc_id: str = "doc-1") -> DocumentMetadata:
    return DocumentMetadata(
        document_id=doc_id,
        index_name=index_name,
        filename=f"{doc_id}.txt",
        chunk_count=3,
    )


class TestDocumentMetadata:
    def test_put_and_get(self, store):
        store.create_index(_make_index("dx"))
        store.put_document(_make_doc("dx", "doc-1"))
        got = store.get_document("dx", "doc-1")
        assert got is not None
        assert got.filename == "doc-1.txt"
        assert got.chunk_count == 3

    def test_get_missing_returns_none(self, store):
        assert store.get_document("no-such-idx", "no-such-doc") is None

    def test_list_documents(self, store):
        store.create_index(_make_index("dx"))
        store.put_document(_make_doc("dx", "a"))
        store.put_document(_make_doc("dx", "b"))
        store.put_document(_make_doc("dx", "c"))
        ids = {d.document_id for d in store.list_documents("dx")}
        assert ids == {"a", "b", "c"}

    def test_delete_by_document_removes_metadata(self, store):
        store.create_index(_make_index("dx"))
        store.put_document(_make_doc("dx", "d1"))
        chunks = _make_chunks(2)
        store.insert("dx", "d1", chunks, [[1.0, 0.0]] * 2, {})
        store.delete_by_document("dx", "d1")
        assert store.get_document("dx", "d1") is None

    def test_delete_index_cascades_documents(self, store):
        store.create_index(_make_index("dx"))
        store.put_document(_make_doc("dx", "d1"))
        store.put_document(_make_doc("dx", "d2"))
        store.delete_index("dx")
        assert store.list_documents("dx") == []
        assert store.get_index("dx") is None


# ===========================================================================
# Durable job queue (both backends)
# ===========================================================================


def _make_payload(content: bytes = b"hello") -> JobPayload:
    return JobPayload(
        filename="doc.txt",
        content=content,
        caller_metadata={"source": "test"},
    )


class TestJobQueue:
    def test_enqueue_and_get(self, store):
        store.create_index(_make_index("jq"))
        job = IngestJob(job_id="j1", status="queued")
        store.enqueue_job(job, "jq", _make_payload())
        got = store.get_job("j1")
        assert got is not None
        assert got.status == "queued"

    def test_claim_returns_payload_and_marks_processing(self, store):
        store.create_index(_make_index("jq"))
        store.enqueue_job(IngestJob(job_id="j1", status="queued"), "jq", _make_payload(b"hi"))
        claimed = store.claim_next_job(worker_id="w1")
        assert claimed is not None
        assert claimed.job.job_id == "j1"
        assert claimed.job.status == "processing"
        assert claimed.index_name == "jq"
        assert claimed.payload.filename == "doc.txt"
        assert claimed.payload.content == b"hi"
        assert claimed.attempt_count == 1
        assert store.get_job("j1").status == "processing"

    def test_claim_is_fifo(self, store):
        store.create_index(_make_index("jq"))
        for i in range(3):
            store.enqueue_job(IngestJob(job_id=f"j{i}", status="queued"), "jq", _make_payload())
            time.sleep(0.005)  # ensure distinct created_at
        c1 = store.claim_next_job("w")
        c2 = store.claim_next_job("w")
        c3 = store.claim_next_job("w")
        assert [c1.job.job_id, c2.job.job_id, c3.job.job_id] == ["j0", "j1", "j2"]

    def test_empty_queue_returns_none(self, store):
        assert store.claim_next_job("w") is None

    def test_update_terminal_status_clears_claim(self, store):
        store.create_index(_make_index("jq"))
        store.enqueue_job(IngestJob(job_id="j1", status="queued"), "jq", _make_payload())
        store.claim_next_job("w")
        store.update_job("j1", status="completed", document_id="d1", progress=1.0)
        # Reaper should NOT re-queue a completed job.
        released = store.release_stale_claims(timedelta(seconds=0))
        assert released == 0
        assert store.get_job("j1").status == "completed"

    def test_update_partial_fields(self, store):
        store.create_index(_make_index("jq"))
        store.enqueue_job(IngestJob(job_id="j1", status="queued"), "jq", _make_payload())
        store.update_job("j1", progress=0.5)
        job = store.get_job("j1")
        assert job.progress == 0.5
        assert job.status == "queued"


# ===========================================================================
# Durable queue concurrency — pgvector only (SKIP LOCKED)
# ===========================================================================


class TestJobQueueConcurrency:
    """Proves two concurrent PgvectorStore workers never claim the same row."""

    def test_two_workers_never_claim_same_job(self, pg_store):
        if not _pgvector_available:
            pytest.skip(_PGVECTOR_SKIP_REASON)
        from services.data.pgvector_store import PgvectorStore

        # Enqueue 20 jobs through one connection.
        pg_store.create_index(_make_index("race"))
        for i in range(20):
            pg_store.enqueue_job(
                IngestJob(job_id=f"r{i}", status="queued"), "race", _make_payload()
            )

        # Spin two independent stores — separate DB connections.
        s1 = PgvectorStore(connection_string=TEST_DB)
        s2 = PgvectorStore(connection_string=TEST_DB)
        claimed_by_1: list = []
        claimed_by_2: list = []

        def drain(store, out):
            while True:
                c = store.claim_next_job(worker_id="wX")
                if c is None:
                    return
                out.append(c.job.job_id)

        try:
            t1 = threading.Thread(target=drain, args=(s1, claimed_by_1))
            t2 = threading.Thread(target=drain, args=(s2, claimed_by_2))
            t1.start()
            t2.start()
            t1.join(timeout=10)
            t2.join(timeout=10)
        finally:
            s1.conn.close()
            s2.conn.close()

        all_claimed = claimed_by_1 + claimed_by_2
        missing = {f"r{i}" for i in range(20)} - set(all_claimed)
        # Load-bearing correctness: every job is claimed exactly once across
        # both workers. Under SKIP LOCKED, concurrent workers pick disjoint
        # rows; the set-size assertion catches any double-claim regression.
        assert len(all_claimed) == 20, f"missing jobs: {missing}"
        assert len(set(all_claimed)) == 20, "a job was claimed by both workers"

    def test_reaper_requeues_stale_processing(self, pg_store):
        if not _pgvector_available:
            pytest.skip(_PGVECTOR_SKIP_REASON)
        pg_store.create_index(_make_index("reap"))
        pg_store.enqueue_job(IngestJob(job_id="rj1", status="queued"), "reap", _make_payload())
        claimed = pg_store.claim_next_job("w1")
        assert claimed is not None
        # Immediately mark as stale by using a zero threshold.
        released = pg_store.release_stale_claims(timedelta(seconds=0))
        assert released == 1
        # Should be re-claimable.
        again = pg_store.claim_next_job("w2")
        assert again is not None
        assert again.job.job_id == "rj1"
        assert again.attempt_count == 2


# ===========================================================================
# Durability — pgvector only (new connection sees prior writes)
# ===========================================================================


def test_pgvector_survives_new_connection():
    if not _pgvector_available:
        pytest.skip(_PGVECTOR_SKIP_REASON)
    from services.data.pgvector_store import PgvectorStore

    first = PgvectorStore(connection_string=TEST_DB)
    try:
        first.create_index(_make_index("durable"))
        first.put_document(_make_doc("durable", "d1"))
        first.enqueue_job(IngestJob(job_id="dj1", status="queued"), "durable", _make_payload())
    finally:
        first.conn.close()

    second = PgvectorStore(connection_string=TEST_DB)
    try:
        assert second.get_index("durable") is not None
        assert second.get_document("durable", "d1") is not None
        assert second.get_job("dj1") is not None
    finally:
        second.delete_index("durable")
        second.conn.close()
