"""Comprehensive Data Service storage tests for both InMemory and PgvectorStore.

Runs the same test suite against both backends using a parameterized fixture.
PgvectorStore tests auto-skip when PostgreSQL with pgvector is unavailable.

Set DB_TEST_URL to override the connection string (used in CI).
"""

import os
import time
from unittest.mock import MagicMock

import pytest

from services.data.models import Chunk, SearchResult
from services.data.store import InMemoryVectorStore

_DEFAULT_TEST_DB = "postgresql://localhost/genai_platform_test"
TEST_DB = os.environ.get("DB_TEST_URL", _DEFAULT_TEST_DB)

_pgvector_available = False

try:
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

    try:
        _conn = psycopg2.connect(TEST_DB)
        _conn.close()
        _pgvector_available = True
    except psycopg2.OperationalError:
        try:
            _base_url = TEST_DB.rsplit("/", 1)[0] + "/postgres"
            _conn = psycopg2.connect(_base_url)
            _conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            with _conn.cursor() as cur:
                cur.execute("CREATE DATABASE genai_platform_test")
            _conn.close()
            _pgvector_available = True
        except Exception:
            pass

    if _pgvector_available:
        _conn = psycopg2.connect(TEST_DB)
        _conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        with _conn.cursor() as cur:
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            except Exception:
                _pgvector_available = False
        _conn.close()
except ImportError:
    pass


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


@pytest.fixture(params=["memory", "pgvector"])
def store(request):
    if request.param == "memory":
        yield InMemoryVectorStore()
    else:
        if not _pgvector_available:
            pytest.skip("PostgreSQL with pgvector unavailable")
        from services.data.pgvector_store import PgvectorStore

        s = PgvectorStore(connection_string=TEST_DB)
        yield s
        with s.conn.cursor() as cur:
            cur.execute("TRUNCATE chunks CASCADE")
        s.conn.commit()
        s.conn.close()


@pytest.fixture
def pg_store():
    """PgvectorStore-only fixture for keyword search tests."""
    if not _pgvector_available:
        pytest.skip("PostgreSQL with pgvector unavailable")
    from services.data.pgvector_store import PgvectorStore

    s = PgvectorStore(connection_string=TEST_DB)
    yield s
    with s.conn.cursor() as cur:
        cur.execute("TRUNCATE chunks CASCADE")
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
        assert job.status == "completed"

        search_resp = svc.Search(
            data_pb2.SearchRequest(index_name="e2e-idx", query="travel", top_k=5),
            ctx,
        )
        assert len(search_resp.results) >= 1

    def test_ingest_multiple_docs_and_delete(self, store):
        from proto import data_pb2
        from services.data.service import DataService

        def fake_embed(texts, model):
            return [[1.0, 0.0, 0.0, 0.0]] * len(texts)

        svc = DataService(vector_store=store, embed_fn=fake_embed)
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
            assert job.status == "completed"

        docs = svc.ListDocuments(data_pb2.ListDocumentsRequest(index_name="multi-idx"), ctx)
        assert len(docs.documents) == 2

        svc.DeleteDocument(
            data_pb2.DeleteDocumentRequest(index_name="multi-idx", document_id="d1"),
            ctx,
        )
        docs = svc.ListDocuments(data_pb2.ListDocumentsRequest(index_name="multi-idx"), ctx)
        assert len(docs.documents) == 1
        assert docs.documents[0].document_id == "d2"
