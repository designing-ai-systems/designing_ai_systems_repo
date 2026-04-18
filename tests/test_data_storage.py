"""Tests for Data Service vector store (book Listings 5.16, 5.17, 5.21)."""

from services.data.models import Chunk, SearchResult
from services.data.store import InMemoryVectorStore, VectorStore


def _make_chunks(n: int) -> list[Chunk]:
    return [
        Chunk(text=f"chunk {i}", start_offset=i * 10, end_offset=(i + 1) * 10) for i in range(n)
    ]


def _unit_vec(dim: int, idx: int) -> list[float]:
    """Create a unit vector with 1.0 at position idx."""
    vec = [0.0] * dim
    vec[idx % dim] = 1.0
    return vec


class TestVectorStoreABC:
    def test_abstract_methods(self):
        assert "insert" in VectorStore.__abstractmethods__
        assert "delete_by_document" in VectorStore.__abstractmethods__
        assert "delete_index" in VectorStore.__abstractmethods__
        assert "search" in VectorStore.__abstractmethods__

    def test_keyword_search_default_raises(self):
        store = InMemoryVectorStore()
        try:
            store.keyword_search("idx", "query")
            assert False, "Should have raised"
        except NotImplementedError:
            pass


class TestInMemoryVectorStoreInsert:
    def test_insert_returns_count(self):
        store = InMemoryVectorStore()
        chunks = _make_chunks(3)
        embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        count = store.insert("idx", "doc-1", chunks, embeddings, {"dept": "eng"})
        assert count == 3

    def test_insert_multiple_documents(self):
        store = InMemoryVectorStore()
        c1 = _make_chunks(2)
        c2 = _make_chunks(3)
        store.insert("idx", "doc-1", c1, [[1.0, 0.0]] * 2, {})
        store.insert("idx", "doc-2", c2, [[0.0, 1.0]] * 3, {})
        results = store.search("idx", [1.0, 0.0], top_k=10)
        assert len(results) == 5


class TestInMemoryVectorStoreSearch:
    def test_basic_search(self):
        store = InMemoryVectorStore()
        chunks = _make_chunks(3)
        embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        store.insert("idx", "doc-1", chunks, embeddings, {})
        results = store.search("idx", [1.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].text == "chunk 0"
        assert results[0].score > 0.9

    def test_top_k_limits_results(self):
        store = InMemoryVectorStore()
        chunks = _make_chunks(10)
        embeddings = [_unit_vec(10, i) for i in range(10)]
        store.insert("idx", "doc-1", chunks, embeddings, {})
        results = store.search("idx", _unit_vec(10, 0), top_k=3)
        assert len(results) == 3

    def test_results_sorted_by_score(self):
        store = InMemoryVectorStore()
        chunks = _make_chunks(3)
        embeddings = [[1.0, 0.0], [0.7, 0.7], [0.0, 1.0]]
        store.insert("idx", "doc-1", chunks, embeddings, {})
        results = store.search("idx", [1.0, 0.0], top_k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_score_threshold(self):
        store = InMemoryVectorStore()
        chunks = _make_chunks(3)
        embeddings = [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]
        store.insert("idx", "doc-1", chunks, embeddings, {})
        results = store.search("idx", [1.0, 0.0], top_k=10, score_threshold=0.8)
        for r in results:
            assert r.score >= 0.8

    def test_metadata_filter(self):
        store = InMemoryVectorStore()
        c1 = _make_chunks(2)
        c2 = _make_chunks(2)
        store.insert("idx", "doc-1", c1, [[1.0, 0.0]] * 2, {"dept": "eng"})
        store.insert("idx", "doc-2", c2, [[0.9, 0.1]] * 2, {"dept": "legal"})
        results = store.search("idx", [1.0, 0.0], top_k=10, metadata_filters={"dept": "legal"})
        assert len(results) == 2
        assert all(r.metadata["dept"] == "legal" for r in results)

    def test_search_empty_index(self):
        store = InMemoryVectorStore()
        results = store.search("nonexistent", [1.0, 0.0], top_k=5)
        assert results == []

    def test_index_isolation(self):
        store = InMemoryVectorStore()
        chunks = _make_chunks(2)
        store.insert("idx-a", "doc-1", chunks, [[1.0, 0.0]] * 2, {})
        store.insert("idx-b", "doc-2", chunks, [[0.0, 1.0]] * 2, {})
        results = store.search("idx-a", [1.0, 0.0], top_k=10)
        assert len(results) == 2
        assert all(r.document_id == "doc-1" for r in results)


class TestInMemoryVectorStoreDelete:
    def test_delete_by_document(self):
        store = InMemoryVectorStore()
        store.insert("idx", "doc-1", _make_chunks(3), [[1.0, 0.0]] * 3, {})
        store.insert("idx", "doc-2", _make_chunks(2), [[0.0, 1.0]] * 2, {})
        deleted = store.delete_by_document("idx", "doc-1")
        assert deleted == 3
        results = store.search("idx", [1.0, 0.0], top_k=10)
        assert len(results) == 2

    def test_delete_by_document_nonexistent(self):
        store = InMemoryVectorStore()
        deleted = store.delete_by_document("idx", "doc-999")
        assert deleted == 0

    def test_delete_index(self):
        store = InMemoryVectorStore()
        store.insert("idx", "doc-1", _make_chunks(3), [[1.0, 0.0]] * 3, {})
        store.insert("idx", "doc-2", _make_chunks(2), [[0.0, 1.0]] * 2, {})
        deleted = store.delete_index("idx")
        assert deleted == 5
        results = store.search("idx", [1.0, 0.0], top_k=10)
        assert results == []

    def test_delete_index_nonexistent(self):
        store = InMemoryVectorStore()
        deleted = store.delete_index("no-such-index")
        assert deleted == 0
