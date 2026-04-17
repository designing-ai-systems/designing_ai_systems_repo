"""Tests for Data Service search orchestration (book Listings 5.20, 5.23)."""

from services.data.embedding import EmbeddingGenerator
from services.data.models import Chunk, Index, IndexConfig, SearchResult
from services.data.search import SearchOrchestrator, reciprocal_rank_fusion
from services.data.store import InMemoryVectorStore


def _fake_embed(texts: list[str], model: str) -> list[list[float]]:
    """Deterministic fake embedder: embeds based on text length."""
    results = []
    for t in texts:
        n = len(t) % 10
        vec = [0.0] * 10
        vec[n] = 1.0
        results.append(vec)
    return results


def _make_index() -> Index:
    return Index(
        name="test-idx",
        config=IndexConfig(name="test-idx", embedding_model="fake-model"),
    )


class TestReciprocalRankFusion:
    def test_single_list(self):
        results = [
            SearchResult(chunk_id="a", document_id="d1", text="a", score=0.9),
            SearchResult(chunk_id="b", document_id="d1", text="b", score=0.8),
        ]
        fused = reciprocal_rank_fusion(results, [])
        assert len(fused) == 2
        assert fused[0].chunk_id == "a"

    def test_merges_two_lists(self):
        list_a = [
            SearchResult(chunk_id="a", document_id="d1", text="a", score=0.9),
            SearchResult(chunk_id="b", document_id="d1", text="b", score=0.8),
        ]
        list_b = [
            SearchResult(chunk_id="b", document_id="d1", text="b", score=0.95),
            SearchResult(chunk_id="c", document_id="d1", text="c", score=0.7),
        ]
        fused = reciprocal_rank_fusion(list_a, list_b)
        assert len(fused) == 3
        # b appears in both lists so should rank highest
        assert fused[0].chunk_id == "b"

    def test_empty_lists(self):
        fused = reciprocal_rank_fusion([], [])
        assert fused == []

    def test_scores_are_rrf_scores(self):
        list_a = [
            SearchResult(chunk_id="a", document_id="d1", text="a", score=0.9),
        ]
        list_b = [
            SearchResult(chunk_id="a", document_id="d1", text="a", score=0.8),
        ]
        fused = reciprocal_rank_fusion(list_a, list_b)
        assert len(fused) == 1
        # RRF score = 1/(60+1) + 1/(60+1) = 2/61
        expected = 2.0 / 61.0
        assert abs(fused[0].score - expected) < 0.001


class TestSearchOrchestrator:
    def test_search(self):
        store = InMemoryVectorStore()
        embed_gen = EmbeddingGenerator(embed_fn=_fake_embed)
        orchestrator = SearchOrchestrator(
            embedding_generator=embed_gen, vector_store=store
        )
        index = _make_index()

        chunks = [Chunk(text="hello", start_offset=0, end_offset=5)]
        embeddings = _fake_embed(["hello"], "fake-model")
        store.insert("test-idx", "doc-1", chunks, embeddings, {})

        results = orchestrator.search(
            index=index, query="hello", top_k=5
        )
        assert len(results) >= 1
        assert results[0].text == "hello"

    def test_search_with_metadata_filter(self):
        store = InMemoryVectorStore()
        embed_gen = EmbeddingGenerator(embed_fn=_fake_embed)
        orchestrator = SearchOrchestrator(
            embedding_generator=embed_gen, vector_store=store
        )
        index = _make_index()

        c1 = [Chunk(text="hello", start_offset=0, end_offset=5)]
        c2 = [Chunk(text="hello", start_offset=0, end_offset=5)]
        store.insert("test-idx", "doc-1", c1, _fake_embed(["hello"], "m"), {"dept": "eng"})
        store.insert("test-idx", "doc-2", c2, _fake_embed(["hello"], "m"), {"dept": "legal"})

        results = orchestrator.search(
            index=index, query="hello", top_k=5, metadata_filters={"dept": "eng"}
        )
        assert all(r.metadata["dept"] == "eng" for r in results)

    def test_hybrid_search(self):
        store = InMemoryVectorStore()
        embed_gen = EmbeddingGenerator(embed_fn=_fake_embed)
        orchestrator = SearchOrchestrator(
            embedding_generator=embed_gen, vector_store=store
        )
        index = _make_index()

        chunks = [Chunk(text="hello", start_offset=0, end_offset=5)]
        embeddings = _fake_embed(["hello"], "fake-model")
        store.insert("test-idx", "doc-1", chunks, embeddings, {})

        # hybrid_search should not raise even if keyword_search is not supported;
        # it falls back gracefully
        results = orchestrator.hybrid_search(
            index=index, query="hello", top_k=5
        )
        assert len(results) >= 1
