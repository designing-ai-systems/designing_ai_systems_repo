"""Tests for Data Service embedding generation (book Listing 5.12)."""

import pytest

from services.data.embedding import EmbeddingGenerator
from services.data.models import Chunk


def _fake_embed(texts: list[str], model: str) -> list[list[float]]:
    return [[float(len(t))] * 4 for t in texts]


class TestEmbeddingGenerator:
    def test_requires_fn_or_client(self):
        with pytest.raises(ValueError):
            EmbeddingGenerator()

    def test_embed_chunks(self):
        gen = EmbeddingGenerator(embed_fn=_fake_embed)
        chunks = [
            Chunk(text="hello", start_offset=0, end_offset=5),
            Chunk(text="world!", start_offset=6, end_offset=12),
        ]
        embeddings = gen.embed_chunks(chunks, model="test-model")
        assert len(embeddings) == 2
        assert embeddings[0] == [5.0, 5.0, 5.0, 5.0]
        assert embeddings[1] == [6.0, 6.0, 6.0, 6.0]

    def test_embed_query(self):
        gen = EmbeddingGenerator(embed_fn=_fake_embed)
        vec = gen.embed_query("test query", model="test-model")
        assert len(vec) == 4
        assert vec[0] == float(len("test query"))

    def test_embed_chunks_batching(self):
        call_count = 0

        def counting_embed(texts, model):
            nonlocal call_count
            call_count += 1
            return [[1.0] * 4 for _ in texts]

        gen = EmbeddingGenerator(embed_fn=counting_embed)
        chunks = [
            Chunk(text=f"chunk{i}", start_offset=i, end_offset=i + 1)
            for i in range(250)
        ]
        embeddings = gen.embed_chunks(chunks, model="m", batch_size=100)
        assert len(embeddings) == 250
        assert call_count == 3

    def test_model_client_fallback(self):
        class FakeModelClient:
            def embed(self, texts, model):
                return [[42.0] * 4 for _ in texts]

        gen = EmbeddingGenerator(model_client=FakeModelClient())
        result = gen.embed_query("test", model="m")
        assert result == [42.0, 42.0, 42.0, 42.0]
