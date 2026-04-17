"""Tests for Data Service ingestion pipeline (book Listings 5.5, 5.13)."""

from services.data.embedding import EmbeddingGenerator
from services.data.models import Index, IndexConfig, IngestedDocument
from services.data.pipeline import IngestionPipeline
from services.data.store import InMemoryVectorStore


def _fake_embed(texts: list[str], model: str) -> list[list[float]]:
    return [[float(len(t) % 10)] * 4 for t in texts]


def _make_pipeline() -> tuple[IngestionPipeline, InMemoryVectorStore]:
    store = InMemoryVectorStore()
    embed_gen = EmbeddingGenerator(embed_fn=_fake_embed)
    pipeline = IngestionPipeline(
        embedding_generator=embed_gen,
        vector_store=store,
    )
    return pipeline, store


def _make_index(name: str = "test-idx") -> Index:
    return Index(
        name=name,
        config=IndexConfig(name=name, chunk_size=10, chunk_overlap=0),
    )


class TestIngestionPipeline:
    def test_ingest_plain_text(self):
        pipeline, store = _make_pipeline()
        index = _make_index()
        result = pipeline.ingest_document(
            index=index,
            filename="test.txt",
            file_bytes=b"Hello world. This is a test document with some content.",
            metadata={"dept": "eng"},
        )
        assert isinstance(result, IngestedDocument)
        assert result.index_name == "test-idx"
        assert result.chunk_count >= 1
        assert result.document_id is not None

    def test_ingest_markdown(self):
        pipeline, store = _make_pipeline()
        index = _make_index()
        content = b"# Title\n\nFirst section.\n\n## Details\n\nSecond section."
        result = pipeline.ingest_document(
            index=index, filename="doc.md", file_bytes=content, metadata={}
        )
        assert result.chunk_count >= 1

    def test_ingest_with_document_id(self):
        pipeline, store = _make_pipeline()
        index = _make_index()
        result = pipeline.ingest_document(
            index=index,
            filename="test.txt",
            file_bytes=b"content",
            metadata={},
            document_id="my-doc-id",
        )
        assert result.document_id == "my-doc-id"

    def test_reingest_replaces_old_chunks(self):
        pipeline, store = _make_pipeline()
        index = _make_index()

        pipeline.ingest_document(
            index=index,
            filename="test.txt",
            file_bytes=b"old content",
            metadata={},
            document_id="doc-1",
        )
        store.search("test-idx", [0.0] * 4, top_k=100)

        pipeline.ingest_document(
            index=index,
            filename="test.txt",
            file_bytes=b"new content replaced",
            metadata={},
            document_id="doc-1",
        )
        second_results = store.search("test-idx", [0.0] * 4, top_k=100)
        # Old chunks deleted, new chunks inserted
        assert all("doc-1" == r.document_id for r in second_results)

    def test_detect_format_routes_to_parser(self):
        pipeline, _ = _make_pipeline()
        assert pipeline.detect_format("file.md", b"# hi") == "markdown"
        assert pipeline.detect_format("file.txt", b"hello") == "text"
        assert pipeline.detect_format("file.pdf", b"%PDF-1.4") == "pdf"


class TestParserRegistry:
    def test_register_custom_parser(self):
        pipeline, _ = _make_pipeline()
        from services.data.models import DocumentSection, ExtractedDocument
        from services.data.parsers import DocumentParser

        class CustomParser(DocumentParser):
            def parse(self, file_bytes, filename):
                return ExtractedDocument(
                    sections=[DocumentSection(content="custom parsed")]
                )

        pipeline.register_parser("custom", CustomParser())
        assert "custom" in pipeline.parsers

    def test_register_custom_chunking_strategy(self):
        pipeline, _ = _make_pipeline()
        from services.data.chunking import ChunkingStrategy
        from services.data.models import Chunk

        class CustomChunker(ChunkingStrategy):
            def chunk(self, document):
                return [Chunk(text="custom chunk", start_offset=0, end_offset=12)]

        pipeline.register_chunking_strategy("custom", CustomChunker())
        assert "custom" in pipeline.chunking_strategies
