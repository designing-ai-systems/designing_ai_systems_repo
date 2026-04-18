"""Tests for Data Service domain models (book Listings 5.1, 5.3, 5.4, 5.7, 5.8, 5.15, 5.17)."""

from datetime import datetime

from services.data.models import (
    Chunk,
    DocumentMetadata,
    DocumentSection,
    ExtractedDocument,
    Index,
    IndexConfig,
    IngestedDocument,
    IngestJob,
    SearchResult,
)


class TestIndexConfig:
    def test_defaults(self):
        config = IndexConfig(name="test-index")
        assert config.name == "test-index"
        assert config.embedding_model == "text-embedding-3-small"
        assert config.embedding_dimensions == 1536
        assert config.chunking_strategy == "fixed"
        assert config.chunk_size == 512
        assert config.chunk_overlap == 50
        assert config.metadata_schema is None

    def test_custom_values(self):
        config = IndexConfig(
            name="legal-docs",
            embedding_model="text-embedding-3-large",
            embedding_dimensions=3072,
            chunking_strategy="recursive",
            chunk_size=1024,
            chunk_overlap=100,
            metadata_schema={"department": "string"},
        )
        assert config.embedding_model == "text-embedding-3-large"
        assert config.chunk_size == 1024
        assert config.metadata_schema == {"department": "string"}


class TestIndex:
    def test_defaults(self):
        config = IndexConfig(name="my-index")
        index = Index(name="my-index", config=config)
        assert index.owner == ""
        assert index.document_count == 0
        assert index.total_chunks == 0
        assert index.created_at is not None
        assert index.last_ingested_at is None

    def test_with_stats(self):
        config = IndexConfig(name="docs")
        now = datetime.utcnow()
        index = Index(
            name="docs",
            config=config,
            owner="team-a",
            document_count=10,
            total_chunks=500,
            created_at=now,
            last_ingested_at=now,
        )
        assert index.owner == "team-a"
        assert index.document_count == 10
        assert index.total_chunks == 500


class TestDocumentSection:
    def test_simple_section(self):
        section = DocumentSection(content="Hello world")
        assert section.content == "Hello world"
        assert section.heading is None
        assert section.level == 0
        assert section.page_number is None

    def test_with_heading(self):
        section = DocumentSection(
            content="Some text", heading="Introduction", level=1, page_number=3
        )
        assert section.heading == "Introduction"
        assert section.level == 1
        assert section.page_number == 3


class TestExtractedDocument:
    def test_text_property(self):
        sections = [
            DocumentSection(content="First paragraph."),
            DocumentSection(content="Second paragraph."),
        ]
        doc = ExtractedDocument(sections=sections)
        assert doc.text == "First paragraph.\n\nSecond paragraph."

    def test_empty_sections(self):
        doc = ExtractedDocument(sections=[])
        assert doc.text == ""
        assert doc.metadata == {}

    def test_with_metadata(self):
        doc = ExtractedDocument(
            sections=[DocumentSection(content="text")],
            metadata={"author": "Alice"},
        )
        assert doc.metadata == {"author": "Alice"}


class TestDocumentMetadata:
    def test_required_fields(self):
        meta = DocumentMetadata(
            document_id="doc-1",
            index_name="my-index",
            filename="report.pdf",
        )
        assert meta.document_id == "doc-1"
        assert meta.index_name == "my-index"
        assert meta.filename == "report.pdf"
        assert meta.ingested_at is not None
        assert meta.chunk_count == 0

    def test_optional_fields(self):
        meta = DocumentMetadata(
            document_id="doc-2",
            index_name="idx",
            filename="doc.txt",
            page_count=5,
            word_count=1200,
            custom_metadata={"dept": "legal"},
        )
        assert meta.page_count == 5
        assert meta.word_count == 1200
        assert meta.custom_metadata == {"dept": "legal"}


class TestChunk:
    def test_basic_chunk(self):
        chunk = Chunk(text="some chunk text", start_offset=0, end_offset=15)
        assert chunk.text == "some chunk text"
        assert chunk.heading is None
        assert chunk.metadata == {}

    def test_with_heading_and_metadata(self):
        chunk = Chunk(
            text="chunk",
            heading="Section 1",
            start_offset=100,
            end_offset=200,
            metadata={"page": "3"},
        )
        assert chunk.heading == "Section 1"
        assert chunk.metadata == {"page": "3"}


class TestSearchResult:
    def test_basic_result(self):
        result = SearchResult(
            chunk_id="c-1",
            document_id="d-1",
            text="relevant text",
            score=0.85,
        )
        assert result.chunk_id == "c-1"
        assert result.score == 0.85
        assert result.metadata == {}

    def test_with_metadata(self):
        result = SearchResult(
            chunk_id="c-2",
            document_id="d-2",
            text="text",
            score=0.9,
            metadata={"dept": "hr"},
        )
        assert result.metadata == {"dept": "hr"}


class TestIngestJob:
    def test_queued_job(self):
        job = IngestJob(job_id="j-1", status="queued")
        assert job.job_id == "j-1"
        assert job.status == "queued"
        assert job.document_id is None
        assert job.progress == 0.0
        assert job.error is None

    def test_completed_job(self):
        job = IngestJob(
            job_id="j-2",
            status="completed",
            document_id="doc-1",
            progress=1.0,
        )
        assert job.status == "completed"
        assert job.document_id == "doc-1"
        assert job.progress == 1.0

    def test_failed_job(self):
        job = IngestJob(
            job_id="j-3",
            status="failed",
            error="Embedding API timeout",
        )
        assert job.status == "failed"
        assert job.error == "Embedding API timeout"


class TestIngestedDocument:
    def test_basic(self):
        doc = IngestedDocument(
            document_id="doc-1",
            chunk_count=25,
            index_name="my-index",
        )
        assert doc.document_id == "doc-1"
        assert doc.chunk_count == 25
        assert doc.index_name == "my-index"
