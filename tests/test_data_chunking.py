"""Tests for Data Service chunking strategies (book Listings 5.9, 5.11)."""

from services.data.chunking import (
    ChunkingStrategy,
    FixedSizeChunking,
    RecursiveChunking,
    StructureAwareChunking,
)
from services.data.models import Chunk, DocumentSection, ExtractedDocument


class TestChunkingStrategyABC:
    def test_is_abstract(self):
        assert "chunk" in ChunkingStrategy.__abstractmethods__


class TestFixedSizeChunking:
    def test_basic_chunking(self):
        strategy = FixedSizeChunking(chunk_size=5, chunk_overlap=0)
        doc = ExtractedDocument(
            sections=[DocumentSection(content="one two three four five six seven eight nine ten")]
        )
        chunks = strategy.chunk(doc)
        assert all(isinstance(c, Chunk) for c in chunks)
        assert len(chunks) >= 2

    def test_chunk_size_respected(self):
        strategy = FixedSizeChunking(chunk_size=3, chunk_overlap=0)
        doc = ExtractedDocument(sections=[DocumentSection(content="a b c d e f g h i")])
        chunks = strategy.chunk(doc)
        for chunk in chunks:
            words = chunk.text.split()
            assert len(words) <= 3

    def test_overlap(self):
        strategy = FixedSizeChunking(chunk_size=4, chunk_overlap=2)
        doc = ExtractedDocument(
            sections=[DocumentSection(content="one two three four five six seven eight")]
        )
        chunks = strategy.chunk(doc)
        assert len(chunks) >= 2
        first_words = set(chunks[0].text.split())
        second_words = set(chunks[1].text.split())
        assert len(first_words & second_words) > 0

    def test_small_document(self):
        strategy = FixedSizeChunking(chunk_size=100, chunk_overlap=10)
        doc = ExtractedDocument(sections=[DocumentSection(content="tiny")])
        chunks = strategy.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].text == "tiny"

    def test_empty_document(self):
        strategy = FixedSizeChunking(chunk_size=10, chunk_overlap=0)
        doc = ExtractedDocument(sections=[])
        chunks = strategy.chunk(doc)
        assert chunks == []

    def test_offsets_set(self):
        strategy = FixedSizeChunking(chunk_size=5, chunk_overlap=0)
        doc = ExtractedDocument(sections=[DocumentSection(content="word " * 20)])
        chunks = strategy.chunk(doc)
        for chunk in chunks:
            assert chunk.start_offset >= 0
            assert chunk.end_offset > chunk.start_offset


class TestRecursiveChunking:
    def test_splits_on_double_newline(self):
        strategy = RecursiveChunking(chunk_size=4, chunk_overlap=0)
        doc = ExtractedDocument(
            sections=[
                DocumentSection(
                    content="para one words here\n\npara two more text\n\npara three final words"
                )
            ]
        )
        chunks = strategy.chunk(doc)
        assert len(chunks) >= 2

    def test_splits_on_single_newline(self):
        strategy = RecursiveChunking(chunk_size=5, chunk_overlap=0)
        doc = ExtractedDocument(
            sections=[DocumentSection(content="line one\nline two\nline three")]
        )
        chunks = strategy.chunk(doc)
        assert len(chunks) >= 2

    def test_falls_back_to_space(self):
        strategy = RecursiveChunking(chunk_size=3, chunk_overlap=0)
        doc = ExtractedDocument(sections=[DocumentSection(content="one two three four five six")])
        chunks = strategy.chunk(doc)
        assert len(chunks) >= 2

    def test_empty_document(self):
        strategy = RecursiveChunking(chunk_size=10, chunk_overlap=0)
        doc = ExtractedDocument(sections=[])
        chunks = strategy.chunk(doc)
        assert chunks == []


class TestStructureAwareChunking:
    def test_preserves_sections(self):
        strategy = StructureAwareChunking(chunk_size=100, chunk_overlap=0)
        doc = ExtractedDocument(
            sections=[
                DocumentSection(content="First section.", heading="Intro", level=1),
                DocumentSection(content="Second section.", heading="Details", level=2),
            ]
        )
        chunks = strategy.chunk(doc)
        assert len(chunks) == 2

    def test_prepends_heading(self):
        strategy = StructureAwareChunking(chunk_size=100, chunk_overlap=0)
        doc = ExtractedDocument(
            sections=[DocumentSection(content="Body text.", heading="Title", level=1)]
        )
        chunks = strategy.chunk(doc)
        assert len(chunks) == 1
        assert "Title" in chunks[0].text
        assert "Body text." in chunks[0].text
        assert chunks[0].heading == "Title"

    def test_oversized_section_splits(self):
        strategy = StructureAwareChunking(chunk_size=5, chunk_overlap=0)
        doc = ExtractedDocument(
            sections=[
                DocumentSection(
                    content="one two three four five six seven eight nine ten",
                    heading="Big Section",
                    level=1,
                )
            ]
        )
        chunks = strategy.chunk(doc)
        assert len(chunks) >= 2

    def test_empty_document(self):
        strategy = StructureAwareChunking(chunk_size=100, chunk_overlap=0)
        doc = ExtractedDocument(sections=[])
        chunks = strategy.chunk(doc)
        assert chunks == []
