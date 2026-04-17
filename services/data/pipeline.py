"""
Document ingestion pipeline for the Data Service.

Orchestrates format detection, parsing, chunking, embedding, and storage.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 5.5: Format detection and parser routing
  - Listing 5.13: Document ingestion
"""

import hashlib
from typing import Dict, Optional

from services.data.chunking import (
    ChunkingStrategy,
    FixedSizeChunking,
    RecursiveChunking,
    StructureAwareChunking,
)
from services.data.embedding import EmbeddingGenerator
from services.data.models import (
    ExtractedDocument,
    Index,
    IngestedDocument,
)
from services.data.parsers import (
    DocumentParser,
    MarkdownParser,
    PlainTextParser,
    detect_format,
)
from services.data.store import VectorStore


class IngestionPipeline:
    """Orchestrates document ingestion (Listings 5.5, 5.13).

    Not an ABC -- this is the concrete pipeline that holds parser and
    chunking strategy registries and coordinates the full ingest flow.
    """

    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        vector_store: VectorStore,
    ):
        self._embedding_generator = embedding_generator
        self._vector_store = vector_store

        self.parsers: Dict[str, DocumentParser] = {
            "text": PlainTextParser(),
            "markdown": MarkdownParser(),
        }

        self.chunking_strategies: Dict[str, ChunkingStrategy] = {
            "fixed": FixedSizeChunking(),
            "recursive": RecursiveChunking(),
            "structure_aware": StructureAwareChunking(),
        }

    def register_parser(self, format_name: str, parser: DocumentParser) -> None:
        self.parsers[format_name] = parser

    def register_chunking_strategy(self, name: str, strategy: ChunkingStrategy) -> None:
        self.chunking_strategies[name] = strategy

    def detect_format(self, filename: str, file_bytes: bytes) -> str:
        return detect_format(filename, file_bytes)

    def extract(self, filename: str, file_bytes: bytes) -> ExtractedDocument:
        fmt = self.detect_format(filename, file_bytes)
        parser = self.parsers.get(fmt)
        if parser is None:
            parser = self.parsers["text"]
        return parser.parse(file_bytes, filename)

    def ingest_document(
        self,
        index: Index,
        filename: str,
        file_bytes: bytes,
        metadata: Dict[str, str],
        document_id: Optional[str] = None,
    ) -> IngestedDocument:
        """Full ingest: detect -> parse -> chunk -> embed -> store (Listing 5.13)."""
        if document_id is None:
            document_id = hashlib.sha256(filename.encode()).hexdigest()[:16]

        self._vector_store.delete_by_document(index.name, document_id)

        extracted = self.extract(filename, file_bytes)

        strategy_name = index.config.chunking_strategy
        strategy = self.chunking_strategies.get(strategy_name)
        if strategy is None:
            strategy = FixedSizeChunking(
                chunk_size=index.config.chunk_size,
                chunk_overlap=index.config.chunk_overlap,
            )
        elif isinstance(strategy, (FixedSizeChunking, RecursiveChunking, StructureAwareChunking)):
            strategy = type(strategy)(
                chunk_size=index.config.chunk_size,
                chunk_overlap=index.config.chunk_overlap,
            )

        chunks = strategy.chunk(extracted)

        if not chunks:
            return IngestedDocument(document_id=document_id, chunk_count=0, index_name=index.name)

        embeddings = self._embedding_generator.embed_chunks(
            chunks, model=index.config.embedding_model
        )

        self._vector_store.insert(
            index_name=index.name,
            document_id=document_id,
            chunks=chunks,
            embeddings=embeddings,
            metadata=metadata,
        )

        return IngestedDocument(
            document_id=document_id,
            chunk_count=len(chunks),
            index_name=index.name,
        )
