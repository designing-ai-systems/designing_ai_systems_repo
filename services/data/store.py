"""
Vector store abstraction and in-memory implementation for the Data Service.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 5.16: VectorStore ABC write operations
  - Listing 5.17: VectorStore search + SearchResult
  - Listing 5.21: keyword_search (optional, raises NotImplementedError by default)
"""

import math
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

from services.data.models import Chunk, SearchResult


class VectorStore(ABC):
    """Abstract vector storage interface (Listings 5.16, 5.17, 5.21)."""

    @abstractmethod
    def insert(
        self,
        index_name: str,
        document_id: str,
        chunks: List[Chunk],
        embeddings: List[List[float]],
        metadata: Dict[str, str],
    ) -> int:
        """Insert chunks with embeddings. Returns count inserted."""
        pass

    @abstractmethod
    def delete_by_document(self, index_name: str, document_id: str) -> int:
        """Delete all chunks for a document. Returns count deleted."""
        pass

    @abstractmethod
    def delete_index(self, index_name: str) -> int:
        """Delete all data for an index. Returns count deleted."""
        pass

    @abstractmethod
    def search(
        self,
        index_name: str,
        query_embedding: List[float],
        top_k: int = 5,
        metadata_filters: Optional[Dict[str, str]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """Find chunks most similar to the query embedding."""
        pass

    def keyword_search(
        self,
        index_name: str,
        query: str,
        top_k: int = 5,
        metadata_filters: Optional[Dict[str, str]] = None,
    ) -> List[SearchResult]:
        """Find chunks matching the query by keyword (Listing 5.21)."""
        raise NotImplementedError(f"{type(self).__name__} does not support keyword search")


@dataclass
class _StoredChunk:
    chunk_id: str
    index_name: str
    document_id: str
    text: str
    embedding: List[float]
    metadata: Dict[str, str]


class InMemoryVectorStore(VectorStore):
    """In-memory implementation for development and testing."""

    def __init__(self):
        self._chunks: List[_StoredChunk] = []

    def insert(
        self,
        index_name: str,
        document_id: str,
        chunks: List[Chunk],
        embeddings: List[List[float]],
        metadata: Dict[str, str],
    ) -> int:
        for chunk, embedding in zip(chunks, embeddings):
            self._chunks.append(
                _StoredChunk(
                    chunk_id=str(uuid.uuid4()),
                    index_name=index_name,
                    document_id=document_id,
                    text=chunk.text,
                    embedding=embedding,
                    metadata=dict(metadata),
                )
            )
        return len(chunks)

    def delete_by_document(self, index_name: str, document_id: str) -> int:
        before = len(self._chunks)
        self._chunks = [
            c
            for c in self._chunks
            if not (c.index_name == index_name and c.document_id == document_id)
        ]
        return before - len(self._chunks)

    def delete_index(self, index_name: str) -> int:
        before = len(self._chunks)
        self._chunks = [c for c in self._chunks if c.index_name != index_name]
        return before - len(self._chunks)

    def search(
        self,
        index_name: str,
        query_embedding: List[float],
        top_k: int = 5,
        metadata_filters: Optional[Dict[str, str]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        candidates = [c for c in self._chunks if c.index_name == index_name]

        if metadata_filters:
            for key, value in metadata_filters.items():
                candidates = [c for c in candidates if c.metadata.get(key) == value]

        scored = []
        for c in candidates:
            score = _cosine_similarity(query_embedding, c.embedding)
            if score_threshold is not None and score < score_threshold:
                continue
            scored.append((score, c))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            SearchResult(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                text=c.text,
                score=score,
                metadata=c.metadata,
            )
            for score, c in scored[:top_k]
        ]


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def create_vector_store() -> VectorStore:
    """Create vector store based on environment configuration."""
    store_type = os.getenv("VECTOR_STORE", "memory")
    if store_type == "pgvector":
        from services.data.pgvector_store import PgvectorStore

        return PgvectorStore(os.getenv("DB_CONNECTION_STRING", ""))
    return InMemoryVectorStore()
