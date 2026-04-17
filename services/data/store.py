"""
Vector store abstraction and in-memory implementation for the Data Service.

The ABC covers chunks (per Listings 5.16, 5.17, 5.21) plus index, document, and
durable-queue operations that the DataService delegates to. See
`chapters/book_discrepancies_chapter5.md` for the rationale on extending the
ABC beyond the listings.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 5.16: VectorStore ABC write operations
  - Listing 5.17: VectorStore search + SearchResult
  - Listing 5.21: keyword_search (optional, raises NotImplementedError by default)
"""

import collections
import copy
import math
import os
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Deque, Dict, List, Optional

from services.data.models import (
    Chunk,
    ClaimedJob,
    DocumentMetadata,
    Index,
    IngestJob,
    JobPayload,
    SearchResult,
)


class VectorStore(ABC):
    """Storage backend for the Data Service.

    Owns chunks + search (per the book) plus index, document, and job state
    so the DataService has a single durable dependency.
    """

    # ------------------------------------------------------------------ chunks

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

    @abstractmethod
    def delete_by_document(self, index_name: str, document_id: str) -> int:
        """Delete chunks and document metadata for a document. Returns chunks deleted."""

    @abstractmethod
    def delete_index(self, index_name: str) -> int:
        """Cascade-delete chunks, documents, and index metadata. Returns chunks deleted."""

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

    def keyword_search(
        self,
        index_name: str,
        query: str,
        top_k: int = 5,
        metadata_filters: Optional[Dict[str, str]] = None,
    ) -> List[SearchResult]:
        """Find chunks matching the query by keyword (Listing 5.21)."""
        raise NotImplementedError(f"{type(self).__name__} does not support keyword search")

    # ------------------------------------------------------------------ indexes

    @abstractmethod
    def create_index(self, index: Index) -> None:
        """Persist a new index. Raises ValueError if name already exists."""

    @abstractmethod
    def get_index(self, name: str) -> Optional[Index]:
        """Return the index metadata or None if not found."""

    @abstractmethod
    def list_indexes(self) -> List[Index]:
        """Return every index in the store."""

    @abstractmethod
    def update_index_stats(
        self,
        name: str,
        document_count: int,
        total_chunks: int,
        last_ingested_at: datetime,
    ) -> None:
        """Refresh the counters on an existing index."""

    # ----------------------------------------------------------------- documents

    @abstractmethod
    def put_document(self, doc: DocumentMetadata) -> None:
        """Upsert document metadata."""

    @abstractmethod
    def get_document(self, index_name: str, document_id: str) -> Optional[DocumentMetadata]:
        """Return document metadata or None if not found."""

    @abstractmethod
    def list_documents(self, index_name: str) -> List[DocumentMetadata]:
        """Return every document in an index."""

    # ----------------------------------------------------------------- job queue

    @abstractmethod
    def enqueue_job(self, job: IngestJob, index_name: str, payload: JobPayload) -> None:
        """Persist a new job in status='queued' with its payload."""

    @abstractmethod
    def get_job(self, job_id: str) -> Optional[IngestJob]:
        """Return the current job state."""

    @abstractmethod
    def update_job(
        self,
        job_id: str,
        *,
        status: Optional[str] = None,
        progress: Optional[float] = None,
        document_id: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """Partial update on a job row; only fields passed are written."""

    @abstractmethod
    def claim_next_job(self, worker_id: str) -> Optional[ClaimedJob]:
        """Atomically transition the oldest queued job to processing for this worker."""

    @abstractmethod
    def release_stale_claims(self, stale_after: timedelta) -> int:
        """Move jobs stuck in 'processing' older than threshold back to 'queued'. Returns count."""


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
        self._indexes: Dict[str, Index] = {}
        self._documents: Dict[str, Dict[str, DocumentMetadata]] = {}
        self._jobs: Dict[str, IngestJob] = {}
        self._job_payloads: Dict[str, JobPayload] = {}
        self._job_index_names: Dict[str, str] = {}
        self._job_attempts: Dict[str, int] = {}
        self._job_claimed_at: Dict[str, datetime] = {}
        self._queue: Deque[str] = collections.deque()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ chunks

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
        docs = self._documents.get(index_name)
        if docs is not None:
            docs.pop(document_id, None)
        return before - len(self._chunks)

    def delete_index(self, index_name: str) -> int:
        before = len(self._chunks)
        self._chunks = [c for c in self._chunks if c.index_name != index_name]
        self._documents.pop(index_name, None)
        self._indexes.pop(index_name, None)
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

    # ------------------------------------------------------------------ indexes

    def create_index(self, index: Index) -> None:
        with self._lock:
            if index.name in self._indexes:
                raise ValueError(f"Index '{index.name}' already exists")
            self._indexes[index.name] = copy.deepcopy(index)
            self._documents.setdefault(index.name, {})

    def get_index(self, name: str) -> Optional[Index]:
        with self._lock:
            idx = self._indexes.get(name)
            return copy.deepcopy(idx) if idx is not None else None

    def list_indexes(self) -> List[Index]:
        with self._lock:
            return [copy.deepcopy(idx) for idx in self._indexes.values()]

    def update_index_stats(
        self,
        name: str,
        document_count: int,
        total_chunks: int,
        last_ingested_at: datetime,
    ) -> None:
        with self._lock:
            idx = self._indexes.get(name)
            if idx is None:
                return
            idx.document_count = document_count
            idx.total_chunks = total_chunks
            idx.last_ingested_at = last_ingested_at

    # ----------------------------------------------------------------- documents

    def put_document(self, doc: DocumentMetadata) -> None:
        with self._lock:
            self._documents.setdefault(doc.index_name, {})[doc.document_id] = copy.deepcopy(doc)

    def get_document(self, index_name: str, document_id: str) -> Optional[DocumentMetadata]:
        with self._lock:
            doc = self._documents.get(index_name, {}).get(document_id)
            return copy.deepcopy(doc) if doc is not None else None

    def list_documents(self, index_name: str) -> List[DocumentMetadata]:
        with self._lock:
            return [copy.deepcopy(d) for d in self._documents.get(index_name, {}).values()]

    # ----------------------------------------------------------------- job queue

    def enqueue_job(self, job: IngestJob, index_name: str, payload: JobPayload) -> None:
        with self._lock:
            self._jobs[job.job_id] = copy.deepcopy(job)
            self._job_payloads[job.job_id] = copy.deepcopy(payload)
            self._job_index_names[job.job_id] = index_name
            self._job_attempts[job.job_id] = 0
            self._queue.append(job.job_id)

    def get_job(self, job_id: str) -> Optional[IngestJob]:
        with self._lock:
            job = self._jobs.get(job_id)
            return copy.deepcopy(job) if job is not None else None

    def update_job(
        self,
        job_id: str,
        *,
        status: Optional[str] = None,
        progress: Optional[float] = None,
        document_id: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if status is not None:
                job.status = status
                if status in ("completed", "failed"):
                    self._job_claimed_at.pop(job_id, None)
            if progress is not None:
                job.progress = progress
            if document_id is not None:
                job.document_id = document_id
            if error is not None:
                job.error = error

    def claim_next_job(self, worker_id: str) -> Optional[ClaimedJob]:
        with self._lock:
            while self._queue:
                job_id = self._queue.popleft()
                job = self._jobs.get(job_id)
                if job is None or job.status != "queued":
                    continue
                job.status = "processing"
                self._job_attempts[job_id] += 1
                self._job_claimed_at[job_id] = datetime.utcnow()
                return ClaimedJob(
                    job=copy.deepcopy(job),
                    index_name=self._job_index_names[job_id],
                    payload=copy.deepcopy(self._job_payloads[job_id]),
                    attempt_count=self._job_attempts[job_id],
                )
            return None

    def release_stale_claims(self, stale_after: timedelta) -> int:
        cutoff = datetime.utcnow() - stale_after
        released = 0
        with self._lock:
            for job_id, claimed_at in list(self._job_claimed_at.items()):
                job = self._jobs.get(job_id)
                if job is None or job.status != "processing":
                    self._job_claimed_at.pop(job_id, None)
                    continue
                if claimed_at < cutoff:
                    job.status = "queued"
                    self._job_claimed_at.pop(job_id, None)
                    self._queue.append(job_id)
                    released += 1
        return released


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
