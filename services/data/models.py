"""
Data Service domain models.

Python dataclasses representing indexes, documents, chunks, and search results.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 5.1: IndexConfig dataclass
  - Listing 5.3: Index dataclass
  - Listing 5.4: DocumentSection, ExtractedDocument (dataclass portion)
  - Listing 5.7: DocumentMetadata dataclass
  - Listing 5.8: Chunk dataclass
  - Listing 5.15: IngestJob dataclass
  - Listing 5.17: SearchResult dataclass
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class IndexConfig:
    """Configuration for creating a knowledge index (Listing 5.1)."""

    name: str
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    chunking_strategy: str = "fixed"
    chunk_size: int = 512
    chunk_overlap: int = 50
    metadata_schema: Optional[Dict] = None


@dataclass
class Index:
    """A knowledge index with configuration and runtime stats (Listing 5.3)."""

    name: str
    config: IndexConfig
    owner: str = ""
    document_count: int = 0
    total_chunks: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_ingested_at: Optional[datetime] = None


@dataclass
class DocumentSection:
    """A section of an extracted document (Listing 5.4)."""

    content: str
    heading: Optional[str] = None
    level: int = 0
    page_number: Optional[int] = None


@dataclass
class ExtractedDocument:
    """Parsed document with structural information (Listing 5.4)."""

    sections: List[DocumentSection]
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n\n".join(s.content for s in self.sections)


@dataclass
class DocumentMetadata:
    """Metadata tracked for each ingested document (Listing 5.7)."""

    document_id: str
    index_name: str
    filename: str
    ingested_at: datetime = field(default_factory=datetime.utcnow)
    chunk_count: int = 0
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    custom_metadata: Optional[Dict[str, str]] = None


@dataclass
class Chunk:
    """A piece of a document ready for embedding (Listing 5.8)."""

    text: str
    start_offset: int
    end_offset: int
    heading: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class SearchResult:
    """A single result from vector or hybrid search (Listing 5.17)."""

    chunk_id: str
    document_id: str
    text: str
    score: float
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class IngestJob:
    """Tracks asynchronous document ingestion (Listing 5.15)."""

    job_id: str
    status: str  # "queued", "processing", "completed", "failed"
    document_id: Optional[str] = None
    progress: float = 0.0
    error: Optional[str] = None


@dataclass
class JobPayload:
    """Everything the worker needs to execute an ingest job.

    Persisted on the job row so any worker (including a fresh process) can
    claim and run the job without relying on submitter-side memory.
    """

    filename: str
    content: bytes
    caller_metadata: Dict[str, str] = field(default_factory=dict)
    requested_document_id: Optional[str] = None


@dataclass
class ClaimedJob:
    """A job atomically transitioned from queued -> processing by claim_next_job."""

    job: IngestJob
    index_name: str
    payload: JobPayload
    attempt_count: int


@dataclass
class IngestedDocument:
    """Result of a successful document ingestion (Listing 5.13 return type)."""

    document_id: str
    chunk_count: int
    index_name: str
