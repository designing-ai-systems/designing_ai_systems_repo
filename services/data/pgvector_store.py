"""
PostgreSQL + pgvector implementation of VectorStore.

Covers chunks + search (per the book) and also owns durable index, document,
and job state — see `chapters/book_discrepancies_chapter5.md` for the ABC
extension rationale.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 5.18: PostgreSQL schema for vector storage
  - Listing 5.19: PgvectorStore search implementation
"""

import json
import os
import threading
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import psycopg2
import psycopg2.errors
from psycopg2.extras import RealDictCursor

from services.data.models import (
    Chunk,
    ClaimedJob,
    DocumentMetadata,
    Index,
    IndexConfig,
    IngestJob,
    JobPayload,
    SearchResult,
)
from services.data.store import VectorStore


class PgvectorStore(VectorStore):
    """PostgreSQL + pgvector backend (Listing 5.19)."""

    def __init__(self, connection_string: Optional[str] = None):
        if not connection_string:
            connection_string = os.getenv(
                "DB_CONNECTION_STRING",
                "postgresql://localhost/genai_platform",
            )
        self.conn = psycopg2.connect(connection_string, cursor_factory=RealDictCursor)
        # psycopg2 connections are shareable across threads but queries on the
        # same connection must be serialized. The DataService worker pool calls
        # into this store from multiple threads; this lock is the guard.
        self._lock = threading.RLock()
        self._create_tables()

    @contextmanager
    def _txn(self):
        """Lock + auto-commit/rollback around a cursor block."""
        with self._lock:
            try:
                with self.conn.cursor() as cur:
                    yield cur
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    def _create_tables(self):
        schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
        with open(schema_path) as f:
            sql = f.read()
        with self._txn() as cur:
            cur.execute(sql)

    # ------------------------------------------------------------------ chunks

    def insert(
        self,
        index_name: str,
        document_id: str,
        chunks: List[Chunk],
        embeddings: List[List[float]],
        metadata: Dict[str, str],
    ) -> int:
        with self._txn() as cur:
            for chunk, embedding in zip(chunks, embeddings):
                chunk_id = str(uuid.uuid4())
                cur.execute(
                    """
                    INSERT INTO chunks
                        (chunk_id, document_id, index_name, chunk_text, embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s::vector, %s::jsonb)
                    """,
                    (
                        chunk_id,
                        document_id,
                        index_name,
                        chunk.text,
                        str(embedding),
                        json.dumps(metadata),
                    ),
                )
        return len(chunks)

    def delete_by_document(self, index_name: str, document_id: str) -> int:
        with self._txn() as cur:
            cur.execute(
                "DELETE FROM chunks WHERE index_name = %s AND document_id = %s",
                (index_name, document_id),
            )
            count = cur.rowcount
            cur.execute(
                "DELETE FROM documents WHERE index_name = %s AND document_id = %s",
                (index_name, document_id),
            )
        return count

    def delete_index(self, index_name: str) -> int:
        with self._txn() as cur:
            cur.execute("DELETE FROM chunks WHERE index_name = %s", (index_name,))
            count = cur.rowcount
            cur.execute("DELETE FROM documents WHERE index_name = %s", (index_name,))
            cur.execute("DELETE FROM ingest_jobs WHERE index_name = %s", (index_name,))
            cur.execute("DELETE FROM data_indexes WHERE name = %s", (index_name,))
        return count

    def search(
        self,
        index_name: str,
        query_embedding: List[float],
        top_k: int = 5,
        metadata_filters: Optional[Dict[str, str]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        query = """
            SELECT chunk_id, document_id, chunk_text, metadata,
                   1 - (embedding <=> %s::vector) AS score
            FROM chunks
            WHERE index_name = %s
        """
        params: list = [str(query_embedding), index_name]

        if metadata_filters:
            for key, value in metadata_filters.items():
                query += " AND metadata->>%s = %s"
                params.extend([key, value])

        if score_threshold:
            query += " AND 1 - (embedding <=> %s::vector) >= %s"
            params.extend([str(query_embedding), score_threshold])

        query += " ORDER BY embedding <=> %s::vector LIMIT %s"
        params.extend([str(query_embedding), top_k])

        with self._txn() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

        return [
            SearchResult(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                text=row["chunk_text"],
                metadata=(
                    row["metadata"]
                    if isinstance(row["metadata"], dict)
                    else json.loads(row["metadata"])
                ),
                score=float(row["score"]),
            )
            for row in rows
        ]

    def keyword_search(
        self,
        index_name: str,
        query: str,
        top_k: int = 5,
        metadata_filters: Optional[Dict[str, str]] = None,
    ) -> List[SearchResult]:
        sql = """
            SELECT chunk_id, document_id, chunk_text, metadata,
                   ts_rank(search_vector, plainto_tsquery('english', %s)) AS score
            FROM chunks
            WHERE index_name = %s
              AND search_vector @@ plainto_tsquery('english', %s)
        """
        params: list = [query, index_name, query]

        if metadata_filters:
            for key, value in metadata_filters.items():
                sql += " AND metadata->>%s = %s"
                params.extend([key, value])

        sql += " ORDER BY score DESC LIMIT %s"
        params.append(top_k)

        with self._txn() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        return [
            SearchResult(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                text=row["chunk_text"],
                metadata=(
                    row["metadata"]
                    if isinstance(row["metadata"], dict)
                    else json.loads(row["metadata"])
                ),
                score=float(row["score"]),
            )
            for row in rows
        ]

    # ------------------------------------------------------------------ indexes

    def create_index(self, index: Index) -> None:
        config_payload = asdict(index.config)
        try:
            with self._txn() as cur:
                cur.execute(
                    """
                    INSERT INTO data_indexes
                        (name, config, owner, document_count, total_chunks,
                         created_at, last_ingested_at)
                    VALUES (%s, %s::jsonb, %s, %s, %s, %s, %s)
                    """,
                    (
                        index.name,
                        json.dumps(config_payload),
                        index.owner or "",
                        index.document_count,
                        index.total_chunks,
                        index.created_at,
                        index.last_ingested_at,
                    ),
                )
        except psycopg2.errors.UniqueViolation as e:
            raise ValueError(f"Index '{index.name}' already exists") from e

    def get_index(self, name: str) -> Optional[Index]:
        with self._txn() as cur:
            cur.execute(
                "SELECT name, config, owner, document_count, total_chunks, "
                "created_at, last_ingested_at FROM data_indexes WHERE name = %s",
                (name,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return _row_to_index(row)

    def list_indexes(self) -> List[Index]:
        with self._txn() as cur:
            cur.execute(
                "SELECT name, config, owner, document_count, total_chunks, "
                "created_at, last_ingested_at FROM data_indexes ORDER BY created_at"
            )
            rows = cur.fetchall()
        return [_row_to_index(row) for row in rows]

    def increment_index_stats(
        self,
        name: str,
        documents_delta: int,
        chunks_delta: int,
        last_ingested_at: datetime,
    ) -> None:
        # Single UPDATE — no read-modify-write race between concurrent workers.
        with self._txn() as cur:
            cur.execute(
                """
                UPDATE data_indexes
                   SET document_count = document_count + %s,
                       total_chunks = total_chunks + %s,
                       last_ingested_at = %s
                 WHERE name = %s
                """,
                (documents_delta, chunks_delta, last_ingested_at, name),
            )

    # ----------------------------------------------------------------- documents

    def put_document(self, doc: DocumentMetadata) -> None:
        with self._txn() as cur:
            cur.execute(
                """
                INSERT INTO documents
                    (document_id, index_name, filename, chunk_count,
                     page_count, word_count, custom_metadata, ingested_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                ON CONFLICT (index_name, document_id) DO UPDATE SET
                    filename = EXCLUDED.filename,
                    chunk_count = EXCLUDED.chunk_count,
                    page_count = EXCLUDED.page_count,
                    word_count = EXCLUDED.word_count,
                    custom_metadata = EXCLUDED.custom_metadata,
                    ingested_at = EXCLUDED.ingested_at
                """,
                (
                    doc.document_id,
                    doc.index_name,
                    doc.filename,
                    doc.chunk_count,
                    doc.page_count,
                    doc.word_count,
                    json.dumps(doc.custom_metadata or {}),
                    doc.ingested_at,
                ),
            )

    def get_document(self, index_name: str, document_id: str) -> Optional[DocumentMetadata]:
        with self._txn() as cur:
            cur.execute(
                "SELECT document_id, index_name, filename, chunk_count, page_count, "
                "word_count, custom_metadata, ingested_at FROM documents "
                "WHERE index_name = %s AND document_id = %s",
                (index_name, document_id),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return _row_to_document(row)

    def list_documents(self, index_name: str) -> List[DocumentMetadata]:
        with self._txn() as cur:
            cur.execute(
                "SELECT document_id, index_name, filename, chunk_count, page_count, "
                "word_count, custom_metadata, ingested_at FROM documents "
                "WHERE index_name = %s ORDER BY ingested_at",
                (index_name,),
            )
            rows = cur.fetchall()
        return [_row_to_document(row) for row in rows]

    # ----------------------------------------------------------------- job queue

    def enqueue_job(self, job: IngestJob, index_name: str, payload: JobPayload) -> None:
        with self._txn() as cur:
            cur.execute(
                """
                INSERT INTO ingest_jobs
                    (job_id, index_name, status, progress, filename, content,
                     caller_metadata, requested_document_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                """,
                (
                    job.job_id,
                    index_name,
                    job.status,
                    job.progress,
                    payload.filename,
                    psycopg2.Binary(payload.content),
                    json.dumps(payload.caller_metadata or {}),
                    payload.requested_document_id,
                ),
            )

    def get_job(self, job_id: str) -> Optional[IngestJob]:
        with self._txn() as cur:
            cur.execute(
                "SELECT job_id, status, document_id, progress, error "
                "FROM ingest_jobs WHERE job_id = %s",
                (job_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return IngestJob(
            job_id=row["job_id"],
            status=row["status"],
            document_id=row["document_id"],
            progress=float(row["progress"]) if row["progress"] is not None else 0.0,
            error=row["error"],
        )

    def update_job(
        self,
        job_id: str,
        *,
        status: Optional[str] = None,
        progress: Optional[float] = None,
        document_id: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        sets: List[str] = []
        params: List[Any] = []
        if status is not None:
            sets.append("status = %s")
            params.append(status)
            if status in ("completed", "failed"):
                sets.append("claimed_by = NULL")
                sets.append("claimed_at = NULL")
        if progress is not None:
            sets.append("progress = %s")
            params.append(progress)
        if document_id is not None:
            sets.append("document_id = %s")
            params.append(document_id)
        if error is not None:
            sets.append("error = %s")
            params.append(error)
        if not sets:
            return
        sets.append("updated_at = CURRENT_TIMESTAMP")
        params.append(job_id)
        sql = f"UPDATE ingest_jobs SET {', '.join(sets)} WHERE job_id = %s"
        with self._txn() as cur:
            cur.execute(sql, params)

    def claim_next_job(self, worker_id: str) -> Optional[ClaimedJob]:
        # FOR UPDATE SKIP LOCKED is the standard durable-queue primitive:
        # concurrent workers on the same table never observe the same row.
        with self._txn() as cur:
            cur.execute(
                """
                UPDATE ingest_jobs
                   SET status = 'processing',
                       claimed_by = %s,
                       claimed_at = CURRENT_TIMESTAMP,
                       attempt_count = attempt_count + 1,
                       updated_at = CURRENT_TIMESTAMP
                 WHERE job_id = (
                     SELECT job_id FROM ingest_jobs
                      WHERE status = 'queued'
                      ORDER BY created_at
                      FOR UPDATE SKIP LOCKED
                      LIMIT 1
                 )
                RETURNING job_id, index_name, status, document_id, progress, error,
                          filename, content, caller_metadata, requested_document_id,
                          attempt_count
                """,
                (worker_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        job = IngestJob(
            job_id=row["job_id"],
            status=row["status"],
            document_id=row["document_id"],
            progress=float(row["progress"]) if row["progress"] is not None else 0.0,
            error=row["error"],
        )
        payload = JobPayload(
            filename=row["filename"],
            content=bytes(row["content"]),
            caller_metadata=(
                row["caller_metadata"]
                if isinstance(row["caller_metadata"], dict)
                else json.loads(row["caller_metadata"] or "{}")
            ),
            requested_document_id=row["requested_document_id"],
        )
        return ClaimedJob(
            job=job,
            index_name=row["index_name"],
            payload=payload,
            attempt_count=int(row["attempt_count"]),
        )

    def release_stale_claims(self, stale_after: timedelta) -> int:
        seconds = int(stale_after.total_seconds())
        with self._txn() as cur:
            cur.execute(
                f"""
                UPDATE ingest_jobs
                   SET status = 'queued',
                       claimed_by = NULL,
                       claimed_at = NULL,
                       updated_at = CURRENT_TIMESTAMP
                 WHERE status = 'processing'
                   AND claimed_at < CURRENT_TIMESTAMP - INTERVAL '{seconds} seconds'
                """,
            )
            count = cur.rowcount
        return count


def _row_to_index(row) -> Index:
    config_dict = row["config"] if isinstance(row["config"], dict) else json.loads(row["config"])
    config = IndexConfig(**config_dict)
    return Index(
        name=row["name"],
        config=config,
        owner=row["owner"] or "",
        document_count=int(row["document_count"] or 0),
        total_chunks=int(row["total_chunks"] or 0),
        created_at=row["created_at"],
        last_ingested_at=row["last_ingested_at"],
    )


def _row_to_document(row) -> DocumentMetadata:
    custom = row["custom_metadata"]
    if isinstance(custom, str):
        custom = json.loads(custom or "{}")
    return DocumentMetadata(
        document_id=row["document_id"],
        index_name=row["index_name"],
        filename=row["filename"],
        ingested_at=row["ingested_at"],
        chunk_count=int(row["chunk_count"] or 0),
        page_count=row["page_count"],
        word_count=row["word_count"],
        custom_metadata=custom or None,
    )
