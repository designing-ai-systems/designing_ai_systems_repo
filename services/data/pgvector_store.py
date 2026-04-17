"""
PostgreSQL + pgvector implementation of VectorStore.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 5.18: PostgreSQL schema for vector storage
  - Listing 5.19: PgvectorStore search implementation
"""

import json
import os
import uuid
from typing import Dict, List, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

from services.data.models import Chunk, SearchResult
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
        self._create_tables()

    def _create_tables(self):
        schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
        with open(schema_path) as f:
            sql = f.read()
        with self.conn.cursor() as cur:
            cur.execute(sql)
        self.conn.commit()

    def insert(
        self,
        index_name: str,
        document_id: str,
        chunks: List[Chunk],
        embeddings: List[List[float]],
        metadata: Dict[str, str],
    ) -> int:
        with self.conn.cursor() as cur:
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
        self.conn.commit()
        return len(chunks)

    def delete_by_document(self, index_name: str, document_id: str) -> int:
        with self.conn.cursor() as cur:
            cur.execute(
                "DELETE FROM chunks WHERE index_name = %s AND document_id = %s",
                (index_name, document_id),
            )
            count = cur.rowcount
        self.conn.commit()
        return count

    def delete_index(self, index_name: str) -> int:
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE index_name = %s", (index_name,))
            count = cur.rowcount
        self.conn.commit()
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

        with self.conn.cursor() as cur:
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

        with self.conn.cursor() as cur:
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
