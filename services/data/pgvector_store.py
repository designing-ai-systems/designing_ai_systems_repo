"""
PostgreSQL + pgvector implementation of VectorStore.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 5.18: PostgreSQL schema for vector storage
  - Listing 5.19: PgvectorStore search implementation
"""

import json
import uuid
from typing import Dict, List, Optional

from services.data.models import Chunk, SearchResult
from services.data.store import VectorStore


class PgvectorStore(VectorStore):
    """PostgreSQL + pgvector backend (Listing 5.19)."""

    def __init__(self, connection_string: str):
        import psycopg2

        self.conn = psycopg2.connect(connection_string)

    def insert(
        self,
        index_name: str,
        document_id: str,
        chunks: List[Chunk],
        embeddings: List[List[float]],
        metadata: Dict[str, str],
    ) -> int:
        cursor = self.conn.cursor()
        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = str(uuid.uuid4())
            cursor.execute(
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
        cursor = self.conn.cursor()
        cursor.execute(
            "DELETE FROM chunks WHERE index_name = %s AND document_id = %s",
            (index_name, document_id),
        )
        count = cursor.rowcount
        self.conn.commit()
        return count

    def delete_index(self, index_name: str) -> int:
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM chunks WHERE index_name = %s", (index_name,))
        count = cursor.rowcount
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

        cursor = self.conn.cursor()
        cursor.execute(query, params)

        return [
            SearchResult(
                chunk_id=row[0],
                document_id=row[1],
                text=row[2],
                metadata=row[3] if isinstance(row[3], dict) else json.loads(row[3]),
                score=float(row[4]),
            )
            for row in cursor.fetchall()
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

        cursor = self.conn.cursor()
        cursor.execute(sql, params)

        return [
            SearchResult(
                chunk_id=row[0],
                document_id=row[1],
                text=row[2],
                metadata=row[3] if isinstance(row[3], dict) else json.loads(row[3]),
                score=float(row[4]),
            )
            for row in cursor.fetchall()
        ]
