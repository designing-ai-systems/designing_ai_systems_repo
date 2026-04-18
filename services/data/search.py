"""
Search orchestration for the Data Service.

Ties together embedding generation and vector store search.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 5.20: Data Service search orchestration
  - Listing 5.23: Hybrid search orchestration
"""

import logging
from typing import Dict, List, Optional

from services.data.embedding import EmbeddingGenerator
from services.data.models import Index, SearchResult
from services.data.store import VectorStore

logger = logging.getLogger(__name__)


class SearchOrchestrator:
    """Orchestrates vector and hybrid search (Listings 5.20, 5.23)."""

    def __init__(self, embedding_generator: EmbeddingGenerator, vector_store: VectorStore):
        self._embedding_generator = embedding_generator
        self._vector_store = vector_store

    def search(
        self,
        index: Index,
        query: str,
        top_k: int = 5,
        metadata_filters: Optional[Dict[str, str]] = None,
        score_threshold: float = 0.0,
    ) -> List[SearchResult]:
        """Embed query then search the vector store (Listing 5.20)."""
        query_embedding = self._embedding_generator.embed_query(
            query=query, model=index.config.embedding_model
        )
        return self._vector_store.search(
            index_name=index.name,
            query_embedding=query_embedding,
            top_k=top_k,
            metadata_filters=metadata_filters,
            score_threshold=score_threshold if score_threshold > 0 else None,
        )

    def hybrid_search(
        self,
        index: Index,
        query: str,
        top_k: int = 5,
        metadata_filters: Optional[Dict[str, str]] = None,
        score_threshold: float = 0.0,
    ) -> List[SearchResult]:
        """Run vector + keyword search, merge with RRF (Listing 5.23)."""
        query_embedding = self._embedding_generator.embed_query(
            query=query, model=index.config.embedding_model
        )

        vector_results = self._vector_store.search(
            index_name=index.name,
            query_embedding=query_embedding,
            top_k=top_k * 2,
            metadata_filters=metadata_filters,
        )

        keyword_results: List[SearchResult] = []
        try:
            keyword_results = self._vector_store.keyword_search(
                index_name=index.name,
                query=query,
                top_k=top_k * 2,
                metadata_filters=metadata_filters,
            )
        except NotImplementedError:
            logger.debug("Backend does not support keyword search; using vector results only")

        fused = reciprocal_rank_fusion(vector_results, keyword_results)

        if score_threshold > 0:
            fused = [r for r in fused if r.score >= score_threshold]

        return fused[:top_k]


def reciprocal_rank_fusion(
    list_a: List[SearchResult],
    list_b: List[SearchResult],
    k: int = 60,
) -> List[SearchResult]:
    """Merge two ranked lists using Reciprocal Rank Fusion."""
    scores: Dict[str, float] = {}
    result_map: Dict[str, SearchResult] = {}

    for rank, r in enumerate(list_a, start=1):
        scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + 1.0 / (k + rank)
        result_map[r.chunk_id] = r

    for rank, r in enumerate(list_b, start=1):
        scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + 1.0 / (k + rank)
        result_map[r.chunk_id] = r

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [
        SearchResult(
            chunk_id=chunk_id,
            document_id=result_map[chunk_id].document_id,
            text=result_map[chunk_id].text,
            score=score,
            metadata=result_map[chunk_id].metadata,
        )
        for chunk_id, score in ranked
    ]
