"""
Data Service client.

Returns domain dataclasses, never exposing Protocol Buffers to the caller.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 5.25: DataClient SDK wrapper
"""

import inspect
from typing import Dict, List, Optional

from proto import data_pb2, data_pb2_grpc
from services.data.models import (
    DocumentMetadata,
    Index,
    IndexConfig,
    IngestJob,
    SearchResult,
)

from .base import BaseClient


class DataClient(BaseClient):
    """Client for the Data Service (Listing 5.25)."""

    def __init__(self, platform):
        super().__init__(platform, "data")
        self._stub = data_pb2_grpc.DataServiceStub(self._channel)

    # ==================== Index Management ====================

    def create_index(self, config: IndexConfig, owner: str = "") -> Index:
        cfg = data_pb2.IndexConfig(
            name=config.name,
            embedding_model=config.embedding_model,
            embedding_dimensions=config.embedding_dimensions,
            chunking_strategy=config.chunking_strategy,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        if config.metadata_schema:
            for k, v in config.metadata_schema.items():
                cfg.metadata_schema[k] = str(v)

        request = data_pb2.CreateIndexRequest(config=cfg, owner=owner)
        resp = self._stub.CreateIndex(request, metadata=self._metadata)
        return self._proto_to_index(resp)

    def get_index(self, index_name: str) -> Index:
        request = data_pb2.GetIndexRequest(index_name=index_name)
        resp = self._stub.GetIndex(request, metadata=self._metadata)
        return self._proto_to_index(resp)

    def list_indexes(self) -> List[Index]:
        resp = self._stub.ListIndexes(data_pb2.ListIndexesRequest(), metadata=self._metadata)
        return [self._proto_to_index(idx) for idx in resp.indexes]

    def delete_index(self, index_name: str) -> bool:
        request = data_pb2.DeleteIndexRequest(index_name=index_name)
        resp = self._stub.DeleteIndex(request, metadata=self._metadata)
        return resp.success

    # ==================== Ingestion ====================

    def ingest(
        self,
        index_name: str,
        filename: str,
        content: bytes,
        content_type: str = "",
        metadata: Optional[Dict[str, str]] = None,
        document_id: str = "",
    ) -> IngestJob:
        request = data_pb2.IngestDocumentRequest(
            index_name=index_name,
            filename=filename,
            content=content,
            content_type=content_type,
            metadata=metadata or {},
            document_id=document_id,
        )
        resp = self._stub.IngestDocument(request, metadata=self._metadata)
        return self._proto_to_ingest_job(resp)

    def get_ingest_status(self, job_id: str) -> IngestJob:
        request = data_pb2.GetIngestJobRequest(job_id=job_id)
        resp = self._stub.GetIngestJob(request, metadata=self._metadata)
        return self._proto_to_ingest_job(resp)

    # ==================== Document Management ====================

    def list_documents(self, index_name: str) -> List[DocumentMetadata]:
        request = data_pb2.ListDocumentsRequest(index_name=index_name)
        resp = self._stub.ListDocuments(request, metadata=self._metadata)
        return [self._proto_to_doc_meta(d) for d in resp.documents]

    def get_document(self, index_name: str, document_id: str) -> DocumentMetadata:
        request = data_pb2.GetDocumentRequest(index_name=index_name, document_id=document_id)
        resp = self._stub.GetDocument(request, metadata=self._metadata)
        return self._proto_to_doc_meta(resp)

    def delete_document(self, index_name: str, document_id: str) -> bool:
        request = data_pb2.DeleteDocumentRequest(index_name=index_name, document_id=document_id)
        resp = self._stub.DeleteDocument(request, metadata=self._metadata)
        return resp.success

    # ==================== Search ====================

    def search(
        self,
        index_name: str,
        query: str,
        top_k: int = 5,
        metadata_filters: Optional[Dict[str, str]] = None,
        score_threshold: float = 0.0,
    ) -> List[SearchResult]:
        request = data_pb2.SearchRequest(
            index_name=index_name,
            query=query,
            top_k=top_k,
            metadata_filters=metadata_filters or {},
            score_threshold=score_threshold,
        )
        resp = self._stub.Search(request, metadata=self._metadata)
        return [self._proto_to_search_result(r) for r in resp.results]

    def hybrid_search(
        self,
        index_name: str,
        query: str,
        top_k: int = 5,
        metadata_filters: Optional[Dict[str, str]] = None,
        score_threshold: float = 0.0,
    ) -> List[SearchResult]:
        request = data_pb2.HybridSearchRequest(
            index_name=index_name,
            query=query,
            top_k=top_k,
            metadata_filters=metadata_filters or {},
            score_threshold=score_threshold,
        )
        resp = self._stub.HybridSearch(request, metadata=self._metadata)
        return [self._proto_to_search_result(r) for r in resp.results]

    # ==================== Dynamic Code Loading ====================

    def register_parser(self, format: str, parser, version: str = "1.0.0") -> bool:
        """Register a custom parser. Uses inspect to find and upload the source file."""
        source_code, class_name = self._extract_source(parser)
        request = data_pb2.RegisterParserRequest(
            format=format,
            class_name=class_name,
            version=version,
            source_code=source_code,
        )
        resp = self._stub.RegisterParser(request, metadata=self._metadata)
        return resp.success

    def register_chunking_strategy(self, name: str, strategy, version: str = "1.0.0") -> bool:
        """Register a custom chunking strategy. Uses inspect to upload source."""
        source_code, class_name = self._extract_source(strategy)
        request = data_pb2.RegisterChunkingStrategyRequest(
            name=name,
            class_name=class_name,
            version=version,
            source_code=source_code,
        )
        resp = self._stub.RegisterChunkingStrategy(request, metadata=self._metadata)
        return resp.success

    @staticmethod
    def _extract_source(obj) -> tuple[bytes, str]:
        """Extract source file bytes and class name from an object instance."""
        cls = type(obj)
        class_name = cls.__name__
        source_file = inspect.getsourcefile(cls)
        if source_file is None:
            raise ValueError(f"Cannot find source file for {class_name}")
        with open(source_file, "rb") as f:
            source_code = f.read()
        return source_code, class_name

    # ==================== Proto Conversion Helpers ====================

    @staticmethod
    def _proto_to_index(resp) -> Index:
        cfg = resp.config
        config = IndexConfig(
            name=cfg.name,
            embedding_model=cfg.embedding_model,
            embedding_dimensions=cfg.embedding_dimensions,
            chunking_strategy=cfg.chunking_strategy,
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            metadata_schema=dict(cfg.metadata_schema) if cfg.metadata_schema else None,
        )
        return Index(
            name=resp.name,
            config=config,
            owner=resp.owner,
            document_count=resp.document_count,
            total_chunks=resp.total_chunks,
        )

    @staticmethod
    def _proto_to_ingest_job(resp) -> IngestJob:
        return IngestJob(
            job_id=resp.job_id,
            status=resp.status,
            document_id=resp.document_id if resp.document_id else None,
            progress=resp.progress,
            error=resp.error if resp.error else None,
        )

    @staticmethod
    def _proto_to_doc_meta(resp) -> DocumentMetadata:
        return DocumentMetadata(
            document_id=resp.document_id,
            index_name=resp.index_name,
            filename=resp.filename,
            chunk_count=resp.chunk_count,
            page_count=resp.page_count if resp.page_count else None,
            word_count=resp.word_count if resp.word_count else None,
            custom_metadata=dict(resp.custom_metadata) if resp.custom_metadata else None,
        )

    @staticmethod
    def _proto_to_search_result(r) -> SearchResult:
        return SearchResult(
            chunk_id=r.chunk_id,
            document_id=r.document_id,
            text=r.text,
            score=r.score,
            metadata=dict(r.metadata),
        )
