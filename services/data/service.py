"""
Data Service - gRPC service implementation.

Thin translation layer: receives proto requests, delegates to domain logic,
formats results as proto responses.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 5.2: Index management operations
  - Listing 5.14: Document management operations
  - Listing 5.15: Asynchronous ingestion interface
"""

import importlib
import importlib.util
import logging
import os
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, Optional

import grpc

from proto import data_pb2, data_pb2_grpc
from services.data.chunking import ChunkingStrategy
from services.data.embedding import EmbeddingGenerator
from services.data.models import (
    DocumentMetadata,
    Index,
    IndexConfig,
    IngestJob,
)
from services.data.parsers import DocumentParser
from services.data.pipeline import IngestionPipeline
from services.data.search import SearchOrchestrator
from services.data.store import VectorStore, create_vector_store
from services.shared.servicer_base import BaseServicer

logger = logging.getLogger(__name__)


class DataService(data_pb2_grpc.DataServiceServicer, BaseServicer):
    """gRPC servicer for the Data Service (Listings 5.2, 5.14, 5.15)."""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embed_fn=None,
    ):
        self._vector_store = vector_store or create_vector_store()

        if embed_fn is None:

            def _placeholder_embed(texts, model):
                return [[0.0] * 1536 for _ in texts]

            embed_fn = _placeholder_embed

        self._embedding_generator = EmbeddingGenerator(embed_fn=embed_fn)
        self._pipeline = IngestionPipeline(
            embedding_generator=self._embedding_generator,
            vector_store=self._vector_store,
        )
        self._search_orchestrator = SearchOrchestrator(
            embedding_generator=self._embedding_generator,
            vector_store=self._vector_store,
        )

        self._indexes: Dict[str, Index] = {}
        self._documents: Dict[str, Dict[str, DocumentMetadata]] = {}
        self._jobs: Dict[str, IngestJob] = {}
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._plugin_dir = tempfile.mkdtemp(prefix="data_plugins_")

    def add_to_server(self, server: grpc.Server):
        data_pb2_grpc.add_DataServiceServicer_to_server(self, server)

    # ==================== Index Management (Listing 5.2) ====================

    def CreateIndex(self, request, context):
        try:
            cfg = request.config
            config = IndexConfig(
                name=cfg.name,
                embedding_model=cfg.embedding_model or "text-embedding-3-small",
                embedding_dimensions=cfg.embedding_dimensions or 1536,
                chunking_strategy=cfg.chunking_strategy or "fixed",
                chunk_size=cfg.chunk_size or 512,
                chunk_overlap=cfg.chunk_overlap or 50,
                metadata_schema=dict(cfg.metadata_schema) if cfg.metadata_schema else None,
            )
            if config.name in self._indexes:
                context.set_code(grpc.StatusCode.ALREADY_EXISTS)
                context.set_details(f"Index '{config.name}' already exists")
                return data_pb2.IndexResponse()

            index = Index(name=config.name, config=config, owner=request.owner)
            self._indexes[config.name] = index
            self._documents[config.name] = {}
            return self._index_to_proto(index)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return data_pb2.IndexResponse()

    def GetIndex(self, request, context):
        index = self._indexes.get(request.index_name)
        if index is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Index '{request.index_name}' not found")
            return data_pb2.IndexResponse()
        return self._index_to_proto(index)

    def ListIndexes(self, request, context):
        protos = [self._index_to_proto(idx) for idx in self._indexes.values()]
        return data_pb2.ListIndexesResponse(indexes=protos)

    def DeleteIndex(self, request, context):
        try:
            index = self._indexes.pop(request.index_name, None)
            if index is None:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Index '{request.index_name}' not found")
                return data_pb2.DeleteIndexResponse(success=False)
            deleted = self._vector_store.delete_index(request.index_name)
            self._documents.pop(request.index_name, None)
            return data_pb2.DeleteIndexResponse(success=True, chunks_deleted=deleted)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return data_pb2.DeleteIndexResponse(success=False)

    # ==================== Ingestion (Listing 5.15) ====================

    def IngestDocument(self, request, context):
        index = self._indexes.get(request.index_name)
        if index is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Index '{request.index_name}' not found")
            return data_pb2.IngestJobResponse()

        job_id = str(uuid.uuid4())
        job = IngestJob(job_id=job_id, status="queued")
        self._jobs[job_id] = job

        self._executor.submit(
            self._run_ingest,
            job_id,
            index,
            request.filename,
            request.content,
            dict(request.metadata),
            request.document_id or None,
        )

        return self._job_to_proto(job)

    def GetIngestJob(self, request, context):
        job = self._jobs.get(request.job_id)
        if job is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Job '{request.job_id}' not found")
            return data_pb2.IngestJobResponse()
        return self._job_to_proto(job)

    def _run_ingest(self, job_id, index, filename, content, metadata, document_id):
        job = self._jobs[job_id]
        try:
            job.status = "processing"
            job.progress = 0.1
            result = self._pipeline.ingest_document(
                index=index,
                filename=filename,
                file_bytes=content,
                metadata=metadata,
                document_id=document_id,
            )
            job.status = "completed"
            job.document_id = result.document_id
            job.progress = 1.0

            index.document_count += 1
            index.total_chunks += result.chunk_count
            index.last_ingested_at = datetime.utcnow()

            doc_meta = DocumentMetadata(
                document_id=result.document_id,
                index_name=index.name,
                filename=filename,
                chunk_count=result.chunk_count,
                word_count=len(content.split()) if isinstance(content, str) else None,
                custom_metadata=metadata if metadata else None,
            )
            if index.name not in self._documents:
                self._documents[index.name] = {}
            self._documents[index.name][result.document_id] = doc_meta

        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            logger.exception("Ingestion failed for job %s", job_id)

    # ==================== Document Management (Listing 5.14) ====================

    def ListDocuments(self, request, context):
        docs = self._documents.get(request.index_name, {})
        protos = [self._doc_meta_to_proto(d) for d in docs.values()]
        return data_pb2.ListDocumentsResponse(documents=protos)

    def GetDocument(self, request, context):
        docs = self._documents.get(request.index_name, {})
        doc = docs.get(request.document_id)
        if doc is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("Document not found")
            return data_pb2.DocumentMetadataResponse()
        return self._doc_meta_to_proto(doc)

    def DeleteDocument(self, request, context):
        try:
            deleted = self._vector_store.delete_by_document(request.index_name, request.document_id)
            docs = self._documents.get(request.index_name, {})
            docs.pop(request.document_id, None)
            return data_pb2.DeleteDocumentResponse(success=True, chunks_deleted=deleted)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return data_pb2.DeleteDocumentResponse(success=False)

    # ==================== Search (Listings 5.20, 5.23) ====================

    def Search(self, request, context):
        index = self._indexes.get(request.index_name)
        if index is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Index '{request.index_name}' not found")
            return data_pb2.SearchResponse()
        try:
            results = self._search_orchestrator.search(
                index=index,
                query=request.query,
                top_k=request.top_k or 5,
                metadata_filters=self._extract_filters(request),
                score_threshold=request.score_threshold,
            )
            return data_pb2.SearchResponse(results=[self._result_to_proto(r) for r in results])
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return data_pb2.SearchResponse()

    def HybridSearch(self, request, context):
        index = self._indexes.get(request.index_name)
        if index is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Index '{request.index_name}' not found")
            return data_pb2.SearchResponse()
        try:
            results = self._search_orchestrator.hybrid_search(
                index=index,
                query=request.query,
                top_k=request.top_k or 5,
                metadata_filters=self._extract_filters(request),
                score_threshold=request.score_threshold,
            )
            return data_pb2.SearchResponse(results=[self._result_to_proto(r) for r in results])
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return data_pb2.SearchResponse()

    # ==================== Dynamic Code Loading ====================

    def RegisterParser(self, request, context):
        try:
            parser = self._load_plugin(request.source_code, request.class_name, DocumentParser)
            self._pipeline.register_parser(request.format, parser)
            return data_pb2.RegisterParserResponse(
                success=True,
                message=f"Parser '{request.class_name}' registered for format '{request.format}'",
            )
        except Exception as e:
            return data_pb2.RegisterParserResponse(success=False, message=str(e))

    def RegisterChunkingStrategy(self, request, context):
        try:
            strategy = self._load_plugin(request.source_code, request.class_name, ChunkingStrategy)
            self._pipeline.register_chunking_strategy(request.name, strategy)
            return data_pb2.RegisterChunkingStrategyResponse(
                success=True,
                message=f"Strategy '{request.class_name}' registered as '{request.name}'",
            )
        except Exception as e:
            return data_pb2.RegisterChunkingStrategyResponse(success=False, message=str(e))

    def _load_plugin(self, source_code: bytes, class_name: str, base_class):
        """Write source to plugin dir, load with importlib, validate, instantiate."""
        module_name = f"plugin_{uuid.uuid4().hex[:8]}"
        plugin_path = os.path.join(self._plugin_dir, f"{module_name}.py")

        with open(plugin_path, "wb") as f:
            f.write(source_code)

        spec = importlib.util.spec_from_file_location(module_name, plugin_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        cls = getattr(module, class_name, None)
        if cls is None:
            raise ValueError(f"Class '{class_name}' not found in uploaded source")
        if not issubclass(cls, base_class):
            raise TypeError(f"'{class_name}' is not a subclass of {base_class.__name__}")

        return cls()

    @staticmethod
    def _extract_filters(request):
        return dict(request.metadata_filters) if request.metadata_filters else None

    # ==================== Proto Conversion Helpers ====================

    def _index_to_proto(self, index: Index) -> data_pb2.IndexResponse:
        config_proto = data_pb2.IndexConfig(
            name=index.config.name,
            embedding_model=index.config.embedding_model,
            embedding_dimensions=index.config.embedding_dimensions,
            chunking_strategy=index.config.chunking_strategy,
            chunk_size=index.config.chunk_size,
            chunk_overlap=index.config.chunk_overlap,
        )
        if index.config.metadata_schema:
            for k, v in index.config.metadata_schema.items():
                config_proto.metadata_schema[k] = str(v)

        return data_pb2.IndexResponse(
            name=index.name,
            config=config_proto,
            owner=index.owner,
            document_count=index.document_count,
            total_chunks=index.total_chunks,
            created_at=index.created_at.isoformat() if index.created_at else "",
            last_ingested_at=index.last_ingested_at.isoformat() if index.last_ingested_at else "",
        )

    def _job_to_proto(self, job: IngestJob) -> data_pb2.IngestJobResponse:
        return data_pb2.IngestJobResponse(
            job_id=job.job_id,
            status=job.status,
            document_id=job.document_id or "",
            progress=job.progress,
            error=job.error or "",
        )

    def _doc_meta_to_proto(self, doc: DocumentMetadata) -> data_pb2.DocumentMetadataResponse:
        return data_pb2.DocumentMetadataResponse(
            document_id=doc.document_id,
            index_name=doc.index_name,
            filename=doc.filename,
            chunk_count=doc.chunk_count,
            page_count=doc.page_count or 0,
            word_count=doc.word_count or 0,
            custom_metadata=doc.custom_metadata or {},
            ingested_at=doc.ingested_at.isoformat() if doc.ingested_at else "",
        )

    @staticmethod
    def _result_to_proto(result) -> data_pb2.SearchResultItem:
        return data_pb2.SearchResultItem(
            chunk_id=result.chunk_id,
            document_id=result.document_id,
            text=result.text,
            score=result.score,
            metadata=result.metadata,
        )
