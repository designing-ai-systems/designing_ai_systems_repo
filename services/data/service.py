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
import socket
import tempfile
import threading
import uuid
from datetime import datetime, timedelta
from typing import List, Optional

import grpc

from proto import data_pb2, data_pb2_grpc
from services.data.chunking import ChunkingStrategy
from services.data.embedding import EmbeddingGenerator
from services.data.models import (
    ClaimedJob,
    DocumentMetadata,
    Index,
    IndexConfig,
    IngestJob,
    JobPayload,
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
        worker_count: Optional[int] = None,
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

        self._plugin_dir = tempfile.mkdtemp(prefix="data_plugins_")

        # Durable-queue worker pool. State lives in the vector store; workers
        # claim via SELECT ... FOR UPDATE SKIP LOCKED (pgvector) or a thread-
        # safe deque (in-memory). Same semantics either way.
        self._worker_id = f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self._worker_count = (
            worker_count if worker_count is not None else int(os.getenv("DATA_WORKER_COUNT", "4"))
        )
        self._poll_interval = float(os.getenv("DATA_WORKER_POLL_SECONDS", "0.5"))
        self._stale_after = timedelta(minutes=int(os.getenv("DATA_STALE_CLAIM_MINUTES", "5")))
        self._max_attempts = int(os.getenv("DATA_MAX_JOB_ATTEMPTS", "3"))
        self._shutdown = threading.Event()
        self._workers: List[threading.Thread] = []

        if self._worker_count > 0:
            # Reclaim anything the previous process left hanging before any
            # new claims start.
            self._vector_store.release_stale_claims(self._stale_after)
            for i in range(self._worker_count):
                t = threading.Thread(
                    target=self._worker_loop,
                    name=f"data-worker-{i}",
                    daemon=True,
                )
                t.start()
                self._workers.append(t)

    def close(self, timeout: float = 30.0) -> None:
        """Stop the worker pool. Safe to call multiple times."""
        if self._shutdown.is_set():
            return
        self._shutdown.set()
        for t in self._workers:
            t.join(timeout=timeout)

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
            index = Index(name=config.name, config=config, owner=request.owner)
            try:
                self._vector_store.create_index(index)
            except ValueError as e:
                context.set_code(grpc.StatusCode.ALREADY_EXISTS)
                context.set_details(str(e))
                return data_pb2.IndexResponse()
            return self._index_to_proto(index)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return data_pb2.IndexResponse()

    def GetIndex(self, request, context):
        index = self._vector_store.get_index(request.index_name)
        if index is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Index '{request.index_name}' not found")
            return data_pb2.IndexResponse()
        return self._index_to_proto(index)

    def ListIndexes(self, request, context):
        protos = [self._index_to_proto(idx) for idx in self._vector_store.list_indexes()]
        return data_pb2.ListIndexesResponse(indexes=protos)

    def DeleteIndex(self, request, context):
        try:
            index = self._vector_store.get_index(request.index_name)
            if index is None:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Index '{request.index_name}' not found")
                return data_pb2.DeleteIndexResponse(success=False)
            deleted = self._vector_store.delete_index(request.index_name)
            return data_pb2.DeleteIndexResponse(success=True, chunks_deleted=deleted)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return data_pb2.DeleteIndexResponse(success=False)

    # ==================== Ingestion (Listing 5.15) ====================

    def IngestDocument(self, request, context):
        index = self._vector_store.get_index(request.index_name)
        if index is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Index '{request.index_name}' not found")
            return data_pb2.IngestJobResponse()

        job = IngestJob(job_id=str(uuid.uuid4()), status="queued")
        payload = JobPayload(
            filename=request.filename,
            content=request.content,
            caller_metadata=dict(request.metadata),
            requested_document_id=request.document_id or None,
        )
        self._vector_store.enqueue_job(job, request.index_name, payload)

        # Best-effort: if workers are running in this process, poke them so a
        # newly-enqueued job is claimed without waiting for the poll tick.
        # (Tests with workers disabled call _process_claimed_job manually.)
        return self._job_to_proto(job)

    def GetIngestJob(self, request, context):
        job = self._vector_store.get_job(request.job_id)
        if job is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Job '{request.job_id}' not found")
            return data_pb2.IngestJobResponse()
        return self._job_to_proto(job)

    def _worker_loop(self):
        """Claim jobs from the store, process them, retry via reaper on failure."""
        while not self._shutdown.is_set():
            try:
                claimed = self._vector_store.claim_next_job(self._worker_id)
            except Exception:
                logger.exception("claim_next_job failed; backing off")
                self._shutdown.wait(self._poll_interval)
                continue
            if claimed is None:
                self._shutdown.wait(self._poll_interval)
                continue
            self._handle_claimed_job(claimed)

    def _handle_claimed_job(self, claimed: ClaimedJob) -> None:
        if claimed.attempt_count > self._max_attempts:
            self._vector_store.update_job(
                claimed.job.job_id,
                status="failed",
                error=f"Exceeded {self._max_attempts} attempts",
            )
            return
        try:
            self._process_claimed_job(claimed)
        except Exception as e:
            logger.exception("Worker crashed on job %s", claimed.job.job_id)
            self._vector_store.update_job(claimed.job.job_id, status="failed", error=str(e))

    def _process_claimed_job(self, claimed: ClaimedJob) -> None:
        index = self._vector_store.get_index(claimed.index_name)
        if index is None:
            self._vector_store.update_job(
                claimed.job.job_id,
                status="failed",
                error=f"Index '{claimed.index_name}' no longer exists",
            )
            return

        self._vector_store.update_job(claimed.job.job_id, progress=0.1)

        result = self._pipeline.ingest_document(
            index=index,
            filename=claimed.payload.filename,
            file_bytes=claimed.payload.content,
            metadata=claimed.payload.caller_metadata,
            document_id=claimed.payload.requested_document_id,
        )

        self._vector_store.increment_index_stats(
            name=index.name,
            documents_delta=1,
            chunks_delta=result.chunk_count,
            last_ingested_at=datetime.utcnow(),
        )

        self._vector_store.put_document(
            DocumentMetadata(
                document_id=result.document_id,
                index_name=index.name,
                filename=claimed.payload.filename,
                chunk_count=result.chunk_count,
                word_count=None,
                custom_metadata=claimed.payload.caller_metadata or None,
            )
        )

        self._vector_store.update_job(
            claimed.job.job_id,
            status="completed",
            document_id=result.document_id,
            progress=1.0,
        )

    # ==================== Document Management (Listing 5.14) ====================

    def ListDocuments(self, request, context):
        docs = self._vector_store.list_documents(request.index_name)
        protos = [self._doc_meta_to_proto(d) for d in docs]
        return data_pb2.ListDocumentsResponse(documents=protos)

    def GetDocument(self, request, context):
        doc = self._vector_store.get_document(request.index_name, request.document_id)
        if doc is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("Document not found")
            return data_pb2.DocumentMetadataResponse()
        return self._doc_meta_to_proto(doc)

    def DeleteDocument(self, request, context):
        try:
            deleted = self._vector_store.delete_by_document(request.index_name, request.document_id)
            return data_pb2.DeleteDocumentResponse(success=True, chunks_deleted=deleted)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return data_pb2.DeleteDocumentResponse(success=False)

    # ==================== Search (Listings 5.20, 5.23) ====================

    def Search(self, request, context):
        index = self._vector_store.get_index(request.index_name)
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
        index = self._vector_store.get_index(request.index_name)
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
