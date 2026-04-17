"""Tests for Data Service gRPC servicer (Listings 5.2, 5.14, 5.15, 5.24)."""

import time
from unittest.mock import MagicMock

from proto import data_pb2
from services.data.service import DataService
from services.data.store import InMemoryVectorStore


def _fake_embed(texts, model):
    return [[float(len(t) % 10)] * 4 for t in texts]


def _make_service() -> DataService:
    return DataService(
        vector_store=InMemoryVectorStore(),
        embed_fn=_fake_embed,
    )


def _make_context():
    ctx = MagicMock()
    ctx.invocation_metadata.return_value = [("x-target-service", "data")]
    return ctx


class TestCreateIndex:
    def test_create_and_get(self):
        svc = _make_service()
        ctx = _make_context()
        config = data_pb2.IndexConfig(
            name="test-idx",
            embedding_model="fake-model",
            embedding_dimensions=4,
            chunking_strategy="fixed",
            chunk_size=10,
            chunk_overlap=0,
        )
        resp = svc.CreateIndex(
            data_pb2.CreateIndexRequest(config=config, owner="me"), ctx
        )
        assert resp.name == "test-idx"
        assert resp.owner == "me"

        get_resp = svc.GetIndex(data_pb2.GetIndexRequest(index_name="test-idx"), ctx)
        assert get_resp.name == "test-idx"

    def test_create_duplicate(self):
        svc = _make_service()
        ctx = _make_context()
        config = data_pb2.IndexConfig(name="test-idx")
        svc.CreateIndex(data_pb2.CreateIndexRequest(config=config), ctx)

        ctx2 = _make_context()
        svc.CreateIndex(data_pb2.CreateIndexRequest(config=config), ctx2)
        ctx2.set_code.assert_called()

    def test_get_nonexistent(self):
        svc = _make_service()
        ctx = _make_context()
        svc.GetIndex(data_pb2.GetIndexRequest(index_name="no-such"), ctx)
        ctx.set_code.assert_called()


class TestListIndexes:
    def test_empty(self):
        svc = _make_service()
        ctx = _make_context()
        resp = svc.ListIndexes(data_pb2.ListIndexesRequest(), ctx)
        assert len(resp.indexes) == 0

    def test_with_indexes(self):
        svc = _make_service()
        ctx = _make_context()
        for name in ["idx-a", "idx-b"]:
            config = data_pb2.IndexConfig(name=name)
            svc.CreateIndex(data_pb2.CreateIndexRequest(config=config), ctx)
        resp = svc.ListIndexes(data_pb2.ListIndexesRequest(), ctx)
        assert len(resp.indexes) == 2


class TestDeleteIndex:
    def test_delete(self):
        svc = _make_service()
        ctx = _make_context()
        config = data_pb2.IndexConfig(name="test-idx")
        svc.CreateIndex(data_pb2.CreateIndexRequest(config=config), ctx)
        resp = svc.DeleteIndex(data_pb2.DeleteIndexRequest(index_name="test-idx"), ctx)
        assert resp.success is True

    def test_delete_nonexistent(self):
        svc = _make_service()
        ctx = _make_context()
        svc.DeleteIndex(data_pb2.DeleteIndexRequest(index_name="no-such"), ctx)
        ctx.set_code.assert_called()


class TestIngestDocument:
    def test_ingest_and_poll(self):
        svc = _make_service()
        ctx = _make_context()
        config = data_pb2.IndexConfig(
            name="test-idx", chunk_size=10, chunk_overlap=0,
            embedding_model="fake-model", embedding_dimensions=4,
        )
        svc.CreateIndex(data_pb2.CreateIndexRequest(config=config), ctx)

        resp = svc.IngestDocument(
            data_pb2.IngestDocumentRequest(
                index_name="test-idx",
                filename="test.txt",
                content=b"Hello world. This is a test document.",
                metadata={"dept": "eng"},
            ),
            ctx,
        )
        assert resp.job_id != ""
        assert resp.status in ("queued", "processing", "completed")

        # Wait for async processing
        for _ in range(50):
            job_resp = svc.GetIngestJob(
                data_pb2.GetIngestJobRequest(job_id=resp.job_id), ctx
            )
            if job_resp.status in ("completed", "failed"):
                break
            time.sleep(0.1)

        assert job_resp.status == "completed"
        assert job_resp.document_id != ""

    def test_ingest_nonexistent_index(self):
        svc = _make_service()
        ctx = _make_context()
        svc.IngestDocument(
            data_pb2.IngestDocumentRequest(
                index_name="no-such",
                filename="test.txt",
                content=b"test",
            ),
            ctx,
        )
        ctx.set_code.assert_called()


class TestDocumentManagement:
    def test_list_and_get_documents(self):
        svc = _make_service()
        ctx = _make_context()
        config = data_pb2.IndexConfig(
            name="test-idx", chunk_size=10, chunk_overlap=0,
            embedding_model="fake-model", embedding_dimensions=4,
        )
        svc.CreateIndex(data_pb2.CreateIndexRequest(config=config), ctx)

        resp = svc.IngestDocument(
            data_pb2.IngestDocumentRequest(
                index_name="test-idx",
                filename="test.txt",
                content=b"Hello world.",
                document_id="doc-1",
            ),
            ctx,
        )
        for _ in range(50):
            job = svc.GetIngestJob(
                data_pb2.GetIngestJobRequest(job_id=resp.job_id), ctx
            )
            if job.status in ("completed", "failed"):
                break
            time.sleep(0.1)

        docs = svc.ListDocuments(
            data_pb2.ListDocumentsRequest(index_name="test-idx"), ctx
        )
        assert len(docs.documents) >= 1

        doc = svc.GetDocument(
            data_pb2.GetDocumentRequest(
                index_name="test-idx", document_id="doc-1"
            ),
            ctx,
        )
        assert doc.document_id == "doc-1"

    def test_delete_document(self):
        svc = _make_service()
        ctx = _make_context()
        config = data_pb2.IndexConfig(
            name="test-idx", chunk_size=10, chunk_overlap=0,
            embedding_model="fake-model", embedding_dimensions=4,
        )
        svc.CreateIndex(data_pb2.CreateIndexRequest(config=config), ctx)

        resp = svc.IngestDocument(
            data_pb2.IngestDocumentRequest(
                index_name="test-idx",
                filename="test.txt",
                content=b"content",
                document_id="doc-1",
            ),
            ctx,
        )
        for _ in range(50):
            job = svc.GetIngestJob(
                data_pb2.GetIngestJobRequest(job_id=resp.job_id), ctx
            )
            if job.status in ("completed", "failed"):
                break
            time.sleep(0.1)

        del_resp = svc.DeleteDocument(
            data_pb2.DeleteDocumentRequest(
                index_name="test-idx", document_id="doc-1"
            ),
            ctx,
        )
        assert del_resp.success is True


class TestSearch:
    def _setup_index_with_docs(self, svc, ctx):
        config = data_pb2.IndexConfig(
            name="test-idx", chunk_size=10, chunk_overlap=0,
            embedding_model="fake-model", embedding_dimensions=4,
        )
        svc.CreateIndex(data_pb2.CreateIndexRequest(config=config), ctx)

        resp = svc.IngestDocument(
            data_pb2.IngestDocumentRequest(
                index_name="test-idx",
                filename="test.txt",
                content=b"Hello world. This is some test content.",
                document_id="doc-1",
            ),
            ctx,
        )
        for _ in range(50):
            job = svc.GetIngestJob(
                data_pb2.GetIngestJobRequest(job_id=resp.job_id), ctx
            )
            if job.status in ("completed", "failed"):
                break
            time.sleep(0.1)

    def test_search(self):
        svc = _make_service()
        ctx = _make_context()
        self._setup_index_with_docs(svc, ctx)

        resp = svc.Search(
            data_pb2.SearchRequest(
                index_name="test-idx", query="hello", top_k=5
            ),
            ctx,
        )
        assert len(resp.results) >= 1

    def test_hybrid_search(self):
        svc = _make_service()
        ctx = _make_context()
        self._setup_index_with_docs(svc, ctx)

        resp = svc.HybridSearch(
            data_pb2.HybridSearchRequest(
                index_name="test-idx", query="hello", top_k=5
            ),
            ctx,
        )
        assert len(resp.results) >= 1

    def test_search_nonexistent_index(self):
        svc = _make_service()
        ctx = _make_context()
        svc.Search(
            data_pb2.SearchRequest(
                index_name="no-such", query="test", top_k=5
            ),
            ctx,
        )
        ctx.set_code.assert_called()


class TestDynamicCodeLoading:
    def test_register_parser(self):
        svc = _make_service()
        ctx = _make_context()
        source_code = b"""
from services.data.parsers import DocumentParser
from services.data.models import ExtractedDocument, DocumentSection

class TestParser(DocumentParser):
    def parse(self, file_bytes, filename):
        return ExtractedDocument(sections=[DocumentSection(content="custom parsed")])
"""
        resp = svc.RegisterParser(
            data_pb2.RegisterParserRequest(
                format="custom",
                class_name="TestParser",
                version="1.0.0",
                source_code=source_code,
            ),
            ctx,
        )
        assert resp.success is True
        assert "custom" in svc._pipeline.parsers

    def test_register_chunking_strategy(self):
        svc = _make_service()
        ctx = _make_context()
        source_code = b"""
from services.data.chunking import ChunkingStrategy
from services.data.models import Chunk

class TestStrategy(ChunkingStrategy):
    def chunk(self, document):
        return [Chunk(text="custom chunk", start_offset=0, end_offset=12)]
"""
        resp = svc.RegisterChunkingStrategy(
            data_pb2.RegisterChunkingStrategyRequest(
                name="custom",
                class_name="TestStrategy",
                version="1.0.0",
                source_code=source_code,
            ),
            ctx,
        )
        assert resp.success is True
        assert "custom" in svc._pipeline.chunking_strategies

    def test_register_invalid_class(self):
        svc = _make_service()
        ctx = _make_context()
        source_code = b"""
class NotAParser:
    pass
"""
        resp = svc.RegisterParser(
            data_pb2.RegisterParserRequest(
                format="bad",
                class_name="NotAParser",
                version="1.0.0",
                source_code=source_code,
            ),
            ctx,
        )
        assert resp.success is False
