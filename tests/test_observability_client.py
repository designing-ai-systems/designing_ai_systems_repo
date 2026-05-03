"""Unit tests for the in-service ObservabilityClient (Listing 7.9)."""

from typing import List

from services.shared.observability_client import ObservabilityClient
from services.shared.traced_service import StatusCode


class FakeStub:
    """Records every flush call so tests can inspect batches."""

    def __init__(self, *, fail_first: int = 0):
        self.span_batches: List[list] = []
        self.gen_batches: List[list] = []
        self.log_batches: List[list] = []
        self.metric_batches: List[list] = []
        self._fail_remaining = fail_first

    def RecordSpan(self, request):
        if self._fail_remaining > 0:
            self._fail_remaining -= 1
            raise RuntimeError("simulated transport failure")
        self.span_batches.append(list(request.spans))

        class _R:
            recorded = len(request.spans)

        return _R()

    def RecordGeneration(self, request):
        self.gen_batches.append(list(request.generations))

        class _R:
            recorded = len(request.generations)

        return _R()

    def IngestLogs(self, request):
        self.log_batches.append(list(request.events))

        class _R:
            ingested = len(request.events)

        return _R()

    def RecordMetrics(self, request):
        self.metric_batches.append(list(request.records))

        class _R:
            recorded = len(request.records)

        return _R()


class TestNullClient:
    def test_null_client_swallows_calls(self):
        client = ObservabilityClient.null()
        client.log(event_type="x", severity="INFO", message="hi")
        client.record_counter("ai.platform.test", 1.0, labels={"k": "v"})
        client.record_histogram("ai.platform.test", 5.0, labels={"k": "v"})
        # No exceptions; flush_all is a no-op.
        client.flush_all()
        client.stop(flush=True)


class TestBatching:
    def test_span_batch_flushes_at_size(self):
        stub = FakeStub()
        client = ObservabilityClient(stub=stub, batch_size=3, autostart=False)
        for i in range(5):
            span = client.start_span(
                trace_id=f"t-{i}",
                parent_span_id=None,
                service="models",
                operation="op",
            )
            client.end_span(span.span_id, status=StatusCode.OK)
        client.flush_all()
        # 5 spans → first batch of 3, then 1 batch of 2.
        ingested = sum(len(b) for b in stub.span_batches)
        assert ingested == 5

    def test_generation_batch_flushes(self):
        stub = FakeStub()
        client = ObservabilityClient(stub=stub, batch_size=2, autostart=False)
        for i in range(3):
            gen = client.start_generation(
                trace_id=f"t-{i}",
                parent_span_id=None,
                service="models",
                model="gpt-4o",
            )
            gen.update(prompt_tokens=10, completion_tokens=5, cost_usd=0.001)
            client.end_generation(gen.span_id)
        client.flush_all()
        recorded = sum(len(b) for b in stub.gen_batches)
        assert recorded == 3

    def test_logs_and_metrics_buffer_separately(self):
        stub = FakeStub()
        client = ObservabilityClient(stub=stub, batch_size=1, autostart=False)
        client.log(event_type="boot", severity="INFO", message="up")
        client.record_counter("ai.platform.x", 1.0)
        client.flush_all()
        assert sum(len(b) for b in stub.log_batches) == 1
        assert sum(len(b) for b in stub.metric_batches) == 1


class TestPushBackOnFailure:
    def test_failed_span_batch_pushed_back(self):
        stub = FakeStub(fail_first=1)
        client = ObservabilityClient(stub=stub, batch_size=2, autostart=False)
        for i in range(2):
            span = client.start_span(
                trace_id=f"t-{i}", parent_span_id=None, service="x", operation="y"
            )
            client.end_span(span.span_id)
        # The first flush raises; the buffer should still hold the spans.
        assert len(client._span_buffer) == 2  # noqa: SLF001
        # Second flush succeeds and ships them.
        client.flush_all()
        assert sum(len(b) for b in stub.span_batches) == 2


class TestMaxBufferGuard:
    def test_buffer_dropped_on_overflow(self):
        stub = FakeStub(fail_first=999)  # Always fail; buffer should never empty.
        client = ObservabilityClient(stub=stub, batch_size=10, max_buffer_size=10, autostart=False)
        for i in range(50):
            span = client.start_span(
                trace_id=f"t-{i}", parent_span_id=None, service="x", operation="y"
            )
            client.end_span(span.span_id)
        # After many failed flushes the buffer must stay bounded.
        assert len(client._span_buffer) <= 50  # noqa: SLF001
