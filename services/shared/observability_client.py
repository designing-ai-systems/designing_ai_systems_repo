"""
ObservabilityClient — in-service buffered telemetry client.

This client is held by every platform service that participates in
distributed tracing. It batches spans, generations, logs, and metrics in
local memory and flushes them in periodic batches to the Observability
Service over gRPC. Telemetry publication never adds a network round trip
to the user's request path.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 7.3: log_fallback_triggered helper (uses log())
  - Listing 7.4: PlatformMetrics constants (used by record_counter / record_histogram)
  - Listing 7.6: start_span / end_span / start_generation / end_generation hooks
  - Listing 7.9: ObservabilityClient batching and buffering
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from services.shared.traced_service import StatusCode

logger = logging.getLogger(__name__)


@dataclass
class _SpanRecord:
    """Domain mirror of proto Span used internally by the client buffer."""

    trace_id: str
    span_id: str
    parent_span_id: str
    service: str
    operation: str
    start_time_ns: int
    end_time_ns: Optional[int] = None
    status: str = StatusCode.OK.value
    error_message: str = ""
    attributes: Dict[str, str] = field(default_factory=dict)
    numeric_attributes: Dict[str, float] = field(default_factory=dict)


@dataclass
class _GenerationRecord:
    """Buffered generation record (specialised span + LLM-specific fields)."""

    span: _SpanRecord
    model: str
    provider: str = ""
    requested_model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    cache_hit: bool = False
    fallback_used: bool = False
    time_to_first_token_ms: float = 0.0

    @property
    def span_id(self) -> str:
        return self.span.span_id

    def update(self, **kwargs: Any) -> None:
        """Update LLM-specific fields after the provider has responded."""
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)


@dataclass
class _LogRecord:
    event_id: str
    trace_id: str
    span_id: str
    timestamp_ns: int
    service: str
    severity: str
    event_type: str
    message: str
    attributes: Dict[str, str] = field(default_factory=dict)
    numeric_attributes: Dict[str, float] = field(default_factory=dict)
    workflow_id: str = ""
    user_id: str = ""


@dataclass
class _MetricRecord:
    name: str
    type: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp_ns: int = 0


class ObservabilityClient:
    """Buffers spans, generations, logs, and metrics; flushes in batches.

    The client is intentionally tolerant of failures: if a flush RPC raises,
    the batch is pushed back to the front of its buffer and retried on the
    next flush. If the buffer grows past ``max_buffer_size`` the oldest
    entries are dropped to avoid unbounded memory growth (Listing 7.9).

    A sentinel "no-op" client is provided via :py:meth:`null` so platform
    services can opt out of telemetry without code branches.
    """

    def __init__(
        self,
        stub: Optional[Any] = None,
        *,
        service_name: str = "",
        batch_size: int = 100,
        flush_interval_seconds: float = 5.0,
        max_buffer_size: int = 10_000,
        autostart: bool = True,
    ) -> None:
        self._stub = stub
        self.service_name = service_name
        self._batch_size = batch_size
        self._flush_interval = flush_interval_seconds
        self._max_buffer_size = max_buffer_size

        self._span_buffer: List[_SpanRecord] = []
        self._generation_buffer: List[_GenerationRecord] = []
        self._log_buffer: List[_LogRecord] = []
        self._metric_buffer: List[_MetricRecord] = []
        # Pending span/generation lookup for end_*() finalization
        self._pending_spans: Dict[str, _SpanRecord] = {}
        self._pending_generations: Dict[str, _GenerationRecord] = {}

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._flush_thread: Optional[threading.Thread] = None
        self._noop = stub is None
        if autostart and not self._noop:
            self.start()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @classmethod
    def null(cls) -> "ObservabilityClient":
        """Return a no-op client. Calls succeed but no data is shipped."""
        return cls(stub=None, autostart=False)

    def start(self) -> None:
        """Start the periodic flush thread (idempotent)."""
        if self._noop or self._flush_thread is not None:
            return
        self._stop_event.clear()
        self._flush_thread = threading.Thread(
            target=self._periodic_flush_loop,
            name="observability-flush",
            daemon=True,
        )
        self._flush_thread.start()

    def stop(self, *, flush: bool = True) -> None:
        """Stop the periodic flush thread and (optionally) flush remaining data."""
        self._stop_event.set()
        if self._flush_thread is not None:
            self._flush_thread.join(timeout=self._flush_interval + 1.0)
            self._flush_thread = None
        if flush and not self._noop:
            self.flush_all()

    # ------------------------------------------------------------------
    # Span lifecycle (called by TracedService.trace_operation)
    # ------------------------------------------------------------------
    def start_span(
        self,
        *,
        trace_id: str,
        parent_span_id: Optional[str],
        service: str,
        operation: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> _SpanRecord:
        span = _SpanRecord(
            trace_id=trace_id,
            span_id=uuid.uuid4().hex,
            parent_span_id=parent_span_id or "",
            service=service,
            operation=operation,
            start_time_ns=time.time_ns(),
            attributes={k: str(v) for k, v in (attributes or {}).items() if v is not None},
        )
        with self._lock:
            self._pending_spans[span.span_id] = span
        return span

    def end_span(
        self,
        span_id: str,
        *,
        status: StatusCode = StatusCode.OK,
        error_message: str = "",
        attributes: Optional[Dict[str, Any]] = None,
        numeric_attributes: Optional[Dict[str, float]] = None,
    ) -> None:
        with self._lock:
            span = self._pending_spans.pop(span_id, None)
            if span is None:
                return
            span.end_time_ns = time.time_ns()
            span.status = status.value if hasattr(status, "value") else str(status)
            span.error_message = error_message or ""
            if attributes:
                span.attributes.update({k: str(v) for k, v in attributes.items() if v is not None})
            if numeric_attributes:
                span.numeric_attributes.update(numeric_attributes)
            self._span_buffer.append(span)
            should_flush = len(self._span_buffer) >= self._batch_size
        if should_flush:
            self._flush_spans()

    # ------------------------------------------------------------------
    # Generation lifecycle (called by TracedService.trace_generation)
    # ------------------------------------------------------------------
    def start_generation(
        self,
        *,
        trace_id: str,
        parent_span_id: Optional[str],
        service: str,
        model: str,
        **attributes: Any,
    ) -> _GenerationRecord:
        span = _SpanRecord(
            trace_id=trace_id,
            span_id=uuid.uuid4().hex,
            parent_span_id=parent_span_id or "",
            service=service,
            operation=f"{service}.generation",
            start_time_ns=time.time_ns(),
        )
        gen = _GenerationRecord(span=span, model=model)
        gen.update(**attributes)
        with self._lock:
            self._pending_generations[span.span_id] = gen
        return gen

    def end_generation(
        self,
        span_id: str,
        *,
        status: StatusCode = StatusCode.OK,
        error_message: str = "",
        **attributes: Any,
    ) -> None:
        with self._lock:
            gen = self._pending_generations.pop(span_id, None)
            if gen is None:
                return
            gen.span.end_time_ns = time.time_ns()
            gen.span.status = status.value if hasattr(status, "value") else str(status)
            gen.span.error_message = error_message or ""
            gen.update(**attributes)
            self._generation_buffer.append(gen)
            should_flush = len(self._generation_buffer) >= self._batch_size
        if should_flush:
            self._flush_generations()

    # ------------------------------------------------------------------
    # Logs (Listing 7.3)
    # ------------------------------------------------------------------
    def log(
        self,
        *,
        event_type: str,
        severity: str,
        message: str,
        trace_id: str = "",
        span_id: str = "",
        attributes: Optional[Dict[str, Any]] = None,
        numeric_attributes: Optional[Dict[str, float]] = None,
        workflow_id: str = "",
        user_id: str = "",
    ) -> None:
        record = _LogRecord(
            event_id=uuid.uuid4().hex,
            trace_id=trace_id,
            span_id=span_id,
            timestamp_ns=time.time_ns(),
            service=self.service_name,
            severity=severity,
            event_type=event_type,
            message=message,
            attributes={k: str(v) for k, v in (attributes or {}).items() if v is not None},
            numeric_attributes=dict(numeric_attributes or {}),
            workflow_id=workflow_id,
            user_id=user_id,
        )
        with self._lock:
            self._log_buffer.append(record)
            should_flush = len(self._log_buffer) >= self._batch_size
        if should_flush:
            self._flush_logs()

    # ------------------------------------------------------------------
    # Metrics (Listing 7.4 / 7.8)
    # ------------------------------------------------------------------
    def record_counter(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        self._record_metric(name=name, value=float(value), labels=labels or {}, kind="COUNTER")

    def record_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        self._record_metric(name=name, value=float(value), labels=labels or {}, kind="HISTOGRAM")

    def _record_metric(
        self,
        *,
        name: str,
        value: float,
        labels: Dict[str, str],
        kind: str,
    ) -> None:
        rec = _MetricRecord(
            name=name,
            type=kind,
            value=value,
            labels={k: str(v) for k, v in labels.items()},
            timestamp_ns=time.time_ns(),
        )
        with self._lock:
            self._metric_buffer.append(rec)
            should_flush = len(self._metric_buffer) >= self._batch_size
        if should_flush:
            self._flush_metrics()

    # ------------------------------------------------------------------
    # Flush plumbing
    # ------------------------------------------------------------------
    def flush_all(self) -> None:
        if self._noop:
            return
        self._flush_spans()
        self._flush_generations()
        self._flush_logs()
        self._flush_metrics()

    def _periodic_flush_loop(self) -> None:
        while not self._stop_event.wait(self._flush_interval):
            try:
                self.flush_all()
            except Exception:
                logger.exception("Observability periodic flush failed")

    def _flush_spans(self) -> None:
        if self._noop or self._stub is None:
            return
        with self._lock:
            if not self._span_buffer:
                return
            batch = self._span_buffer[: self._batch_size]
            self._span_buffer = self._span_buffer[self._batch_size :]
        try:
            from proto import observability_pb2

            request = observability_pb2.RecordSpanRequest(spans=[_span_to_proto(s) for s in batch])
            self._stub.RecordSpan(request)
        except Exception:
            logger.warning("RecordSpan flush failed; pushing batch back")
            with self._lock:
                if len(self._span_buffer) + len(batch) <= self._max_buffer_size:
                    self._span_buffer = batch + self._span_buffer
                else:
                    keep = max(0, self._max_buffer_size - len(self._span_buffer))
                    self._span_buffer = batch[-keep:] + self._span_buffer

    def _flush_generations(self) -> None:
        if self._noop or self._stub is None:
            return
        with self._lock:
            if not self._generation_buffer:
                return
            batch = self._generation_buffer[: self._batch_size]
            self._generation_buffer = self._generation_buffer[self._batch_size :]
        try:
            from proto import observability_pb2

            request = observability_pb2.RecordGenerationRequest(
                generations=[_generation_to_proto(g) for g in batch]
            )
            self._stub.RecordGeneration(request)
        except Exception:
            logger.warning("RecordGeneration flush failed; pushing batch back")
            with self._lock:
                if len(self._generation_buffer) + len(batch) <= self._max_buffer_size:
                    self._generation_buffer = batch + self._generation_buffer
                else:
                    keep = max(0, self._max_buffer_size - len(self._generation_buffer))
                    self._generation_buffer = batch[-keep:] + self._generation_buffer

    def _flush_logs(self) -> None:
        if self._noop or self._stub is None:
            return
        with self._lock:
            if not self._log_buffer:
                return
            batch = self._log_buffer[: self._batch_size]
            self._log_buffer = self._log_buffer[self._batch_size :]
        try:
            from proto import observability_pb2

            request = observability_pb2.IngestLogsRequest(
                events=[_log_to_proto(record) for record in batch]
            )
            self._stub.IngestLogs(request)
        except Exception:
            logger.warning("IngestLogs flush failed; pushing batch back")
            with self._lock:
                if len(self._log_buffer) + len(batch) <= self._max_buffer_size:
                    self._log_buffer = batch + self._log_buffer
                else:
                    keep = max(0, self._max_buffer_size - len(self._log_buffer))
                    self._log_buffer = batch[-keep:] + self._log_buffer

    def _flush_metrics(self) -> None:
        if self._noop or self._stub is None:
            return
        with self._lock:
            if not self._metric_buffer:
                return
            batch = self._metric_buffer[: self._batch_size]
            self._metric_buffer = self._metric_buffer[self._batch_size :]
        try:
            from proto import observability_pb2

            request = observability_pb2.RecordMetricsRequest(
                records=[_metric_to_proto(record) for record in batch]
            )
            self._stub.RecordMetrics(request)
        except Exception:
            logger.warning("RecordMetrics flush failed; pushing batch back")
            with self._lock:
                if len(self._metric_buffer) + len(batch) <= self._max_buffer_size:
                    self._metric_buffer = batch + self._metric_buffer
                else:
                    keep = max(0, self._max_buffer_size - len(self._metric_buffer))
                    self._metric_buffer = batch[-keep:] + self._metric_buffer


# ----------------------------------------------------------------------
# Internal: domain → proto helpers
# ----------------------------------------------------------------------
def _ts_from_ns(ns: int):
    """Convert nanoseconds since epoch into a Timestamp proto."""
    from google.protobuf.timestamp_pb2 import Timestamp

    ts = Timestamp()
    ts.FromNanoseconds(ns)
    return ts


def _span_to_proto(record: _SpanRecord):
    from proto import observability_pb2

    span = observability_pb2.Span(
        trace_id=record.trace_id,
        span_id=record.span_id,
        parent_span_id=record.parent_span_id,
        service=record.service,
        operation=record.operation,
        status=(
            observability_pb2.OK
            if record.status == StatusCode.OK.value
            else observability_pb2.SPAN_ERROR
        ),
        error_message=record.error_message,
    )
    span.start_time.CopyFrom(_ts_from_ns(record.start_time_ns))
    if record.end_time_ns is not None:
        span.end_time.CopyFrom(_ts_from_ns(record.end_time_ns))
    if record.attributes:
        for k, v in record.attributes.items():
            span.attributes[k] = v
    if record.numeric_attributes:
        for k, v in record.numeric_attributes.items():
            span.numeric_attributes[k] = v
    return span


def _generation_to_proto(record: _GenerationRecord):
    from proto import observability_pb2

    gen = observability_pb2.Generation(
        model=record.model,
        provider=record.provider,
        requested_model=record.requested_model,
        prompt_tokens=record.prompt_tokens,
        completion_tokens=record.completion_tokens,
        cost_usd=record.cost_usd,
        cache_hit=record.cache_hit,
        fallback_used=record.fallback_used,
        time_to_first_token_ms=record.time_to_first_token_ms,
    )
    gen.span.CopyFrom(_span_to_proto(record.span))
    return gen


def _log_to_proto(record: _LogRecord):
    from proto import observability_pb2

    severity_map = {
        "DEBUG": observability_pb2.DEBUG,
        "INFO": observability_pb2.INFO,
        "WARNING": observability_pb2.WARNING,
        "ERROR": observability_pb2.ERROR,
        "CRITICAL": observability_pb2.CRITICAL,
    }
    log = observability_pb2.LogEvent(
        event_id=record.event_id,
        trace_id=record.trace_id,
        span_id=record.span_id,
        service=record.service,
        severity=severity_map.get(record.severity.upper(), observability_pb2.INFO),
        event_type=record.event_type,
        message=record.message,
        workflow_id=record.workflow_id,
        user_id=record.user_id,
    )
    log.timestamp.CopyFrom(_ts_from_ns(record.timestamp_ns))
    for k, v in record.attributes.items():
        log.attributes[k] = v
    for k, v in record.numeric_attributes.items():
        log.numeric_attributes[k] = v
    return log


def _metric_to_proto(record: _MetricRecord):
    from proto import observability_pb2

    type_map = {
        "COUNTER": observability_pb2.COUNTER,
        "HISTOGRAM": observability_pb2.HISTOGRAM,
    }
    metric = observability_pb2.MetricRecord(
        name=record.name,
        type=type_map.get(record.type.upper(), observability_pb2.COUNTER),
        value=record.value,
    )
    metric.timestamp.CopyFrom(_ts_from_ns(record.timestamp_ns))
    for k, v in record.labels.items():
        metric.labels[k] = v
    return metric
