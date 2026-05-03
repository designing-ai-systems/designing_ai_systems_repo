"""
TracedService — automatic distributed tracing base class.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 7.6: TracedService base class with span and generation context managers
  - Listing 7.7: Model Service chat() with automatic generation tracking
"""

from __future__ import annotations

import enum
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator, Optional

if TYPE_CHECKING:
    from services.shared.observability_client import ObservabilityClient


class StatusCode(str, enum.Enum):
    """Span status codes (matches the Listing 7.6 sentinel values)."""

    OK = "OK"
    ERROR = "ERROR"


@dataclass
class TraceContext:
    """The trace state propagated through every service hop.

    The API Gateway seeds ``trace_id`` and the workflow's outermost
    ``span_id``; each downstream service derives a child context whose
    ``parent_span_id`` is the caller's ``span_id``.
    """

    trace_id: str
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    workflow_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    tags: list = field(default_factory=list)


class TracedService:
    """Base class providing automatic tracing for all platform services.

    Listing 7.6: services that handle user requests inherit this class
    and use ``trace_operation`` / ``trace_generation`` to wrap every
    unit of work, so the workflow developer who calls them never has
    to create spans manually.
    """

    def __init__(self, service_name: str, observability: "ObservabilityClient"):
        self.service_name = service_name
        self.observability = observability

    @contextmanager
    def trace_operation(
        self,
        operation: str,
        trace_context: TraceContext,
        **attributes: Any,
    ) -> Iterator[TraceContext]:
        """Wrap a unit of work in a span. Yields a child context for nested calls."""
        span = self.observability.start_span(
            trace_id=trace_context.trace_id,
            parent_span_id=trace_context.span_id,
            service=self.service_name,
            operation=f"{self.service_name}.{operation}",
            attributes=attributes,
        )
        child_context = TraceContext(
            trace_id=trace_context.trace_id,
            span_id=span.span_id,
            parent_span_id=trace_context.span_id,
            workflow_id=trace_context.workflow_id,
            user_id=trace_context.user_id,
            session_id=trace_context.session_id,
            tags=list(trace_context.tags),
        )
        try:
            yield child_context
            self.observability.end_span(span.span_id, status=StatusCode.OK)
        except Exception as exc:
            self.observability.end_span(
                span.span_id,
                status=StatusCode.ERROR,
                error_message=str(exc),
            )
            raise

    @contextmanager
    def trace_generation(
        self,
        trace_context: TraceContext,
        model: str,
        **attributes: Any,
    ) -> Iterator[Any]:
        """Wrap an LLM call. Yields the generation handle (a leaf node)."""
        gen = self.observability.start_generation(
            trace_id=trace_context.trace_id,
            parent_span_id=trace_context.span_id,
            service=self.service_name,
            model=model,
            **attributes,
        )
        try:
            yield gen
            self.observability.end_generation(gen.span_id, status=StatusCode.OK)
        except Exception as exc:
            self.observability.end_generation(
                gen.span_id,
                status=StatusCode.ERROR,
                error_message=str(exc),
            )
            raise
