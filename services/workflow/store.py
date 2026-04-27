"""
Workflow Service registry — stores `WorkflowSpec` records.

Storage abstraction matches the sessions/data pattern (`SessionStorage` ABC +
`InMemorySessionStorage` + `PostgresSessionStorage`). The Postgres backend
is deferred — see `chapters/book_discrepancies_chapter8.md` discrepancy #2.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 8.22: WorkflowSpec
  - Section 8.6: Registry operations
"""

import os
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from services.workflow.models import WorkflowSpec


class WorkflowRegistry(ABC):
    """Abstract registry for workflow specs."""

    @abstractmethod
    def register(self, spec: WorkflowSpec) -> Tuple[str, int]:
        """Register or re-register a workflow. Returns (workflow_id, version)."""

    @abstractmethod
    def get(self, name: str) -> Optional[WorkflowSpec]:
        """Get the latest spec for a workflow, or None if unknown."""

    @abstractmethod
    def list(self) -> List[WorkflowSpec]:
        """List all known workflows."""

    @abstractmethod
    def update(self, spec: WorkflowSpec) -> int:
        """Update a workflow. Raises KeyError if unknown. Returns new version."""

    @abstractmethod
    def delete(self, name: str) -> bool:
        """Delete a workflow. Returns True if it existed."""


class InMemoryWorkflowRegistry(WorkflowRegistry):
    """Dict-backed registry. Adequate for the book demo; loses state on restart."""

    def __init__(self) -> None:
        self._specs: Dict[str, WorkflowSpec] = {}
        self._ids: Dict[str, str] = {}

    def register(self, spec: WorkflowSpec) -> Tuple[str, int]:
        if spec.name in self._specs:
            existing = self._specs[spec.name]
            spec.version = existing.version + 1
        else:
            self._ids[spec.name] = uuid.uuid4().hex
            spec.version = 1
        self._specs[spec.name] = spec
        return self._ids[spec.name], spec.version

    def get(self, name: str) -> Optional[WorkflowSpec]:
        return self._specs.get(name)

    def list(self) -> List[WorkflowSpec]:
        return list(self._specs.values())

    def update(self, spec: WorkflowSpec) -> int:
        if spec.name not in self._specs:
            raise KeyError(f"Workflow '{spec.name}' is not registered")
        spec.version = self._specs[spec.name].version + 1
        self._specs[spec.name] = spec
        return spec.version

    def delete(self, name: str) -> bool:
        if name not in self._specs:
            return False
        del self._specs[name]
        self._ids.pop(name, None)
        return True


def create_registry() -> WorkflowRegistry:
    """Construct a registry based on the WORKFLOW_REGISTRY env var.

    Currently only "memory" is supported; "postgres" is reserved for a
    follow-up that adds `PostgresWorkflowRegistry` against the schema in
    `services/workflow/schema.sql`.
    """
    backend = os.getenv("WORKFLOW_REGISTRY", "memory")
    if backend != "memory":
        raise ValueError(
            f"Unsupported WORKFLOW_REGISTRY backend: {backend!r} "
            "(only 'memory' is implemented; postgres is on the roadmap)"
        )
    return InMemoryWorkflowRegistry()
