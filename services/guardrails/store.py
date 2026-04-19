"""
Guardrails Policy Store — storage abstraction for guardrail policies.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 6.21: Input guardrail configuration
  - Listing 6.23: Tiered output violation handlers
  - Listing 6.25: Human approval gate configuration
"""

import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from services.guardrails.models import PolicyDefinition, ViolationRecord


class PolicyStore(ABC):
    """Abstract base class for guardrail policy stores."""

    @abstractmethod
    def add_policy(self, policy: PolicyDefinition) -> str:
        """Add a policy. Returns policy name."""
        ...

    @abstractmethod
    def get_policy(self, name: str) -> Optional[PolicyDefinition]:
        """Get a policy by name."""
        ...

    @abstractmethod
    def get_policies(self, application_id: Optional[str] = None) -> List[PolicyDefinition]:
        """List policies, optionally filtered by application."""
        ...

    @abstractmethod
    def record_violation(self, record: ViolationRecord) -> str:
        """Record a violation. Returns violation ID."""
        ...

    @abstractmethod
    def get_violations(
        self,
        policy_name: Optional[str] = None,
        severity: Optional[str] = None,
    ) -> List[ViolationRecord]:
        """Query recorded violations."""
        ...


class InMemoryPolicyStore(PolicyStore):
    """In-memory policy store for development and testing."""

    def __init__(self):
        self._policies: Dict[str, PolicyDefinition] = {}
        self._violations: List[ViolationRecord] = []

    def add_policy(self, policy: PolicyDefinition) -> str:
        self._policies[policy.name] = policy
        return policy.name

    def get_policy(self, name: str) -> Optional[PolicyDefinition]:
        return self._policies.get(name)

    def get_policies(self, application_id: Optional[str] = None) -> List[PolicyDefinition]:
        if application_id:
            return [p for p in self._policies.values() if p.application_id == application_id]
        return list(self._policies.values())

    def record_violation(self, record: ViolationRecord) -> str:
        if not record.violation_id:
            record.violation_id = str(uuid.uuid4())
        self._violations.append(record)
        return record.violation_id

    def get_violations(
        self,
        policy_name: Optional[str] = None,
        severity: Optional[str] = None,
    ) -> List[ViolationRecord]:
        results = self._violations
        if policy_name:
            results = [v for v in results if v.policy_name == policy_name]
        if severity:
            results = [v for v in results if v.severity == severity]
        return results
