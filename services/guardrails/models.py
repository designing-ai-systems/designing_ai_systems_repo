"""
Guardrails Service domain models.

Python dataclasses for policy results, violation records,
and guardrail check configuration.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 6.19: CheckPolicyResponse shape (PolicyResult)
  - Listing 6.20: Multi-point guardrail evaluation pattern
  - Listing 6.23: Tiered violation handling
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# Listing 6.19 (mirrors CheckPolicyResponse)
@dataclass
class PolicyResult:
    allowed: bool = True
    denial_reason: Optional[str] = None
    violated_rules: List[str] = field(default_factory=list)
    suggested_action: Optional[str] = None


@dataclass
class ViolationRecord:
    violation_id: str = ""
    policy_name: str = ""
    action: str = ""
    severity: str = "low"
    context: Dict[str, str] = field(default_factory=dict)
    details: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# Listing 6.21 / 6.23 shape
@dataclass
class GuardrailCheck:
    type: str = ""
    action: str = "block"
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyDefinition:
    name: str = ""
    type: str = ""
    rules: List[Dict[str, Any]] = field(default_factory=list)
    application_id: str = ""
