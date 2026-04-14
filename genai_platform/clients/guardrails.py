"""
Guardrails Service client.

Enforces safety policies and compliance checks.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 6.19: GuardrailsService gRPC contract
  - Listing 6.20: Multi-point guardrail evaluation
  - Listing 6.21: Input guardrail configuration
  - Listing 6.23: Tiered violation handling
"""

from typing import Any, Dict, List, Optional

from proto import guardrails_pb2, guardrails_pb2_grpc
from services.guardrails.models import PolicyResult

from .base import BaseClient


class GuardrailsClient(BaseClient):
    """Client for Guardrails Service — input validation, output filtering, policy checks."""

    def __init__(self, platform):
        super().__init__(platform, "guardrails")
        self._stub = guardrails_pb2_grpc.GuardrailsServiceStub(self._channel)

    # --- Input validation (Listing 6.20, 6.21) ---

    def validate_input(
        self,
        content: str,
        checks: Optional[List[str]] = None,
        context: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        request = guardrails_pb2.ValidateInputRequest(
            content=content,
            checks=checks or [],
            context=context or {},
        )
        response = self._stub.ValidateInput(request, metadata=self.metadata)
        return {
            "allowed": response.allowed,
            "denial_reason": response.denial_reason,
            "triggered_checks": list(response.triggered_checks),
        }

    # --- Output filtering (Listing 6.23) ---

    def filter_output(
        self,
        content: str,
        filters: Optional[List[str]] = None,
        context: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        request = guardrails_pb2.FilterOutputRequest(
            content=content,
            filters=filters or [],
            context=context or {},
        )
        response = self._stub.FilterOutput(request, metadata=self.metadata)
        return {
            "content": response.content,
            "modified": response.modified,
            "applied_filters": list(response.applied_filters),
        }

    # --- Policy check (Listing 6.19) ---

    def check_policy(
        self,
        policy_name: str,
        action: str,
        context: Optional[Dict[str, str]] = None,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> PolicyResult:
        import json

        request = guardrails_pb2.CheckPolicyRequest(
            policy_name=policy_name,
            action=action,
            context=context or {},
            arguments_json=json.dumps(arguments) if arguments else "",
        )
        response = self._stub.CheckPolicy(request, metadata=self.metadata)
        return PolicyResult(
            allowed=response.allowed,
            denial_reason=response.denial_reason or None,
            violated_rules=list(response.violated_rules),
            suggested_action=response.suggested_action or None,
        )

    # --- Violation reporting ---

    def report_violation(
        self,
        policy_name: str,
        action: str,
        severity: str = "low",
        context: Optional[Dict[str, str]] = None,
        details: str = "",
    ) -> Dict[str, Any]:
        request = guardrails_pb2.ReportViolationRequest(
            policy_name=policy_name,
            action=action,
            severity=severity,
            context=context or {},
            details=details,
        )
        response = self._stub.ReportViolation(request, metadata=self.metadata)
        return {"violation_id": response.violation_id, "recorded": response.recorded}
