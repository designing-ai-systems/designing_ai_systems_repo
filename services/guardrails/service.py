"""
Guardrails Service — gRPC service implementation (grpc.aio).

Thin translation layer: receives proto requests, delegates to
PolicyStore for evaluation, returns proto responses.

Async servicers align with the book's await-based guardrail examples
(Listings 6.19–6.23).

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 6.19: GuardrailsService gRPC contract
  - Listing 6.20: Multi-point guardrail evaluation
  - Listing 6.21: Input guardrail configuration
  - Listing 6.23: Tiered violation handling
"""

import logging
import re

import grpc
import grpc.aio

from proto import guardrails_pb2, guardrails_pb2_grpc
from services.guardrails.models import PolicyResult, ViolationRecord
from services.guardrails.store import InMemoryPolicyStore, PolicyStore
from services.shared.servicer_base import BaseAioServicer

logger = logging.getLogger(__name__)

PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(previous|above|all)\s+(instructions|prompts)",
    r"you\s+are\s+now\s+in\s+.*mode",
    r"system\s*:\s*",
    r"<\s*/?\s*system\s*>",
]

PII_PATTERNS = [
    r"\b\d{3}-\d{2}-\d{4}\b",
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
]


class GuardrailsServiceImpl(guardrails_pb2_grpc.GuardrailsServiceServicer, BaseAioServicer):
    def __init__(self, policy_store: PolicyStore | None = None):
        self.policy_store = policy_store or InMemoryPolicyStore()

    def add_to_aio_server(self, server: grpc.aio.Server) -> None:
        guardrails_pb2_grpc.add_GuardrailsServiceServicer_to_server(self, server)

    async def ValidateInput(self, request, context):
        try:
            triggered = []
            known_checks = {"prompt_injection", "pii_detection"}
            for check in request.checks:
                if check == "prompt_injection":
                    for pattern in PROMPT_INJECTION_PATTERNS:
                        if re.search(pattern, request.content, re.IGNORECASE):
                            triggered.append("prompt_injection")
                            break
                elif check == "pii_detection":
                    for pattern in PII_PATTERNS:
                        if re.search(pattern, request.content):
                            triggered.append("pii_detection")
                            break
                elif check not in known_checks:
                    logger.warning("Unrecognized check type %r — skipped", check)

            allowed = len(triggered) == 0
            reason = ""
            if not allowed:
                reason = f"Content blocked by checks: {', '.join(triggered)}"

            return guardrails_pb2.ValidateInputResponse(
                allowed=allowed,
                denial_reason=reason,
                triggered_checks=triggered,
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return guardrails_pb2.ValidateInputResponse(allowed=True)

    async def FilterOutput(self, request, context):
        try:
            content = request.content
            applied = []
            for f in request.filters:
                if f == "pii_redaction":
                    for pattern in PII_PATTERNS:
                        new_content = re.sub(pattern, "[REDACTED]", content)
                        if new_content != content:
                            applied.append("pii_redaction")
                            content = new_content

            return guardrails_pb2.FilterOutputResponse(
                content=content,
                modified=len(applied) > 0,
                applied_filters=applied,
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return guardrails_pb2.FilterOutputResponse(content=request.content, modified=False)

    async def CheckPolicy(self, request, context):
        try:
            policy = self.policy_store.get_policy(request.policy_name)
            if not policy:
                return guardrails_pb2.CheckPolicyResponse(
                    allowed=True,
                    denial_reason="",
                )

            result = self._evaluate_policy(policy, request.action, dict(request.context))

            return guardrails_pb2.CheckPolicyResponse(
                allowed=result.allowed,
                denial_reason=result.denial_reason or "",
                violated_rules=result.violated_rules,
                suggested_action=result.suggested_action or "",
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return guardrails_pb2.CheckPolicyResponse(allowed=True)

    async def ReportViolation(self, request, context):
        try:
            record = ViolationRecord(
                policy_name=request.policy_name,
                action=request.action,
                severity=request.severity,
                context=dict(request.context),
                details=request.details,
            )
            vid = self.policy_store.record_violation(record)
            return guardrails_pb2.ReportViolationResponse(violation_id=vid, recorded=True)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return guardrails_pb2.ReportViolationResponse(recorded=False)

    def _evaluate_policy(self, policy, action: str, ctx: dict) -> PolicyResult:
        """Evaluate rules against an action and context."""
        violated = []
        for rule in policy.rules:
            rule_type = rule.get("type", "")
            if rule_type == "required_field":
                field_name = rule.get("field", "")
                if field_name and field_name not in ctx:
                    violated.append(f"required_field:{field_name}")
            elif rule_type == "blocked_action":
                blocked = rule.get("actions", [])
                if action in blocked:
                    violated.append(f"blocked_action:{action}")

        if violated:
            return PolicyResult(
                allowed=False,
                denial_reason=f"Policy '{policy.name}' violated",
                violated_rules=violated,
                suggested_action="Review and fix the violations",
            )
        return PolicyResult(allowed=True)
