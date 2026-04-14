"""Tests for Guardrails Service domain dataclasses (Listing 6.19, 6.23)."""

from services.guardrails.models import (
    GuardrailCheck,
    PolicyDefinition,
    PolicyResult,
    ViolationRecord,
)


class TestPolicyResult:
    def test_allowed_by_default(self):
        r = PolicyResult()
        assert r.allowed is True
        assert r.violated_rules == []

    def test_denied(self):
        r = PolicyResult(
            allowed=False,
            denial_reason="Missing referral",
            violated_rules=["referral-required"],
            suggested_action="Submit referral first",
        )
        assert r.allowed is False
        assert len(r.violated_rules) == 1


class TestViolationRecord:
    def test_record(self):
        v = ViolationRecord(
            policy_name="booking-rules",
            action="healthcare.scheduling.book",
            severity="blocked",
            context={"user_id": "patient-123"},
        )
        assert v.severity == "blocked"
        assert v.timestamp is not None


class TestGuardrailCheck:
    def test_check(self):
        c = GuardrailCheck(type="prompt_injection", action="block")
        assert c.type == "prompt_injection"
        assert c.action == "block"


class TestPolicyDefinition:
    def test_definition(self):
        p = PolicyDefinition(
            name="intake-input-policies",
            type="input",
            application_id="patient-intake",
            rules=[{"type": "prompt_injection", "action": "block"}],
        )
        assert p.name == "intake-input-policies"
        assert len(p.rules) == 1
