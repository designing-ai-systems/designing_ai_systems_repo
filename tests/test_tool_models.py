"""Tests for Tool Service domain dataclasses (Listings 6.2, 6.3, 6.16, 6.19)."""

from services.tools.models import (
    CostMetadata,
    ExecutionLimits,
    RateLimits,
    ToolBehavior,
    ToolDefinition,
    ToolExecutionResult,
    ToolTask,
)


class TestToolBehavior:
    def test_defaults(self):
        b = ToolBehavior()
        assert b.is_read_only is False
        assert b.is_idempotent is False
        assert b.requires_confirmation is False
        assert b.typical_latency_ms == 0
        assert b.side_effects == []

    def test_read_only_tool(self):
        b = ToolBehavior(is_read_only=True, is_idempotent=True)
        assert b.is_read_only is True
        assert b.is_idempotent is True


class TestRateLimits:
    def test_defaults(self):
        r = RateLimits()
        assert r.requests_per_minute == 60
        assert r.requests_per_session == 0
        assert r.daily_limit == 0

    def test_session_limit(self):
        r = RateLimits(requests_per_session=3)
        assert r.requests_per_session == 3


class TestCostMetadata:
    def test_defaults(self):
        c = CostMetadata()
        assert c.estimated_cost_usd == 0.0
        assert c.billing_category == ""


class TestExecutionLimits:
    def test_defaults(self):
        e = ExecutionLimits()
        assert e.timeout_seconds == 30
        assert e.max_retries == 3

    def test_custom(self):
        e = ExecutionLimits(timeout_seconds=10, memory_limit_mb=128)
        assert e.timeout_seconds == 10
        assert e.memory_limit_mb == 128


class TestToolDefinition:
    def test_minimal(self):
        t = ToolDefinition(name="test.tool", description="A test tool")
        assert t.name == "test.tool"
        assert t.version == "1.0.0"
        assert t.capabilities == []
        assert t.tags == []

    def test_full(self):
        t = ToolDefinition(
            name="healthcare.scheduling.book_appointment",
            version="2.0.0",
            owner="scheduling-team",
            description="Book an appointment",
            parameters={"patient_id": {"type": "string"}},
            behavior=ToolBehavior(is_read_only=False, is_idempotent=False),
            rate_limits=RateLimits(requests_per_session=3),
            cost=CostMetadata(estimated_cost_usd=0.05),
            capabilities=["scheduling", "booking"],
            tags=["patient-facing"],
            endpoint="https://api.clinic.com/bookings",
            credential_ref="scheduling-api-prod",
        )
        assert t.owner == "scheduling-team"
        assert t.behavior.is_idempotent is False
        assert t.rate_limits.requests_per_session == 3


class TestToolExecutionResult:
    def test_success(self):
        r = ToolExecutionResult(tool_name="test", success=True, result={"id": "123"})
        assert r.success is True
        assert r.result == {"id": "123"}

    def test_failure(self):
        r = ToolExecutionResult(tool_name="test", success=False, error="timeout")
        assert r.success is False
        assert r.error == "timeout"


class TestToolTask:
    def test_pending(self):
        t = ToolTask(id="task-1", tool_name="test", status="pending")
        assert t.status == "pending"
        assert t.result is None
