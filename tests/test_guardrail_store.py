"""Tests for InMemoryPolicyStore."""

from services.guardrails.models import PolicyDefinition, ViolationRecord
from services.guardrails.store import InMemoryPolicyStore


class TestAddAndGet:
    def test_add_and_get_policy(self):
        store = InMemoryPolicyStore()
        policy = PolicyDefinition(name="test-policy", type="input", application_id="app-1")
        store.add_policy(policy)
        got = store.get_policy("test-policy")
        assert got is not None
        assert got.type == "input"

    def test_get_nonexistent(self):
        store = InMemoryPolicyStore()
        assert store.get_policy("nope") is None

    def test_list_by_application(self):
        store = InMemoryPolicyStore()
        store.add_policy(PolicyDefinition(name="p1", application_id="app-1"))
        store.add_policy(PolicyDefinition(name="p2", application_id="app-2"))
        store.add_policy(PolicyDefinition(name="p3", application_id="app-1"))
        results = store.get_policies(application_id="app-1")
        assert len(results) == 2

    def test_list_all(self):
        store = InMemoryPolicyStore()
        store.add_policy(PolicyDefinition(name="p1"))
        store.add_policy(PolicyDefinition(name="p2"))
        assert len(store.get_policies()) == 2


class TestViolations:
    def test_record_and_query(self):
        store = InMemoryPolicyStore()
        vid = store.record_violation(
            ViolationRecord(policy_name="booking-rules", severity="blocked")
        )
        assert vid  # non-empty ID

        violations = store.get_violations(policy_name="booking-rules")
        assert len(violations) == 1
        assert violations[0].severity == "blocked"

    def test_filter_by_severity(self):
        store = InMemoryPolicyStore()
        store.record_violation(ViolationRecord(policy_name="p1", severity="low"))
        store.record_violation(ViolationRecord(policy_name="p1", severity="high"))
        store.record_violation(ViolationRecord(policy_name="p2", severity="high"))

        high = store.get_violations(severity="high")
        assert len(high) == 2

        p1_high = store.get_violations(policy_name="p1", severity="high")
        assert len(p1_high) == 1
