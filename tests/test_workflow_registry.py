"""Tests for WorkflowRegistry ABC + InMemoryWorkflowRegistry."""

import pytest

from services.workflow.models import WorkflowSpec
from services.workflow.store import InMemoryWorkflowRegistry, create_registry


class TestRegister:
    def test_assigns_workflow_id_and_returns_initial_version(self):
        reg = InMemoryWorkflowRegistry()
        wf_id, version = reg.register(WorkflowSpec(name="patient_intake", api_path="/p"))
        assert wf_id  # non-empty
        assert version == 1

    def test_re_registering_same_name_bumps_version(self):
        reg = InMemoryWorkflowRegistry()
        _, v1 = reg.register(WorkflowSpec(name="w", api_path="/w"))
        _, v2 = reg.register(WorkflowSpec(name="w", api_path="/w"))
        assert v1 == 1
        assert v2 == 2

    def test_workflow_id_is_stable_across_re_registration(self):
        reg = InMemoryWorkflowRegistry()
        id1, _ = reg.register(WorkflowSpec(name="w", api_path="/w"))
        id2, _ = reg.register(WorkflowSpec(name="w", api_path="/w"))
        assert id1 == id2


class TestGet:
    def test_returns_none_for_unknown(self):
        reg = InMemoryWorkflowRegistry()
        assert reg.get("does-not-exist") is None

    def test_returns_latest_spec(self):
        reg = InMemoryWorkflowRegistry()
        reg.register(WorkflowSpec(name="w", api_path="/w", response_mode="sync"))
        reg.register(WorkflowSpec(name="w", api_path="/w", response_mode="stream"))
        spec = reg.get("w")
        assert spec.response_mode == "stream"
        assert spec.version == 2


class TestList:
    def test_empty(self):
        reg = InMemoryWorkflowRegistry()
        assert reg.list() == []

    def test_returns_all_workflows(self):
        reg = InMemoryWorkflowRegistry()
        reg.register(WorkflowSpec(name="a", api_path="/a"))
        reg.register(WorkflowSpec(name="b", api_path="/b"))
        names = sorted(s.name for s in reg.list())
        assert names == ["a", "b"]


class TestUpdate:
    def test_update_increments_version(self):
        reg = InMemoryWorkflowRegistry()
        reg.register(WorkflowSpec(name="w", api_path="/w"))
        new_version = reg.update(WorkflowSpec(name="w", api_path="/w-v2"))
        assert new_version == 2
        assert reg.get("w").api_path == "/w-v2"

    def test_update_unknown_raises(self):
        reg = InMemoryWorkflowRegistry()
        with pytest.raises(KeyError):
            reg.update(WorkflowSpec(name="unknown", api_path="/x"))


class TestDelete:
    def test_delete_existing_returns_true_and_removes(self):
        reg = InMemoryWorkflowRegistry()
        reg.register(WorkflowSpec(name="w", api_path="/w"))
        assert reg.delete("w") is True
        assert reg.get("w") is None

    def test_delete_unknown_returns_false(self):
        reg = InMemoryWorkflowRegistry()
        assert reg.delete("missing") is False


class TestCreateRegistry:
    def test_default_returns_in_memory(self, monkeypatch):
        monkeypatch.delenv("WORKFLOW_REGISTRY", raising=False)
        reg = create_registry()
        assert isinstance(reg, InMemoryWorkflowRegistry)

    def test_explicit_memory(self, monkeypatch):
        monkeypatch.setenv("WORKFLOW_REGISTRY", "memory")
        reg = create_registry()
        assert isinstance(reg, InMemoryWorkflowRegistry)
