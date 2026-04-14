"""Tests for InMemoryToolRegistry (Listings 6.4, 6.5, 6.8, 6.11, 6.12)."""

from services.tools.models import ToolBehavior, ToolDefinition
from services.tools.store import InMemoryToolRegistry


class TestRegister:
    def test_register_returns_status(self):
        registry = InMemoryToolRegistry()
        tool = ToolDefinition(name="test.tool", description="A tool")
        assert registry.register(tool) == "registered"

    def test_register_multiple_versions(self):
        registry = InMemoryToolRegistry()
        registry.register(ToolDefinition(name="test.tool", version="1.0.0"))
        registry.register(ToolDefinition(name="test.tool", version="2.0.0"))
        assert registry.list_versions("test.tool") == ["1.0.0", "2.0.0"]


class TestGet:
    def test_get_latest(self):
        registry = InMemoryToolRegistry()
        registry.register(ToolDefinition(name="test.tool", version="1.0.0", description="v1"))
        registry.register(ToolDefinition(name="test.tool", version="2.0.0", description="v2"))
        tool = registry.get("test.tool")
        assert tool.version == "2.0.0"
        assert tool.description == "v2"

    def test_get_specific_version(self):
        registry = InMemoryToolRegistry()
        registry.register(ToolDefinition(name="test.tool", version="1.0.0", description="v1"))
        registry.register(ToolDefinition(name="test.tool", version="2.0.0", description="v2"))
        tool = registry.get("test.tool", version="1.0.0")
        assert tool.description == "v1"

    def test_get_nonexistent(self):
        registry = InMemoryToolRegistry()
        assert registry.get("nope") is None


class TestDiscover:
    def _populated_registry(self):
        registry = InMemoryToolRegistry()
        registry.register(
            ToolDefinition(
                name="healthcare.scheduling.book",
                capabilities=["scheduling"],
                tags=["patient-facing"],
                behavior=ToolBehavior(is_read_only=False),
            )
        )
        registry.register(
            ToolDefinition(
                name="healthcare.scheduling.check_availability",
                capabilities=["scheduling", "availability"],
                tags=["patient-facing", "read-only"],
                behavior=ToolBehavior(is_read_only=True),
            )
        )
        registry.register(
            ToolDefinition(
                name="healthcare.billing.verify_insurance",
                capabilities=["insurance", "verification"],
                tags=["patient-facing", "hipaa-compliant"],
                behavior=ToolBehavior(is_read_only=True),
            )
        )
        return registry

    def test_discover_by_namespace(self):
        registry = self._populated_registry()
        tools = registry.discover(namespace="healthcare.scheduling.*")
        assert len(tools) == 2

    def test_discover_by_capability(self):
        registry = self._populated_registry()
        tools = registry.discover(capabilities=["insurance"])
        assert len(tools) == 1
        assert tools[0].name == "healthcare.billing.verify_insurance"

    def test_discover_read_only(self):
        registry = self._populated_registry()
        tools = registry.discover(read_only=True)
        assert len(tools) == 2
        assert all(t.behavior.is_read_only for t in tools)

    def test_discover_by_tags(self):
        registry = self._populated_registry()
        tools = registry.discover(tags=["hipaa-compliant"])
        assert len(tools) == 1

    def test_discover_all(self):
        registry = self._populated_registry()
        tools = registry.discover()
        assert len(tools) == 3


class TestDeprecate:
    def test_deprecate_version_pattern(self):
        registry = InMemoryToolRegistry()
        registry.register(ToolDefinition(name="test.tool", version="1.0.0"))
        registry.register(ToolDefinition(name="test.tool", version="1.1.0"))
        registry.register(ToolDefinition(name="test.tool", version="2.0.0"))
        result = registry.deprecate("test.tool", "1.*", "2025-06-01", "Use v2")
        assert result is True

    def test_deprecate_nonexistent(self):
        registry = InMemoryToolRegistry()
        result = registry.deprecate("nope", "1.0.0", "2025-06-01", "")
        assert result is False
