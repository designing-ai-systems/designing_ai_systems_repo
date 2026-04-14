"""
Tool Registry — storage abstraction for tool definitions.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 6.4: Tool registration
  - Listing 6.5: Tool discovery by namespace
  - Listing 6.7: Discovery request with namespace/capabilities/tags
  - Listing 6.10: Version-constrained discovery
"""

import fnmatch
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from services.tools.models import ToolDefinition


class ToolRegistry(ABC):
    """Abstract base class for tool registries."""

    @abstractmethod
    def register(self, tool: ToolDefinition) -> str:
        """Register a tool. Returns status string."""
        ...

    @abstractmethod
    def get(self, name: str, version: Optional[str] = None) -> Optional[ToolDefinition]:
        """Get a tool by name and optional version."""
        ...

    @abstractmethod
    def discover(
        self,
        namespace: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        read_only: bool = False,
        version_constraint: Optional[str] = None,
    ) -> List[ToolDefinition]:
        """Discover tools matching filters."""
        ...

    @abstractmethod
    def list_versions(self, name: str) -> List[str]:
        """List all versions of a tool."""
        ...

    @abstractmethod
    def deprecate(self, name: str, version: str, sunset_date: str, migration_guide: str) -> bool:
        """Mark a tool version as deprecated."""
        ...


class InMemoryToolRegistry(ToolRegistry):
    """In-memory tool registry for development and testing."""

    def __init__(self):
        self._tools: Dict[str, Dict[str, ToolDefinition]] = {}
        self._deprecated: Dict[str, Dict[str, dict]] = {}

    def register(self, tool: ToolDefinition) -> str:
        if tool.name not in self._tools:
            self._tools[tool.name] = {}
        self._tools[tool.name][tool.version] = tool
        return "registered"

    def get(self, name: str, version: Optional[str] = None) -> Optional[ToolDefinition]:
        versions = self._tools.get(name, {})
        if not versions:
            return None
        if version:
            return versions.get(version)
        latest = max(versions.keys())
        return versions[latest]

    def discover(
        self,
        namespace: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        read_only: bool = False,
        version_constraint: Optional[str] = None,
    ) -> List[ToolDefinition]:
        results = []
        for versions in self._tools.values():
            tool = max(versions.values(), key=lambda t: t.version)

            if namespace and not fnmatch.fnmatch(tool.name, namespace):
                continue
            if capabilities and not set(capabilities).intersection(tool.capabilities):
                continue
            if tags and not set(tags).intersection(tool.tags):
                continue
            if read_only and tool.behavior and not tool.behavior.is_read_only:
                continue

            results.append(tool)
        return results

    def list_versions(self, name: str) -> List[str]:
        return sorted(self._tools.get(name, {}).keys())

    def deprecate(self, name: str, version: str, sunset_date: str, migration_guide: str) -> bool:
        versions = self._tools.get(name, {})
        matched = [v for v in versions if fnmatch.fnmatch(v, version)]
        if not matched:
            return False
        if name not in self._deprecated:
            self._deprecated[name] = {}
        for v in matched:
            self._deprecated[name][v] = {
                "sunset_date": sunset_date,
                "migration_guide": migration_guide,
            }
        return True
