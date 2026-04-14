"""
Tool Registry — storage abstraction for tool definitions.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 6.4: Tool registration
  - Listing 6.5: Tool discovery by namespace
  - Listing 6.7: Discovery request with namespace/capabilities/tags
  - Listing 6.10: Version-constrained discovery
"""

import fnmatch
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from services.tools.models import ToolDefinition


def _parse_semver(version: str) -> Tuple[int, ...]:
    """Parse 'major.minor.patch' into a comparable tuple."""
    return tuple(int(x) for x in version.split("."))


def _matches_version_constraint(version: str, constraint: str) -> bool:
    """Check whether *version* satisfies *constraint*.

    Supported forms (Listing 6.10):
      - Exact:  "2.1.0"
      - Caret:  "^1.0.0"  (any version with the same major)
      - Range:  ">=1.0.0 <3.0.0"
    """
    constraint = constraint.strip()
    v = _parse_semver(version)

    if constraint.startswith("^"):
        base = _parse_semver(constraint[1:])
        return v[0] == base[0] and v >= base

    range_match = re.match(r">=\s*([\d.]+)\s+<\s*([\d.]+)", constraint)
    if range_match:
        lo = _parse_semver(range_match.group(1))
        hi = _parse_semver(range_match.group(2))
        return lo <= v < hi

    return v == _parse_semver(constraint)


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
        latest = max(versions.keys(), key=_parse_semver)
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
            if version_constraint:
                matched = [
                    t for t in versions.values()
                    if _matches_version_constraint(t.version, version_constraint)
                ]
            else:
                matched = [max(versions.values(), key=lambda t: _parse_semver(t.version))]

            for tool in matched:
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
        return sorted(self._tools.get(name, {}).keys(), key=_parse_semver)

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
