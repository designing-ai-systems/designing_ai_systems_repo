"""
Credential Store — secure credential management for tool execution.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 6.13: Tool registration references credentials by name
  - Listing 6.14: CredentialStore interface (store, retrieve, rotate)

The interface is async because production backends (HashiCorp Vault,
AWS Secrets Manager, HTTP token endpoints) perform network I/O. A sync
API would block threads or force awkward thread-pool bridges; async lets
callers compose with other async I/O (HTTP clients, gRPC aio, etc.).
"""

import fnmatch
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Credential:
    name: str = ""
    credential_type: str = "api_key"
    value: str = ""
    rotation_policy: Optional[str] = None
    allowed_tools: List[str] = field(default_factory=list)


# Listing 6.14 (async — matches real secret-manager I/O)
class CredentialStore(ABC):
    """Platform credential management."""

    @abstractmethod
    async def store(
        self,
        name: str,
        credential_type: str,
        value: str,
        rotation_policy: Optional[str] = None,
        allowed_tools: Optional[List[str]] = None,
    ) -> None: ...

    @abstractmethod
    async def retrieve(self, name: str, requesting_tool: str) -> Credential: ...

    @abstractmethod
    async def rotate(self, name: str, new_value: str) -> None: ...


class InMemoryCredentialStore(CredentialStore):
    """In-memory credential store for development and testing."""

    def __init__(self):
        self._credentials: Dict[str, Credential] = {}

    async def store(
        self,
        name: str,
        credential_type: str,
        value: str,
        rotation_policy: Optional[str] = None,
        allowed_tools: Optional[List[str]] = None,
    ) -> None:
        self._credentials[name] = Credential(
            name=name,
            credential_type=credential_type,
            value=value,
            rotation_policy=rotation_policy,
            allowed_tools=allowed_tools or [],
        )

    async def retrieve(self, name: str, requesting_tool: str) -> Credential:
        cred = self._credentials.get(name)
        if cred is None:
            raise KeyError(f"Credential '{name}' not found")
        if cred.allowed_tools:
            if not any(fnmatch.fnmatch(requesting_tool, pat) for pat in cred.allowed_tools):
                raise PermissionError(
                    f"Tool '{requesting_tool}' not authorized for credential '{name}'"
                )
        return cred

    async def rotate(self, name: str, new_value: str) -> None:
        cred = self._credentials.get(name)
        if cred is None:
            raise KeyError(f"Credential '{name}' not found")
        cred.value = new_value
