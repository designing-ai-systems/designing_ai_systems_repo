"""
Model Service stores.

Provides in-memory registries for prompts and custom models.
"""

from __future__ import annotations

import datetime
from typing import Dict, List, Optional

from proto import models_pb2


def _now_iso() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


class PromptRegistry:
    """In-memory prompt registry with versioning."""

    def __init__(self):
        self._prompts: Dict[str, List[models_pb2.Prompt]] = {}

    def register(
        self,
        name: str,
        content: str,
        metadata: Optional[models_pb2.PromptMetadata] = None,
    ) -> models_pb2.Prompt:
        version = len(self._prompts.get(name, [])) + 1
        created_at = _now_iso()
        prompt = models_pb2.Prompt(
            name=name,
            version=version,
            content=content,
            metadata=metadata if metadata else models_pb2.PromptMetadata(),
            created_at=created_at,
        )
        self._prompts.setdefault(name, []).append(prompt)
        return prompt

    def get(self, name: str, version: int = 0) -> Optional[models_pb2.Prompt]:
        versions = self._prompts.get(name, [])
        if not versions:
            return None
        if version <= 0:
            return versions[-1]
        if 1 <= version <= len(versions):
            return versions[version - 1]
        return None

    def list_latest(self) -> List[models_pb2.Prompt]:
        return [versions[-1] for versions in self._prompts.values() if versions]


class ModelRegistry:
    """
    In-memory registry for custom/self-hosted models.

    This registry is ONLY for custom models (not commercial providers).
    Commercial models (OpenAI, Anthropic) are handled by provider adapters.
    """

    def __init__(self):
        self._models: Dict[str, models_pb2.RegisteredModel] = {}

    def register(
        self,
        name: str,
        endpoint: str,
        capabilities: models_pb2.ModelCapabilities,
        health_check: str,
        adapter_type: str = "openai",
        provider: str = None,
    ) -> models_pb2.RegisteredModel:
        """
        Register a custom model.

        Args:
            name: Model identifier
            endpoint: HTTP endpoint where model is hosted
            capabilities: Model capabilities
            health_check: Health check path
            adapter_type: Which adapter to use ("openai", "anthropic", "vllm", etc.)
            provider: Optional display name (defaults to "custom")
        """
        registered_at = _now_iso()
        model = models_pb2.RegisteredModel(
            name=name,
            endpoint=endpoint,
            capabilities=capabilities,
            health_check=health_check,
            status="provisioning",
            registered_at=registered_at,
            provider=provider if provider else "custom",
            adapter_type=adapter_type,
        )
        self._models[name] = model
        return model

    def list_all(self) -> List[models_pb2.RegisteredModel]:
        """List all registered custom models."""
        return list(self._models.values())

    def get(self, name: str) -> Optional[models_pb2.RegisteredModel]:
        """Get a registered model by name."""
        return self._models.get(name)
