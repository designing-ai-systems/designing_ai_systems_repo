"""Tests for InMemoryCredentialStore (Listing 6.14)."""

import pytest

from services.tools.credential_store import InMemoryCredentialStore


class TestStore:
    async def test_store_and_retrieve(self):
        store = InMemoryCredentialStore()
        await store.store("api-key", "api_key", "secret-value")
        cred = await store.retrieve("api-key", requesting_tool="any.tool")
        assert cred.value == "secret-value"
        assert cred.credential_type == "api_key"

    async def test_retrieve_nonexistent_raises(self):
        store = InMemoryCredentialStore()
        with pytest.raises(KeyError, match="not found"):
            await store.retrieve("nope", requesting_tool="any.tool")


class TestAccessControl:
    async def test_allowed_tools_permits_matching(self):
        store = InMemoryCredentialStore()
        await store.store(
            "scheduling-api-prod",
            "api_key",
            "key-123",
            allowed_tools=["healthcare.scheduling.*"],
        )
        cred = await store.retrieve("scheduling-api-prod", "healthcare.scheduling.book")
        assert cred.value == "key-123"

    async def test_allowed_tools_permits_exact_quickstart_tool_name(self):
        """Matches examples/quickstart_tools.py + Listing 6.13 credential_ref wiring."""
        store = InMemoryCredentialStore()
        await store.store(
            "scheduling-api-prod",
            "api_key",
            "key-123",
            allowed_tools=["healthcare.scheduling.book_appointment"],
        )
        cred = await store.retrieve("scheduling-api-prod", "healthcare.scheduling.book_appointment")
        assert cred.value == "key-123"

    async def test_allowed_tools_denies_non_matching(self):
        store = InMemoryCredentialStore()
        await store.store(
            "scheduling-api-prod",
            "api_key",
            "key-123",
            allowed_tools=["healthcare.scheduling.*"],
        )
        with pytest.raises(PermissionError, match="not authorized"):
            await store.retrieve("scheduling-api-prod", "healthcare.billing.charge")

    async def test_no_allowed_tools_permits_all(self):
        store = InMemoryCredentialStore()
        await store.store("open-key", "api_key", "open-123")
        cred = await store.retrieve("open-key", "any.tool.anywhere")
        assert cred.value == "open-123"


class TestRotate:
    async def test_rotate_updates_value(self):
        store = InMemoryCredentialStore()
        await store.store("api-key", "api_key", "old-value")
        await store.rotate("api-key", "new-value")
        cred = await store.retrieve("api-key", requesting_tool="any.tool")
        assert cred.value == "new-value"

    async def test_rotate_nonexistent_raises(self):
        store = InMemoryCredentialStore()
        with pytest.raises(KeyError, match="not found"):
            await store.rotate("nope", "value")
