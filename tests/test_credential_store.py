"""Tests for InMemoryCredentialStore (Listing 6.14)."""

import pytest

from services.tools.credential_store import InMemoryCredentialStore


class TestStore:
    def test_store_and_retrieve(self):
        store = InMemoryCredentialStore()
        store.store("api-key", "api_key", "secret-value")
        cred = store.retrieve("api-key", requesting_tool="any.tool")
        assert cred.value == "secret-value"
        assert cred.credential_type == "api_key"

    def test_retrieve_nonexistent_raises(self):
        store = InMemoryCredentialStore()
        with pytest.raises(KeyError, match="not found"):
            store.retrieve("nope", requesting_tool="any.tool")


class TestAccessControl:
    def test_allowed_tools_permits_matching(self):
        store = InMemoryCredentialStore()
        store.store(
            "scheduling-api-prod",
            "api_key",
            "key-123",
            allowed_tools=["healthcare.scheduling.*"],
        )
        cred = store.retrieve("scheduling-api-prod", "healthcare.scheduling.book")
        assert cred.value == "key-123"

    def test_allowed_tools_denies_non_matching(self):
        store = InMemoryCredentialStore()
        store.store(
            "scheduling-api-prod",
            "api_key",
            "key-123",
            allowed_tools=["healthcare.scheduling.*"],
        )
        with pytest.raises(PermissionError, match="not authorized"):
            store.retrieve("scheduling-api-prod", "healthcare.billing.charge")

    def test_no_allowed_tools_permits_all(self):
        store = InMemoryCredentialStore()
        store.store("open-key", "api_key", "open-123")
        cred = store.retrieve("open-key", "any.tool.anywhere")
        assert cred.value == "open-123"


class TestRotate:
    def test_rotate_updates_value(self):
        store = InMemoryCredentialStore()
        store.store("api-key", "api_key", "old-value")
        store.rotate("api-key", "new-value")
        cred = store.retrieve("api-key", requesting_tool="any.tool")
        assert cred.value == "new-value"

    def test_rotate_nonexistent_raises(self):
        store = InMemoryCredentialStore()
        with pytest.raises(KeyError, match="not found"):
            store.rotate("nope", "value")
