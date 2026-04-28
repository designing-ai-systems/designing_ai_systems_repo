"""Tests for BaseClient's TLS-vs-plaintext channel selection.

The SDK should:
- use plaintext for ``localhost`` / ``127.0.0.1`` URLs (local dev)
- use plaintext when ``GENAI_GATEWAY_INSECURE=1`` is set (compose-network
  services and locally-launched workflow containers — the in-cluster
  gateway is plain gRPC even though its URL doesn't look like
  ``localhost``)
- use TLS otherwise (production with a public gateway hostname)
"""

from genai_platform.clients.base import _use_insecure_channel


def test_localhost_url_is_plaintext(monkeypatch):
    monkeypatch.delenv("GENAI_GATEWAY_INSECURE", raising=False)
    assert _use_insecure_channel("localhost:50051") is True


def test_127_0_0_1_url_is_plaintext(monkeypatch):
    monkeypatch.delenv("GENAI_GATEWAY_INSECURE", raising=False)
    assert _use_insecure_channel("127.0.0.1:50051") is True


def test_compose_hostname_defaults_to_tls(monkeypatch):
    """Without the env-var override, a non-localhost URL means TLS."""
    monkeypatch.delenv("GENAI_GATEWAY_INSECURE", raising=False)
    assert _use_insecure_channel("gateway:50051") is False


def test_compose_hostname_with_env_var_is_plaintext(monkeypatch):
    """In compose, services and workflow containers set this env var to
    opt out of TLS — the in-cluster gateway is plain gRPC."""
    monkeypatch.setenv("GENAI_GATEWAY_INSECURE", "1")
    assert _use_insecure_channel("gateway:50051") is True


def test_env_var_zero_does_not_force_plaintext(monkeypatch):
    """Only ``=1`` opts in; any other value leaves the default behavior."""
    monkeypatch.setenv("GENAI_GATEWAY_INSECURE", "0")
    assert _use_insecure_channel("gateway:50051") is False


def test_public_hostname_with_env_unset_is_tls(monkeypatch):
    monkeypatch.delenv("GENAI_GATEWAY_INSECURE", raising=False)
    assert _use_insecure_channel("api.example.com:443") is False
