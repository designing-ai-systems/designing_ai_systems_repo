"""Tests for CircuitBreaker (Listing 6.18)."""

from services.tools.circuit_breaker import CircuitBreaker


class TestClosedState:
    def test_starts_closed(self):
        cb = CircuitBreaker()
        assert cb.get_state("tool-a") == "closed"

    def test_allows_requests_when_closed(self):
        cb = CircuitBreaker()
        assert cb.allow_request("tool-a") is True

    def test_stays_closed_on_success(self):
        cb = CircuitBreaker()
        cb.record_result("tool-a", success=True)
        assert cb.get_state("tool-a") == "closed"

    def test_stays_closed_below_threshold(self):
        cb = CircuitBreaker(failure_threshold=5)
        for _ in range(4):
            cb.record_result("tool-a", success=False)
        assert cb.get_state("tool-a") == "closed"


class TestOpenState:
    def test_opens_at_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_result("tool-a", success=False)
        assert cb.get_state("tool-a") == "open"

    def test_rejects_requests_when_open(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=9999)
        cb.record_result("tool-a", success=False)
        cb.record_result("tool-a", success=False)
        assert cb.allow_request("tool-a") is False

    def test_success_resets_counter(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_result("tool-a", success=False)
        cb.record_result("tool-a", success=False)
        cb.record_result("tool-a", success=True)
        cb.record_result("tool-a", success=False)
        assert cb.get_state("tool-a") == "closed"


class TestHalfOpenState:
    def test_transitions_to_half_open_after_recovery(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0)
        cb.record_result("tool-a", success=False)
        cb.record_result("tool-a", success=False)
        assert cb.get_state("tool-a") == "open"
        assert cb.allow_request("tool-a") is True
        assert cb.get_state("tool-a") == "half_open"

    def test_half_open_allows_limited_requests(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0, half_open_max=2)
        cb.record_result("tool-a", success=False)
        cb.record_result("tool-a", success=False)
        assert cb.allow_request("tool-a") is True  # triggers half_open, attempt 1
        assert cb.allow_request("tool-a") is True  # attempt 2
        assert cb.allow_request("tool-a") is False  # over limit

    def test_half_open_success_closes(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0)
        cb.record_result("tool-a", success=False)
        cb.record_result("tool-a", success=False)
        cb.allow_request("tool-a")  # transition to half_open
        cb.record_result("tool-a", success=True)
        assert cb.get_state("tool-a") == "closed"

    def test_half_open_failure_reopens(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0)
        cb.record_result("tool-a", success=False)
        cb.record_result("tool-a", success=False)
        cb.allow_request("tool-a")
        cb.record_result("tool-a", success=False)
        assert cb.get_state("tool-a") == "open"


class TestIsolation:
    def test_independent_tool_states(self):
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_result("tool-a", success=False)
        cb.record_result("tool-a", success=False)
        assert cb.get_state("tool-a") == "open"
        assert cb.get_state("tool-b") == "closed"
        assert cb.allow_request("tool-b") is True
