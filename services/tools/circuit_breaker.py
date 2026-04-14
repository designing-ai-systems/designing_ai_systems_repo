"""
Circuit Breaker — prevents cascading failures in tool execution.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 6.18: CircuitBreaker class with closed/open/half-open states

States:
  - closed: tool is healthy, requests flow normally
  - open: tool exceeded failure threshold, requests fail immediately
  - half_open: recovery timeout elapsed, limited test requests allowed
"""

import time
from dataclasses import dataclass
from typing import Dict


@dataclass
class _ToolState:
    consecutive_failures: int = 0
    state: str = "closed"
    last_failure_time: float = 0.0
    half_open_attempts: int = 0


# Listing 6.18
class CircuitBreaker:
    """Tracks tool health and prevents cascading failures."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_max: int = 2,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max = half_open_max
        self._states: Dict[str, _ToolState] = {}

    def get_state(self, tool_name: str) -> str:
        return self._get_tool_state(tool_name).state

    def allow_request(self, tool_name: str) -> bool:
        ts = self._get_tool_state(tool_name)
        if ts.state == "closed":
            return True
        if ts.state == "open":
            if self._recovery_elapsed(ts):
                ts.state = "half_open"
                ts.half_open_attempts = 1
                return True
            return False
        # half_open
        if ts.half_open_attempts < self.half_open_max:
            ts.half_open_attempts += 1
            return True
        return False

    def record_result(self, tool_name: str, success: bool) -> None:
        ts = self._get_tool_state(tool_name)
        if success:
            ts.consecutive_failures = 0
            ts.state = "closed"
            ts.half_open_attempts = 0
        else:
            ts.consecutive_failures += 1
            ts.last_failure_time = time.time()
            if ts.state == "half_open":
                ts.state = "open"
            elif ts.consecutive_failures >= self.failure_threshold:
                ts.state = "open"

    def _get_tool_state(self, tool_name: str) -> _ToolState:
        if tool_name not in self._states:
            self._states[tool_name] = _ToolState()
        return self._states[tool_name]

    def _recovery_elapsed(self, ts: _ToolState) -> bool:
        return (time.time() - ts.last_failure_time) >= self.recovery_timeout
