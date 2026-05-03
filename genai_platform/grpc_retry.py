"""
gRPC retry interceptor for SDK service calls (Listings 8.13–8.14).

Wraps every outgoing gRPC call from a service client (`SessionClient`,
`ModelClient`, `WorkflowClient`, …) with exponential-backoff retries on a
small set of codes that indicate transient infrastructure trouble.

Retryable codes (Listing 8.13):
- UNAVAILABLE        — server unreachable / not yet ready
- DEADLINE_EXCEEDED  — call timed out before completing
- RESOURCE_EXHAUSTED — server-side rate limit or load shedding

Anything else (INVALID_ARGUMENT, NOT_FOUND, PERMISSION_DENIED, …) propagates
immediately. The retry layer is for transient infra blips, not for hiding
business-logic errors.

This is a *different layer* from the Model Service's internal retry to
external providers (Chapter 3): that one handles failures between Model
Service and OpenAI/Anthropic; this one handles failures between any SDK
client and the platform's gateway.
"""

import time
from typing import Callable

import grpc

RETRYABLE_CODES = frozenset(
    {
        grpc.StatusCode.UNAVAILABLE,
        grpc.StatusCode.DEADLINE_EXCEEDED,
        grpc.StatusCode.RESOURCE_EXHAUSTED,
    }
)

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BASE_BACKOFF_SECONDS = 0.1  # 100 ms; doubles per attempt → 0.1, 0.2, 0.4


class RetryInterceptor(grpc.UnaryUnaryClientInterceptor):
    """Listing 8.13 — retry transient gRPC failures with exponential backoff."""

    def __init__(
        self,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        base_backoff_seconds: float = DEFAULT_BASE_BACKOFF_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.max_attempts = max_attempts
        self.base_backoff_seconds = base_backoff_seconds
        self._sleep = sleep

    def intercept_unary_unary(self, continuation, client_call_details, request):
        last_response = None
        for attempt in range(self.max_attempts):
            try:
                response = continuation(client_call_details, request)
                code = response.code() if hasattr(response, "code") else None
            except grpc.RpcError as e:
                response = e
                code = e.code()

            if code is None or code == grpc.StatusCode.OK:
                return response
            if code not in RETRYABLE_CODES:
                return response

            last_response = response
            if attempt < self.max_attempts - 1:
                self._sleep(self.base_backoff_seconds * (2**attempt))

        return last_response
