"""Tests for the gRPC RetryInterceptor (Listings 8.13–8.14)."""

import grpc

from genai_platform.grpc_retry import RETRYABLE_CODES, RetryInterceptor


class FakeCall:
    """Stands in for a grpc unary-unary outcome object."""

    def __init__(self, code: grpc.StatusCode, result_value=None):
        self._code = code
        self._result_value = result_value

    def code(self):
        return self._code

    def result(self):
        return self._result_value


class FakeContinuation:
    """A `continuation` substitute that returns scripted responses in order."""

    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls = 0

    def __call__(self, details, request):
        self.calls += 1
        return self._responses.pop(0)


class FakeDetails:
    method = "/Service/Method"
    timeout = None
    metadata = ()
    credentials = None
    wait_for_ready = None
    compression = None


class TestSuccessPath:
    def test_returns_immediately_when_first_call_succeeds(self):
        slept = []
        cont = FakeContinuation(FakeCall(grpc.StatusCode.OK, "ok"))
        interceptor = RetryInterceptor(sleep=slept.append)

        out = interceptor.intercept_unary_unary(cont, FakeDetails(), object())

        assert out.result() == "ok"
        assert cont.calls == 1
        assert slept == []  # no backoff


class TestRetryableCodes:
    def test_retries_on_unavailable_then_succeeds(self):
        slept = []
        cont = FakeContinuation(
            FakeCall(grpc.StatusCode.UNAVAILABLE),
            FakeCall(grpc.StatusCode.OK, "ok"),
        )
        interceptor = RetryInterceptor(sleep=slept.append)

        out = interceptor.intercept_unary_unary(cont, FakeDetails(), object())

        assert out.code() == grpc.StatusCode.OK
        assert cont.calls == 2
        assert slept == [0.1]  # one backoff between attempts 1 and 2

    def test_retries_each_retryable_code(self):
        for code in RETRYABLE_CODES:
            cont = FakeContinuation(FakeCall(code), FakeCall(grpc.StatusCode.OK, "ok"))
            out = RetryInterceptor(sleep=lambda _: None).intercept_unary_unary(
                cont, FakeDetails(), object()
            )
            assert out.code() == grpc.StatusCode.OK, f"failed for {code}"

    def test_exhausts_retries_then_returns_last_failure(self):
        slept = []
        cont = FakeContinuation(
            FakeCall(grpc.StatusCode.UNAVAILABLE),
            FakeCall(grpc.StatusCode.UNAVAILABLE),
            FakeCall(grpc.StatusCode.UNAVAILABLE),
        )
        interceptor = RetryInterceptor(max_attempts=3, sleep=slept.append)

        out = interceptor.intercept_unary_unary(cont, FakeDetails(), object())

        assert out.code() == grpc.StatusCode.UNAVAILABLE
        assert cont.calls == 3
        # Two backoffs between three attempts: 100ms, 200ms.
        assert slept == [0.1, 0.2]


class TestNonRetryableCodes:
    def test_invalid_argument_does_not_retry(self):
        slept = []
        cont = FakeContinuation(FakeCall(grpc.StatusCode.INVALID_ARGUMENT))
        interceptor = RetryInterceptor(sleep=slept.append)

        out = interceptor.intercept_unary_unary(cont, FakeDetails(), object())

        assert out.code() == grpc.StatusCode.INVALID_ARGUMENT
        assert cont.calls == 1
        assert slept == []

    def test_not_found_does_not_retry(self):
        cont = FakeContinuation(FakeCall(grpc.StatusCode.NOT_FOUND))
        out = RetryInterceptor(sleep=lambda _: None).intercept_unary_unary(
            cont, FakeDetails(), object()
        )
        assert out.code() == grpc.StatusCode.NOT_FOUND
        assert cont.calls == 1


class TestBackoffSchedule:
    def test_doubles_backoff_each_attempt(self):
        slept = []
        cont = FakeContinuation(
            FakeCall(grpc.StatusCode.UNAVAILABLE),
            FakeCall(grpc.StatusCode.UNAVAILABLE),
            FakeCall(grpc.StatusCode.UNAVAILABLE),
            FakeCall(grpc.StatusCode.UNAVAILABLE),
            FakeCall(grpc.StatusCode.OK, "ok"),
        )
        interceptor = RetryInterceptor(max_attempts=5, base_backoff_seconds=0.1, sleep=slept.append)

        interceptor.intercept_unary_unary(cont, FakeDetails(), object())

        # 4 retries → 4 backoff intervals doubling each time.
        assert slept == [0.1, 0.2, 0.4, 0.8]
