"""Tests for the @workflow decorator (Listing 8.2 — flat kwargs).

This is the form the reference implementation ships, deliberately diverging
from the nested-dict draft of Listing 8.2; see
chapters/book_discrepancies_chapter8.md, discrepancy #1.
"""

import pytest

from genai_platform import workflow


class TestPassThrough:
    def test_decorator_returns_function_that_runs_unchanged(self):
        @workflow(name="patient_intake", api_path="/p")
        def handle(question: str) -> dict:
            return {"echo": question}

        assert handle("hello") == {"echo": "hello"}


class TestMetadata:
    def test_attaches_minimal_metadata(self):
        @workflow(name="w", api_path="/w")
        def f():
            pass

        meta = f._workflow_metadata
        assert meta["name"] == "w"
        assert meta["api_path"] == "/w"
        assert meta["response_mode"] == "sync"

    def test_response_mode_explicit(self):
        @workflow(name="w", api_path="/w", response_mode="stream")
        def f():
            pass

        assert f._workflow_metadata["response_mode"] == "stream"

    def test_invalid_response_mode_rejected(self):
        with pytest.raises(ValueError, match="response_mode"):

            @workflow(name="w", api_path="/w", response_mode="not-a-mode")
            def f():
                pass

    def test_scaling_fields_flatten_into_metadata(self):
        @workflow(
            name="w",
            api_path="/w",
            min_replicas=2,
            max_replicas=20,
            target_cpu_percent=50,
        )
        def f():
            pass

        meta = f._workflow_metadata
        assert meta["min_replicas"] == 2
        assert meta["max_replicas"] == 20
        assert meta["target_cpu_percent"] == 50

    def test_resource_fields_flatten_into_metadata(self):
        @workflow(
            name="w",
            api_path="/w",
            cpu="2",
            memory="8Gi",
            gpu_type="nvidia-t4",
            num_gpus=1,
        )
        def f():
            pass

        meta = f._workflow_metadata
        assert meta["cpu"] == "2"
        assert meta["memory"] == "8Gi"
        assert meta["gpu_type"] == "nvidia-t4"
        assert meta["num_gpus"] == 1

    def test_reliability_fields_flatten_into_metadata(self):
        @workflow(name="w", api_path="/w", timeout_seconds=120, max_retries=5)
        def f():
            pass

        meta = f._workflow_metadata
        assert meta["timeout_seconds"] == 120
        assert meta["max_retries"] == 5

    def test_defaults_match_chapter_8(self):
        @workflow(name="w", api_path="/w")
        def f():
            pass

        meta = f._workflow_metadata
        assert meta["min_replicas"] == 1
        assert meta["max_replicas"] == 10
        assert meta["target_cpu_percent"] == 70
        assert meta["cpu"] == "500m"
        assert meta["memory"] == "512Mi"
        assert meta["gpu_type"] == ""
        assert meta["num_gpus"] == 0
        assert meta["timeout_seconds"] == 30
        assert meta["max_retries"] == 3


class TestDecoratorPreservesIntrospection:
    def test_function_name_preserved(self):
        @workflow(name="w", api_path="/w")
        def handle_patient_question():
            pass

        assert handle_patient_question.__name__ == "handle_patient_question"

    def test_docstring_preserved(self):
        @workflow(name="w", api_path="/w")
        def f():
            """Summary."""

        assert f.__doc__ == "Summary."
