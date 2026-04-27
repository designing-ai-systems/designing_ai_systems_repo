"""Tests for JobStore ABC + InMemoryJobStore (Listings 8.10–8.11)."""

from services.workflow.jobs_store import InMemoryJobStore


class TestCreateJob:
    def test_returns_unique_job_id_with_initial_state(self):
        store = InMemoryJobStore()
        jid = store.create_job(
            workflow_id="w-1", input_json='{"q": "x"}', assigned_endpoint="localhost:8001"
        )
        assert jid

        job = store.get_job(jid)
        assert job is not None
        assert job.job_id == jid
        assert job.workflow_id == "w-1"
        assert job.input_json == '{"q": "x"}'
        assert job.assigned_endpoint == "localhost:8001"
        assert job.status == "pending"
        assert job.created_at > 0

    def test_two_jobs_get_distinct_ids(self):
        store = InMemoryJobStore()
        a = store.create_job(workflow_id="w", input_json="{}")
        b = store.create_job(workflow_id="w", input_json="{}")
        assert a != b


class TestGetJob:
    def test_unknown_returns_none(self):
        assert InMemoryJobStore().get_job("missing") is None


class TestUpdateProgress:
    def test_sets_message_and_returns_true(self):
        store = InMemoryJobStore()
        jid = store.create_job(workflow_id="w", input_json="{}")
        assert store.update_progress(jid, "step 2 of 3") is True
        assert store.get_job(jid).progress_message == "step 2 of 3"

    def test_returns_false_for_unknown(self):
        assert InMemoryJobStore().update_progress("missing", "x") is False

    def test_bumps_updated_at(self):
        import time

        store = InMemoryJobStore()
        jid = store.create_job(workflow_id="w", input_json="{}")
        before = store.get_job(jid).updated_at
        time.sleep(0.002)  # ensure ms tick
        store.update_progress(jid, "next")
        after = store.get_job(jid).updated_at
        assert after >= before


class TestSaveCheckpoint:
    def test_persists_checkpoint_payload(self):
        store = InMemoryJobStore()
        jid = store.create_job(workflow_id="w", input_json="{}")
        assert store.save_checkpoint(jid, '{"phase": "retrieved"}') is True
        assert store.get_job(jid).checkpoint_json == '{"phase": "retrieved"}'


class TestComplete:
    def test_sets_status_succeeded_and_result(self):
        store = InMemoryJobStore()
        jid = store.create_job(workflow_id="w", input_json="{}")
        assert store.complete(jid, '{"answer": 42}') is True
        job = store.get_job(jid)
        assert job.status == "succeeded"
        assert job.result_json == '{"answer": 42}'


class TestFail:
    def test_sets_status_failed_and_error(self):
        store = InMemoryJobStore()
        jid = store.create_job(workflow_id="w", input_json="{}")
        assert store.fail(jid, "boom") is True
        job = store.get_job(jid)
        assert job.status == "failed"
        assert job.error == "boom"


class TestCancel:
    def test_sets_status_cancelled(self):
        store = InMemoryJobStore()
        jid = store.create_job(workflow_id="w", input_json="{}")
        assert store.cancel(jid) is True
        assert store.get_job(jid).status == "cancelled"
