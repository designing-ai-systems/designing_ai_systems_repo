"""In-process tests for ExperimentationServiceImpl (Listing 7.14)."""

import grpc

from proto import experiments_pb2
from services.experiments.service import ExperimentationServiceImpl


class FakeContext:
    def __init__(self):
        self.code = None
        self.details_str = None

    def set_code(self, code):
        self.code = code

    def set_details(self, details):
        self.details_str = details

    async def abort(self, code, details):
        self.code = code
        self.details_str = details
        raise grpc.RpcError(details)


class TestTargets:
    async def test_register_target_assigns_version_when_omitted(self):
        svc = ExperimentationServiceImpl()
        first = await svc.RegisterTarget(
            experiments_pb2.RegisterTargetRequest(name="p", version=0, type=experiments_pb2.PROMPT),
            FakeContext(),
        )
        second = await svc.RegisterTarget(
            experiments_pb2.RegisterTargetRequest(name="p", version=0, type=experiments_pb2.PROMPT),
            FakeContext(),
        )
        assert first.version == 1
        assert second.version == 2

    async def test_get_target_history_orders_versions(self):
        svc = ExperimentationServiceImpl()
        for v in (3, 1, 2):
            await svc.RegisterTarget(
                experiments_pb2.RegisterTargetRequest(name="p", version=v),
                FakeContext(),
            )
        history = await svc.GetTargetHistory(
            experiments_pb2.GetTargetHistoryRequest(name="p"), FakeContext()
        )
        assert [t.version for t in history.versions] == [1, 2, 3]


class TestRunEvaluationStream:
    async def test_streams_progress_and_finishes_completed(self):
        svc = ExperimentationServiceImpl()
        await svc.RegisterTarget(
            experiments_pb2.RegisterTargetRequest(name="p", version=1), FakeContext()
        )
        ds_req = experiments_pb2.CreateDatasetRequest(name="ds")
        case = ds_req.test_cases.add()
        case.id = "tc-1"
        case.input_query = "?"
        case.ideal_response = "answer"
        case.key_elements.append("answer")
        await svc.CreateDataset(ds_req, FakeContext())
        progresses = [
            m
            async for m in svc.RunEvaluation(
                experiments_pb2.RunEvaluationRequest(
                    dataset_name="ds", target_ids=["p:1"], metrics=["key_elements"]
                ),
                FakeContext(),
            )
        ]
        assert progresses
        assert progresses[-1].status == "completed"
        assert progresses[-1].results.target_results[0].metric_scores["key_elements"] == 1.0


class TestExperimentLifecycle:
    async def test_assign_then_record_outcome_then_get_results(self):
        svc = ExperimentationServiceImpl()
        exp = experiments_pb2.Experiment(
            name="my-exp",
            workflow_id="wf-1",
            success_metrics=["score"],
            minimum_sample_size=2,
        )
        c = exp.variants.add()
        c.name = "control"
        c.traffic_allocation = 0.5
        t = exp.variants.add()
        t.name = "treatment"
        t.traffic_allocation = 0.5
        await svc.CreateExperiment(
            experiments_pb2.CreateExperimentRequest(experiment=exp), FakeContext()
        )

        # Two assignments → record outcomes for both → get results
        a1 = await svc.AssignVariant(
            experiments_pb2.AssignVariantRequest(experiment_name="my-exp", assignment_key="u-1"),
            FakeContext(),
        )
        a2 = await svc.AssignVariant(
            experiments_pb2.AssignVariantRequest(experiment_name="my-exp", assignment_key="u-2"),
            FakeContext(),
        )
        for a, value in ((a1, 1.0), (a2, 0.5)):
            await svc.RecordOutcome(
                experiments_pb2.RecordOutcomeRequest(
                    experiment_name="my-exp",
                    assignment_id=a.assignment_id,
                    outcomes={"score": value},
                ),
                FakeContext(),
            )
        results = await svc.GetExperimentResults(
            experiments_pb2.GetExperimentResultsRequest(experiment_name="my-exp"),
            FakeContext(),
        )
        assert results.experiment_name == "my-exp"
        assert sum(vs.sample_size for vs in results.variant_summaries) == 2
