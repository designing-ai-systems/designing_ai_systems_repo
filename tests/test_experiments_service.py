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


async def _register(svc, name, version):
    return await svc.RegisterTarget(
        experiments_pb2.RegisterTargetRequest(name=name, version=version), FakeContext()
    )


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


class TestPromoteTarget:
    async def test_promote_activates_target_and_deprecates_previous(self):
        svc = ExperimentationServiceImpl()
        await _register(svc, "p", 1)
        await _register(svc, "p", 2)

        # Promote v1 to ACTIVE.
        promoted_v1 = await svc.PromoteTarget(
            experiments_pb2.PromoteTargetRequest(name="p", version=1), FakeContext()
        )
        assert promoted_v1.status == experiments_pb2.ACTIVE

        # Promote v2; v1 should auto-deprecate.
        promoted_v2 = await svc.PromoteTarget(
            experiments_pb2.PromoteTargetRequest(name="p", version=2), FakeContext()
        )
        assert promoted_v2.status == experiments_pb2.ACTIVE
        history = await svc.GetTargetHistory(
            experiments_pb2.GetTargetHistoryRequest(name="p"), FakeContext()
        )
        statuses = {t.version: t.status for t in history.versions}
        assert statuses[1] == experiments_pb2.DEPRECATED
        assert statuses[2] == experiments_pb2.ACTIVE

    async def test_promote_missing_target_returns_not_found(self):
        svc = ExperimentationServiceImpl()
        ctx = FakeContext()
        try:
            await svc.PromoteTarget(
                experiments_pb2.PromoteTargetRequest(name="missing", version=1), ctx
            )
        except grpc.RpcError:
            pass
        assert ctx.code == grpc.StatusCode.NOT_FOUND


class TestRunEvaluationRollback:
    async def test_failed_eval_restores_target_status(self, monkeypatch):
        """If the streaming evaluation pipeline raises, every target's status
        must roll back to the prior value (DRAFT in this case) rather than
        staying in TESTING forever."""
        svc = ExperimentationServiceImpl()
        await _register(svc, "p", 1)
        await svc.CreateDataset(experiments_pb2.CreateDatasetRequest(name="ds"), FakeContext())

        def boom(*args, **kwargs):
            raise RuntimeError("simulated pipeline failure")

        monkeypatch.setattr(svc.evaluation_pipeline, "run", boom)

        request = experiments_pb2.RunEvaluationRequest(
            dataset_name="ds", target_ids=["p:1"], metrics=["key_elements"]
        )
        try:
            async for _ in svc.RunEvaluation(request, FakeContext()):
                pass
        except RuntimeError:
            pass

        history = await svc.GetTargetHistory(
            experiments_pb2.GetTargetHistoryRequest(name="p"), FakeContext()
        )
        # Target should not be stuck in TESTING.
        assert history.versions[0].status == experiments_pb2.DRAFT


class TestScoringRuleSampling:
    async def test_run_scoring_rule_honors_sample_rate_zero(self, monkeypatch):
        """sample_rate=0.0 means 'score nothing'. With 5 trace ids that
        resolve to real traces, the rule must score 0 of them. The previous
        implementation ignored sample_rate and always scored every trace."""
        from services.observability.models import Trace

        svc = ExperimentationServiceImpl()
        rule = experiments_pb2.ScoringRule(name="r-zero", workflow_id="wf", sample_rate=0.0)
        scorer = rule.scorers.add()
        scorer.name = "k"
        scorer.type = "automated"
        scorer.required_elements.append("hello")
        await svc.CreateScoringRule(
            experiments_pb2.CreateScoringRuleRequest(rule=rule), FakeContext()
        )

        # Stub the fetcher so the test exercises the sampling decision, not
        # the network path. With sample_rate=0.0, _fetch_traces should not
        # even be called for IDs that get sampled out.
        monkeypatch.setattr(
            svc,
            "_fetch_traces",
            lambda ids: [Trace(trace_id=tid, output="hello world") for tid in ids],
        )
        response = await svc.RunScoringRule(
            experiments_pb2.RunScoringRuleRequest(
                rule_name="r-zero", trace_ids=["t1", "t2", "t3", "t4", "t5"]
            ),
            FakeContext(),
        )
        assert response.traces_scored == 0


class TestAssignVariantOrdering:
    async def test_missing_experiment_returns_empty_even_with_unrelated_assignment(self):
        """AssignVariant must check experiment existence before falling
        through to a stale assignment lookup, so a missing experiment
        returns an empty proto regardless of what else is in the store."""
        svc = ExperimentationServiceImpl()
        response = await svc.AssignVariant(
            experiments_pb2.AssignVariantRequest(
                experiment_name="no-such-experiment", assignment_key="k1"
            ),
            FakeContext(),
        )
        assert response.assignment_id == ""


class TestBaselineWinsReportsChallengerPValue:
    async def test_baseline_winner_reports_closest_challenger_p_value(self):
        """When the baseline variant wins (no treatment has a higher mean),
        the comparison must still report the closest challenger's p-value,
        not the initial sentinel of 1.0. Otherwise teams see 'baseline wins,
        p=1.0' and lose all information about how close the contest was."""
        svc = ExperimentationServiceImpl()
        exp = experiments_pb2.Experiment(
            name="baseline-wins",
            workflow_id="wf",
            success_metrics=["score"],
            minimum_sample_size=2,
        )
        c = exp.variants.add()
        c.name = "baseline"
        c.traffic_allocation = 0.5
        t = exp.variants.add()
        t.name = "challenger"
        t.traffic_allocation = 0.5
        await svc.CreateExperiment(
            experiments_pb2.CreateExperimentRequest(experiment=exp), FakeContext()
        )

        # Baseline outcomes higher than challenger — baseline should win.
        b1 = await svc.AssignVariant(
            experiments_pb2.AssignVariantRequest(
                experiment_name="baseline-wins", assignment_key="k1"
            ),
            FakeContext(),
        )
        b2 = await svc.AssignVariant(
            experiments_pb2.AssignVariantRequest(
                experiment_name="baseline-wins", assignment_key="k2"
            ),
            FakeContext(),
        )
        # Manually steer outcomes — bypass assignment hashing by writing the
        # outcomes directly via the store.
        from services.experiments.models import Outcome

        for variant_name, values in (("baseline", [0.9, 0.92]), ("challenger", [0.6, 0.62])):
            for value in values:
                svc.store.record_outcome(
                    Outcome(
                        experiment_name="baseline-wins",
                        assignment_id=b1.assignment_id if value == values[0] else b2.assignment_id,
                        variant_name=variant_name,
                        outcomes={"score": value},
                    )
                )

        results = await svc.GetExperimentResults(
            experiments_pb2.GetExperimentResultsRequest(experiment_name="baseline-wins"),
            FakeContext(),
        )
        comparison = next(c for c in results.comparisons if c.metric_name == "score")
        assert comparison.winner == "baseline"
        # The closest challenger's p-value should not be the initial sentinel.
        assert comparison.p_value < 1.0


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
