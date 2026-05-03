"""
Quickstart: platform.experiments — improvement loop end to end.

Listings 7.16, 7.17, 7.18, 7.20 stitched together so a workflow author
can copy/paste a complete A/B test against an offline dataset.

Run:  python examples/quickstart_experiments.py
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import grpc

from genai_platform import GenAIPlatform
from proto import observability_pb2_grpc
from services.experiments.service import ExperimentationServiceImpl
from services.gateway.main import main as start_gateway
from services.observability.service import ObservabilityServiceImpl
from services.shared.server import run_aio_service_main


def start_in_thread(target, name):
    threading.Thread(target=target, daemon=True, name=name).start()


def start_observability():
    run_aio_service_main("observability", ObservabilityServiceImpl)


def start_experiments():
    obs_addr = os.getenv("OBSERVABILITY_SERVICE_ADDR", "localhost:50059")
    channel = grpc.insecure_channel(obs_addr)
    obs_stub = observability_pb2_grpc.ObservabilityServiceStub(channel)
    run_aio_service_main(
        "experiments",
        lambda: ExperimentationServiceImpl(observability_stub=obs_stub),
    )


def step_register(platform: GenAIPlatform):
    print("\n[Listing 7.16] register prompt v3")
    target = platform.experiments.register_target(
        name="patient-intake-v2",
        version=3,
        target_type="PROMPT",
        change_description="add insurance confirmation",
        author="sarah",
    )
    print(f"  target {target.name}:{target.version}  status={target.status}")
    return target


def step_dataset(platform: GenAIPlatform):
    print("\n[Listing 7.17] curated test cases")
    dataset = platform.experiments.create_dataset(
        name="intake-bench",
        test_cases=[
            {
                "id": "doc-list",
                "input_query": "What documents do I need?",
                "ideal_response": "Bring your insurance card and your photo ID.",
                "key_elements": ["insurance", "photo", "id"],
            },
            {
                "id": "appt-time",
                "input_query": "When is my appointment?",
                "ideal_response": "Your appointment with Dr Patel is at 10:30 AM Tuesday.",
                "key_elements": ["dr", "patel", "10:30", "tuesday"],
            },
        ],
    )
    print(f"  dataset '{dataset.name}' has {len(dataset.test_cases)} curated cases")


def step_run_evaluation(platform: GenAIPlatform):
    print("\n[Listing 7.16] offline evaluation against the curated dataset")
    final = None
    for progress in platform.experiments.run_evaluation(
        dataset_name="intake-bench",
        targets=["patient-intake-v2:3"],
        metrics=["key_elements"],
    ):
        final = progress
    if final and final.results.target_results:
        tr = final.results.target_results[0]
        print(f"  {tr.target_id} overall_score={tr.overall_score:.2f}")
        for metric, score in tr.metric_scores.items():
            print(f"     {metric}: {score:.2f}")


def step_scoring_rule(platform: GenAIPlatform):
    print("\n[Listing 7.18] online scoring rule (production traffic monitor)")
    rule = platform.experiments.create_scoring_rule(
        name="intake-quality-monitor",
        workflow_id="patient-intake",
        sample_rate=0.1,
        scorers=[
            {"name": "key_elements", "type": "key_elements", "required_elements": ["insurance"]},
        ],
        alert_on={"key_elements": {"below": 0.8, "window": "1h"}},
    )
    print(
        f"  rule '{rule.name}' active for workflow "
        f"'{rule.workflow_id}'  sample_rate={rule.sample_rate}"
    )


def step_experiment(platform: GenAIPlatform):
    print("\n[Listing 7.20] experiment-aware workflow: assign + record + read")
    platform.experiments.create_experiment(
        name="intake-prompt-ab",
        workflow_id="patient-intake",
        variants=[
            {
                "name": "control",
                "traffic_allocation": 0.5,
                "prompt_variant": {"prompt_name": "patient-intake-v2", "version": 1},
            },
            {
                "name": "treatment",
                "traffic_allocation": 0.5,
                "prompt_variant": {"prompt_name": "patient-intake-v2", "version": 3},
            },
        ],
        success_metrics=["resolved"],
        minimum_sample_size=10,
    )
    counts = {"control": 0, "treatment": 0, None: 0}
    for i in range(20):
        assignment = platform.experiments.assign_variant(
            experiment_name="intake-prompt-ab",
            assignment_key=f"patient-{i}",
            workflow_id="patient-intake",
        )
        if assignment is None:
            counts[None] += 1
        else:
            counts[assignment.variant.name] += 1
            score = 0.92 if assignment.variant.name == "treatment" else 0.65
            platform.experiments.record_outcome(
                experiment_name="intake-prompt-ab",
                assignment_id=assignment.assignment_id,
                outcomes={"resolved": score},
            )
    print(f"  variant split over 20 patients: {counts}")
    results = platform.experiments.get_experiment_results(experiment_name="intake-prompt-ab")
    for vs in results.variant_summaries:
        means = ", ".join(f"{m.metric_name}={m.mean:.3f}" for m in vs.metrics)
        print(f"   {vs.variant_name}: n={vs.sample_size}  {means}")
    for c in results.comparisons:
        print(
            f"   comparison {c.metric_name}: winner={c.winner} "
            f"effect={c.effect_size:.3f} p={c.p_value:.3f}"
        )
    print(f"  ready_to_conclude={results.ready_to_conclude}")


def main():
    print("=" * 60)
    print("  Quickstart: platform.experiments")
    print("=" * 60)
    os.environ.setdefault("OBSERVABILITY_SERVICE_ADDR", "localhost:50059")
    print("\nStarting Observability + Experimentation services and Gateway...")
    start_in_thread(start_observability, "ObservabilityService")
    time.sleep(1)
    start_in_thread(start_experiments, "ExperimentsService")
    time.sleep(1)
    start_in_thread(start_gateway, "Gateway")
    time.sleep(1)
    print("Services ready.")

    platform = GenAIPlatform()
    try:
        step_register(platform)
        step_dataset(platform)
        step_run_evaluation(platform)
        step_scoring_rule(platform)
        step_experiment(platform)
        print("\n" + "=" * 60)
        print("  Quickstart complete")
        print("=" * 60)
    except Exception as e:  # noqa: BLE001
        import traceback

        print(f"\nError: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
