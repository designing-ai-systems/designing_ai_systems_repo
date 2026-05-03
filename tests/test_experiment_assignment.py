"""Unit tests for A/B testing assignment + statistics (Listing 7.20)."""

from services.experiments.ab_testing import assign_variant, welch_t_test
from services.experiments.models import Experiment, Variant


def _make_experiment(allocation=(0.5, 0.5)):
    return Experiment(
        name="exp-1",
        variants=[
            Variant(name="control", traffic_allocation=allocation[0]),
            Variant(name="treatment", traffic_allocation=allocation[1]),
        ],
    )


class TestAssignVariant:
    def test_consistent_assignment(self):
        exp = _make_experiment()
        first = assign_variant(exp, "patient-12345")
        for _ in range(50):
            assert assign_variant(exp, "patient-12345").name == first.name

    def test_traffic_allocation_respected(self):
        exp = _make_experiment(allocation=(0.7, 0.3))
        counts = {"control": 0, "treatment": 0}
        for i in range(2000):
            counts[assign_variant(exp, f"k-{i}").name] += 1
        # Allow a 5pp tolerance.
        assert 0.65 <= counts["control"] / 2000 <= 0.75
        assert 0.25 <= counts["treatment"] / 2000 <= 0.35

    def test_unallocated_returns_none(self):
        exp = _make_experiment(allocation=(0.0, 0.0))
        assert assign_variant(exp, "any") is None

    def test_no_experiment_returns_none(self):
        assert assign_variant(None, "any") is None


class TestWelchTTest:
    def test_returns_zero_for_tiny_samples(self):
        t, p, diff = welch_t_test([1.0], [2.0])
        assert t == 0.0
        assert p == 1.0
        assert diff == 0.0

    def test_positive_effect_size_when_a_higher(self):
        a = [0.9, 0.85, 0.92, 0.88, 0.91, 0.87, 0.95, 0.93, 0.89, 0.9]
        b = [0.5, 0.55, 0.48, 0.52, 0.5, 0.49, 0.51, 0.5, 0.47, 0.53]
        _, p_value, diff = welch_t_test(a, b)
        assert diff > 0.3
        assert p_value < 0.05
