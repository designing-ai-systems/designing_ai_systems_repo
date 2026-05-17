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

    def test_uses_t_distribution_for_small_samples(self):
        """A two-sample test on n=2 per arm should yield a p-value around
        0.10, not below 0.05. The previous normal-CDF approximation reported
        p ~ 0.005 here, declaring an effect "significant" when the t-test
        (which accounts for degrees of freedom) does not.

        Reference (scipy.stats.ttest_ind(equal_var=False)):
            t = -2.828, df = 2.0, two-tailed p ≈ 0.1056
        """
        a = [1.0, 1.5]
        b = [2.0, 2.5]
        t_stat, p_value, diff = welch_t_test(a, b)
        assert abs(diff - (-1.0)) < 1e-9
        assert abs(t_stat + 2.828) < 0.01
        # Normal-CDF approximation gave p ≈ 0.0047; the real t-test gives 0.106.
        assert 0.08 <= p_value <= 0.13

    def test_t_distribution_is_more_conservative_than_normal(self):
        """The t-distribution has heavier tails than the standard normal, so
        the same t-statistic yields a strictly larger p-value under the
        t-distribution than under the normal CDF — especially for small n.

        For a small sample where the previous normal-CDF approximation gave
        p ≈ 0.005 (significant), the proper Welch's test gives p ≈ 0.106
        (not significant). The fix prevents that false-positive significance.
        """
        # n_a = n_b = 2, t = -2.828, df = 2.0
        _, p_t, _ = welch_t_test([1.0, 1.5], [2.0, 2.5])
        # Under the t-distribution this p-value lives near 0.106, not below 0.05.
        assert p_t > 0.05
