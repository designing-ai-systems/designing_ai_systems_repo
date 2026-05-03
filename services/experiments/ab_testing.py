"""
A/B testing utilities — consistent hashing + Welch's t-test.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 7.20: experiment-aware workflow that uses consistent assignment
"""

from __future__ import annotations

import hashlib
import math
from typing import Optional, Sequence, Tuple

from services.experiments.models import Experiment, Variant

_BUCKET_RESOLUTION = 10_000  # 0.0001 precision


def _bucket(experiment_name: str, assignment_key: str) -> int:
    """Map (experiment, key) into [0, _BUCKET_RESOLUTION)."""
    digest = hashlib.md5(f"{experiment_name}|{assignment_key}".encode("utf-8")).digest()
    # Take the first 8 bytes as an unsigned int and modulo it.
    raw = int.from_bytes(digest[:8], "big", signed=False)
    return raw % _BUCKET_RESOLUTION


def assign_variant(
    experiment: Experiment,
    assignment_key: str,
) -> Optional[Variant]:
    """Deterministically pick a variant for ``assignment_key``.

    The same (experiment, key) pair always returns the same variant.
    Variants whose ``traffic_allocation`` sums to less than 1.0 leave a
    "control" gap; keys that hash into the gap return ``None`` so the
    caller falls through to the default behaviour.
    """
    if experiment is None or not experiment.variants:
        return None
    bucket = _bucket(experiment.name, assignment_key)
    fraction = bucket / _BUCKET_RESOLUTION
    cumulative = 0.0
    for variant in experiment.variants:
        cumulative += max(0.0, variant.traffic_allocation)
        if fraction < cumulative:
            return variant
    return None


# ---------------------------------------------------------------------------
# Welch's t-test (two-sample, unequal variance)
# ---------------------------------------------------------------------------


def welch_t_test(
    sample_a: Sequence[float],
    sample_b: Sequence[float],
) -> Tuple[float, float, float]:
    """Return ``(t_statistic, p_value, effect_size_diff_means)``.

    Falls back to ``(0.0, 1.0, 0.0)`` for samples too small to compute.
    The p-value is two-tailed and uses the standard-normal approximation
    for the t-distribution (good enough for n>=30 per arm; acceptable
    for the chapter-7 demo data).
    """
    if len(sample_a) < 2 or len(sample_b) < 2:
        return 0.0, 1.0, 0.0

    mean_a = sum(sample_a) / len(sample_a)
    mean_b = sum(sample_b) / len(sample_b)
    var_a = sum((x - mean_a) ** 2 for x in sample_a) / (len(sample_a) - 1)
    var_b = sum((x - mean_b) ** 2 for x in sample_b) / (len(sample_b) - 1)
    se = math.sqrt(var_a / len(sample_a) + var_b / len(sample_b))
    if se == 0:
        return 0.0, 1.0, mean_a - mean_b

    t_stat = (mean_a - mean_b) / se
    p_value = 2.0 * (1.0 - _standard_normal_cdf(abs(t_stat)))
    return t_stat, max(0.0, min(1.0, p_value)), mean_a - mean_b


def _standard_normal_cdf(z: float) -> float:
    """Standard-normal CDF approximation via the error function."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
