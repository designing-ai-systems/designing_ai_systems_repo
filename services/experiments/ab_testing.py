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

    Implements the two-sample Welch's t-test for unequal variances:
        t = (mean_a - mean_b) / sqrt(var_a/n_a + var_b/n_b)
    with the Welch-Satterthwaite approximation for the degrees of freedom
    and the regularized incomplete beta function for the t-distribution
    survival probability. Falls back to ``(0.0, 1.0, 0.0)`` when either
    sample has fewer than two observations.
    """
    n_a = len(sample_a)
    n_b = len(sample_b)
    if n_a < 2 or n_b < 2:
        return 0.0, 1.0, 0.0

    mean_a = sum(sample_a) / n_a
    mean_b = sum(sample_b) / n_b
    diff = mean_a - mean_b

    var_a = sum((x - mean_a) ** 2 for x in sample_a) / (n_a - 1)
    var_b = sum((x - mean_b) ** 2 for x in sample_b) / (n_b - 1)
    se_sq = var_a / n_a + var_b / n_b
    if se_sq <= 0.0:
        return 0.0, 1.0, diff

    t_stat = diff / math.sqrt(se_sq)

    # Welch-Satterthwaite degrees of freedom.
    df_num = se_sq * se_sq
    df_denom = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    if df_denom <= 0.0:
        return t_stat, 1.0, diff
    df = df_num / df_denom

    # Two-tailed p-value via the regularized incomplete beta function:
    #   Pr(|T| >= |t|, df) = I_{df / (df + t^2)} (df / 2, 1 / 2)
    x = df / (df + t_stat * t_stat)
    p_value = _regularized_incomplete_beta(df / 2.0, 0.5, x)
    return t_stat, max(0.0, min(1.0, p_value)), diff


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """Return I_x(a, b), the regularized incomplete beta function.

    Numerical Recipes 6.4: combines the closed-form prefactor with a
    continued-fraction expansion that converges quickly across the
    typical t-test parameter range.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_prefactor = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    prefactor = math.exp(log_prefactor)
    # Choose the branch that converges fastest.
    if x < (a + 1.0) / (a + b + 2.0):
        return prefactor * _beta_continued_fraction(a, b, x) / a
    return 1.0 - prefactor * _beta_continued_fraction(b, a, 1.0 - x) / b


def _beta_continued_fraction(
    a: float,
    b: float,
    x: float,
    max_iter: int = 200,
    eps: float = 3e-12,
) -> float:
    """Lentz's algorithm for the continued-fraction part of I_x(a, b)."""
    tiny = 1e-30
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    return h
