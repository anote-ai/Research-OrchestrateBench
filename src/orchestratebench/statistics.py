"""Statistical rigor utilities for orchestratebench.

Bootstrap confidence intervals and paired significance tests for evaluation
metrics. Multi-agent orchestration benchmarks are high-variance — LLM
non-determinism compounds across agents and cascade failures are rare,
high-variance events — so point estimates alone are insufficient. Reviewers
will ask whether differences between frameworks are statistically significant
and how many runs back each estimate.

All resampling is at the *scenario level* (the whole ``ExecutionTrace`` list is
resampled), not at the individual agent-step level, matching the benchmark's
statistical-rigor requirements. Pure NumPy — no SciPy dependency.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np

from .core import ExecutionTrace

# A metric maps a list of traces to a single scalar (e.g. ``success_rate``).
MetricFn = Callable[[List[ExecutionTrace]], float]


def bootstrap_ci(
    values: List[float],
    n_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean of ``values``.

    Resamples ``values`` with replacement ``n_resamples`` times and returns the
    empirical ``(low, high)`` percentiles of the resampled means at the given
    confidence level. Deterministic for a fixed ``seed``.
    """
    if not values:
        return (0.0, 0.0)
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    n = arr.shape[0]
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = arr[idx].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low = float(np.percentile(means, alpha * 100.0))
    high = float(np.percentile(means, (1.0 - alpha) * 100.0))
    return (low, high)


def metric_ci(
    traces: List[ExecutionTrace],
    metric_fn: MetricFn,
    n_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Point estimate plus bootstrap CI for a trace-level metric.

    Resamples whole traces (scenario level) and recomputes ``metric_fn`` on each
    resample, so the CI reflects scenario-to-scenario variability. Returns
    ``(point, low, high)``. Deterministic for a fixed ``seed``.
    """
    point = metric_fn(traces)
    if not traces:
        return (point, point, point)
    rng = np.random.default_rng(seed)
    n = len(traces)
    stats = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        stats[i] = metric_fn([traces[j] for j in idx])
    alpha = (1.0 - confidence) / 2.0
    low = float(np.percentile(stats, alpha * 100.0))
    high = float(np.percentile(stats, (1.0 - alpha) * 100.0))
    return (point, low, high)


def paired_bootstrap_values(
    values_a: List[float],
    values_b: List[float],
    n_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Dict[str, object]:
    """Paired bootstrap test for the mean difference of two numeric vectors.

    ``values_a[i]`` and ``values_b[i]`` must refer to the same paired scenario.
    Returns ``{"mean_a", "mean_b", "diff", "ci", "p_value"}``.
    Deterministic for a fixed ``seed``.
    """
    if len(values_a) != len(values_b):
        raise ValueError(
            f"paired bootstrap needs equal-length value lists; "
            f"got {len(values_a)} vs {len(values_b)}"
        )
    if not values_a:
        return {
            "mean_a": 0.0,
            "mean_b": 0.0,
            "diff": 0.0,
            "ci": (0.0, 0.0),
            "p_value": 1.0,
        }

    arr_a = np.asarray(values_a, dtype=float)
    arr_b = np.asarray(values_b, dtype=float)
    diffs = arr_a - arr_b
    diff = float(diffs.mean())
    mean_a = float(arr_a.mean())
    mean_b = float(arr_b.mean())

    rng = np.random.default_rng(seed)
    n = arr_a.shape[0]
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_diffs = (arr_a[idx] - arr_b[idx]).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low = float(np.percentile(boot_diffs, alpha * 100.0))
    high = float(np.percentile(boot_diffs, (1.0 - alpha) * 100.0))
    if diff >= 0.0:
        p_value = float(np.mean(boot_diffs <= 0.0)) * 2.0
    else:
        p_value = float(np.mean(boot_diffs >= 0.0)) * 2.0
    return {
        "mean_a": mean_a,
        "mean_b": mean_b,
        "diff": diff,
        "ci": (low, high),
        "p_value": min(p_value, 1.0),
    }


def paired_bootstrap_test(
    traces_a: List[ExecutionTrace],
    traces_b: List[ExecutionTrace],
    metric_fn: MetricFn,
    n_resamples: int = 10000,
    seed: int = 0,
) -> Dict[str, object]:
    """Paired bootstrap test for ``metric_fn(a) - metric_fn(b)``.

    The two lists must be equal length and paired by scenario: index ``i`` of
    each list is the same scenario evaluated under two policies. The same
    resampled scenario indices are applied to both sides every iteration, so
    scenario difficulty cancels out and only the policy effect remains.

    Returns ``{"diff", "ci", "p_value"}`` where ``p_value`` is the two-sided
    probability of observing a difference at least this extreme under the null
    hypothesis of no difference. Deterministic for a fixed ``seed``.
    """
    if len(traces_a) != len(traces_b):
        raise ValueError(
            f"paired test needs equal-length trace lists; "
            f"got {len(traces_a)} vs {len(traces_b)}"
        )
    diff = metric_fn(traces_a) - metric_fn(traces_b)
    if not traces_a:
        return {"diff": diff, "ci": (0.0, 0.0), "p_value": 1.0}
    rng = np.random.default_rng(seed)
    n = len(traces_a)
    diffs = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        diffs[i] = metric_fn([traces_a[j] for j in idx]) - metric_fn(
            [traces_b[j] for j in idx]
        )
    low = float(np.percentile(diffs, 2.5))
    high = float(np.percentile(diffs, 97.5))
    # Two-sided p-value: fraction of resampled diffs on the opposite side of
    # zero from the observed diff, doubled and clamped to 1.0.
    if diff >= 0.0:
        p_value = float(np.mean(diffs <= 0.0)) * 2.0
    else:
        p_value = float(np.mean(diffs >= 0.0)) * 2.0
    return {"diff": diff, "ci": (low, high), "p_value": min(p_value, 1.0)}
