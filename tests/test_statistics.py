"""Tests for orchestratebench.statistics."""

from __future__ import annotations

from orchestratebench.data import make_benchmark_tasks, make_execution_trace
from orchestratebench.evaluate import mean_cost, policy_comparison, success_rate
from orchestratebench.statistics import (
    bootstrap_ci,
    metric_ci,
    paired_bootstrap_test,
)

import pytest


def _make_traces(n: int = 5, success: bool = True):
    tasks = make_benchmark_tasks(n=n, seed=1)
    return [make_execution_trace(t, success=success, seed=i) for i, t in enumerate(tasks)]


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------


def test_bootstrap_ci_constant_values_degenerate() -> None:
    low, high = bootstrap_ci([1.0] * 50)
    assert low == pytest.approx(1.0)
    assert high == pytest.approx(1.0)


def test_bootstrap_ci_deterministic_with_seed() -> None:
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert bootstrap_ci(vals, seed=42) == bootstrap_ci(vals, seed=42)


def test_bootstrap_ci_brackets_mean() -> None:
    low, high = bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0], seed=0)  # mean 3.0
    assert low <= 3.0 <= high


def test_bootstrap_ci_empty_returns_zeros() -> None:
    assert bootstrap_ci([]) == (0.0, 0.0)


def test_bootstrap_ci_wider_data_gives_wider_interval() -> None:
    narrow = bootstrap_ci([3.0, 3.0, 3.0, 3.1, 2.9], seed=0)
    wide = bootstrap_ci([0.0, 6.0, 1.0, 5.0, 3.0], seed=0)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


# ---------------------------------------------------------------------------
# metric_ci
# ---------------------------------------------------------------------------


def test_metric_ci_point_inside_interval() -> None:
    traces = _make_traces(20, success=True)
    point, low, high = metric_ci(traces, success_rate, seed=0)
    assert point == 1.0
    assert low <= point <= high


def test_metric_ci_deterministic_with_seed() -> None:
    traces = _make_traces(20)
    assert metric_ci(traces, mean_cost, seed=7) == metric_ci(traces, mean_cost, seed=7)


def test_metric_ci_empty_returns_point_thrice() -> None:
    assert metric_ci([], success_rate) == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# paired_bootstrap_test
# ---------------------------------------------------------------------------


def test_paired_identical_traces_no_difference() -> None:
    traces = _make_traces(20)
    res = paired_bootstrap_test(traces, traces, success_rate, seed=0)
    assert res["diff"] == pytest.approx(0.0)
    assert res["p_value"] == pytest.approx(1.0)


def test_paired_clear_difference_is_significant() -> None:
    a = _make_traces(20, success=True)   # success_rate 1.0
    b = _make_traces(20, success=False)  # success_rate 0.0
    res = paired_bootstrap_test(a, b, success_rate, seed=0)
    assert res["diff"] == pytest.approx(1.0)
    assert res["p_value"] < 0.05


def test_paired_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="equal-length"):
        paired_bootstrap_test(_make_traces(3), _make_traces(4), success_rate)


def test_paired_deterministic_with_seed() -> None:
    a = _make_traces(20, success=True)
    b = _make_traces(20, success=False)
    assert paired_bootstrap_test(a, b, success_rate, seed=3) == paired_bootstrap_test(
        a, b, success_rate, seed=3
    )


# ---------------------------------------------------------------------------
# policy_comparison(with_ci=...) — backward-compatible opt-in
# ---------------------------------------------------------------------------


def test_policy_comparison_without_ci_is_unchanged() -> None:
    result = policy_comparison({"fixed": _make_traces(10)})
    assert "success_rate" in result["fixed"]
    assert "success_rate_ci" not in result["fixed"]


def test_policy_comparison_with_ci_adds_intervals() -> None:
    result = policy_comparison({"fixed": _make_traces(10)}, with_ci=True, n_resamples=500, seed=0)
    fixed = result["fixed"]
    assert "success_rate_ci" in fixed
    low, high = fixed["success_rate_ci"]
    assert low <= fixed["success_rate"] <= high
