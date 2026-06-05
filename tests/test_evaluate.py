"""Tests for orchestratebench evaluate."""

from __future__ import annotations

from orchestratebench.core import TaskStatus
from orchestratebench.data import (
    make_benchmark_tasks,
    make_execution_trace,
    make_task,
)
from orchestratebench.evaluate import (
    confidence_calibration,
    mean_cost,
    mean_latency,
    mean_retry_rate,
    orchestration_efficiency_score,
    policy_comparison,
    routing_accuracy,
    routing_distribution,
    skip_rate,
    success_rate,
    task_dependency_score,
    throughput_score,
)


def _make_traces(n: int = 5, success: bool = True):
    tasks = make_benchmark_tasks(n=n, seed=1)
    return [make_execution_trace(t, success=success, seed=i) for i, t in enumerate(tasks)]


def test_success_rate_all_success() -> None:
    traces = _make_traces(5, success=True)
    assert success_rate(traces) == 1.0


def test_success_rate_all_fail() -> None:
    traces = _make_traces(5, success=False)
    assert success_rate(traces) == 0.0


def test_success_rate_empty() -> None:
    assert success_rate([]) == 0.0


def test_mean_latency_positive() -> None:
    traces = _make_traces(5)
    assert mean_latency(traces) > 0


def test_mean_cost_positive() -> None:
    traces = _make_traces(5)
    assert mean_cost(traces) > 0


def test_routing_accuracy_perfect() -> None:
    assert routing_accuracy(["a", "b", "c"], ["a", "b", "c"]) == 1.0


def test_routing_accuracy_empty() -> None:
    assert routing_accuracy([], []) == 0.0


def test_routing_distribution_counts() -> None:
    traces = _make_traces(3)
    dist = routing_distribution(traces)
    assert sum(dist.values()) == 3


def test_orchestration_efficiency_score_perfect() -> None:
    traces = _make_traces(5, success=True)
    # All traces succeed; latency is in [100, 3000] — some may exceed 2000ms budget
    score = orchestration_efficiency_score(traces, latency_budget_ms=5000.0, cost_budget_usd=1.0)
    assert score == 1.0


def test_orchestration_efficiency_score_all_fail() -> None:
    traces = _make_traces(5, success=False)
    score = orchestration_efficiency_score(traces)
    assert score == 0.0


def test_orchestration_efficiency_score_empty() -> None:
    assert orchestration_efficiency_score([]) == 0.0


def test_task_dependency_score_no_deps() -> None:
    traces = _make_traces(4)
    # No dependencies set
    score = task_dependency_score(traces)
    assert score == 1.0


def test_task_dependency_score_with_deps() -> None:
    traces = _make_traces(3)
    traces[0].dependencies_declared = ["dep-a", "dep-b"]
    traces[0].dependencies_resolved = ["dep-a"]
    score = task_dependency_score(traces)
    assert score == (0.5 + 1.0 + 1.0) / 3


def test_task_dependency_score_skipped_trace_is_penalized() -> None:
    traces = _make_traces(2)
    traces[1].status = TaskStatus.SKIPPED
    traces[1].dependencies_declared = ["dep-a", "dep-b"]
    traces[1].dependencies_resolved = ["dep-a"]
    score = task_dependency_score(traces)
    assert score == 0.75


def test_mean_retry_rate_zero() -> None:
    traces = _make_traces(4)
    assert mean_retry_rate(traces) == 0.0


def test_skip_rate_zero() -> None:
    traces = _make_traces(4)
    assert skip_rate(traces) == 0.0


def test_throughput_score_positive() -> None:
    traces = _make_traces(5)
    score = throughput_score(traces)
    assert score > 0.0


def test_confidence_calibration_range() -> None:
    traces = _make_traces(5)
    cal = confidence_calibration(traces)
    assert cal is not None
    assert 0.0 <= cal <= 1.0


def test_policy_comparison_includes_new_metrics() -> None:
    traces = _make_traces(4)
    result = policy_comparison({"fixed": traces})
    assert "orchestration_efficiency_score" in result["fixed"]
    assert "task_dependency_score" in result["fixed"]
