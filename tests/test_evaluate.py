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
    per_class_routing_metrics,
    policy_comparison,
    routing_accuracy,
    routing_confusion_matrix,
    routing_distribution,
    routing_macro_f1,
    skip_rate,
    success_rate,
    task_dependency_score,
    throughput_score,
)

import pytest


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
    traces[0].dependencies_resolved = ["dep-a", "dep-b"]
    score = task_dependency_score(traces)
    assert 0.0 <= score <= 1.0


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


# ---------------------------------------------------------------------------
# Per-class routing metrics
# ---------------------------------------------------------------------------


def test_per_class_routing_metrics_perfect_prediction() -> None:
    pred = ["direct_tool", "decompose", "direct_tool", "code_execution"]
    ref = ["direct_tool", "decompose", "direct_tool", "code_execution"]
    metrics = per_class_routing_metrics(pred, ref)
    for cls in ("direct_tool", "decompose", "code_execution"):
        assert metrics[cls]["precision"] == 1.0
        assert metrics[cls]["recall"] == 1.0
        assert metrics[cls]["f1"] == 1.0
    assert metrics["direct_tool"]["support"] == 2.0
    assert metrics["decompose"]["support"] == 1.0


def test_per_class_routing_metrics_class_imbalance_exposes_minority_failure() -> None:
    # 8 majority correct, but every minority instance misrouted to majority
    pred = ["direct_tool"] * 10
    ref = ["direct_tool"] * 8 + ["decompose", "code_execution"]
    metrics = per_class_routing_metrics(pred, ref)

    # Naive accuracy looks fine (8/10 = 0.8) but minority recall is 0
    assert routing_accuracy(pred, ref) == 0.8
    assert metrics["decompose"]["recall"] == 0.0
    assert metrics["decompose"]["f1"] == 0.0
    assert metrics["code_execution"]["recall"] == 0.0
    # Majority class precision drops because it absorbed misroutes
    assert metrics["direct_tool"]["precision"] == 0.8
    assert metrics["direct_tool"]["recall"] == 1.0


def test_per_class_routing_metrics_empty_inputs() -> None:
    assert per_class_routing_metrics([], []) == {}


def test_per_class_routing_metrics_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        per_class_routing_metrics(["a"], ["a", "b"])


def test_routing_macro_f1_unweighted_average() -> None:
    # Two classes, one perfect (f1=1.0), one totally wrong (f1=0.0)
    pred = ["a", "a", "a", "a"]
    ref = ["a", "a", "b", "b"]
    # class a: precision 2/4=0.5, recall 2/2=1.0, f1 = 2*0.5*1.0/1.5 = 0.6667
    # class b: precision 0/0=0.0, recall 0/2=0.0, f1 = 0.0
    # macro = (0.6667 + 0.0) / 2 = 0.3333
    macro = routing_macro_f1(pred, ref)
    assert macro == pytest.approx(1 / 3, abs=1e-4)


def test_routing_macro_f1_empty_returns_zero() -> None:
    assert routing_macro_f1([], []) == 0.0


def test_routing_confusion_matrix_diagonal_when_perfect() -> None:
    pred = ["decompose", "direct_tool", "decompose"]
    ref = ["decompose", "direct_tool", "decompose"]
    cm = routing_confusion_matrix(pred, ref)
    # Diagonal entries match support; off-diagonal all zero
    assert cm["decompose"]["decompose"] == 2
    assert cm["direct_tool"]["direct_tool"] == 1
    assert cm["decompose"]["direct_tool"] == 0
    assert cm["direct_tool"]["decompose"] == 0


def test_routing_confusion_matrix_off_diagonal_captures_misroutes() -> None:
    pred = ["direct_tool", "direct_tool", "decompose"]
    ref = ["decompose", "direct_tool", "decompose"]
    cm = routing_confusion_matrix(pred, ref)
    # Row = ground truth, column = prediction
    assert cm["decompose"]["direct_tool"] == 1  # 1 misroute decompose -> direct_tool
    assert cm["decompose"]["decompose"] == 1
    assert cm["direct_tool"]["direct_tool"] == 1
    # Row sums equal support
    assert sum(cm["decompose"].values()) == 2
    assert sum(cm["direct_tool"].values()) == 1


def test_routing_confusion_matrix_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        routing_confusion_matrix(["a"], ["a", "b"])
