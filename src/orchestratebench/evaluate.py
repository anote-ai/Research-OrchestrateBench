"""Evaluation metrics for orchestratebench."""

from __future__ import annotations

from typing import Dict, List, Optional

from .core import ExecutionTrace, RoutingDecision, TaskStatus
from .statistics import metric_ci


def success_rate(traces: List[ExecutionTrace]) -> float:
    if not traces:
        return 0.0
    return sum(1 for t in traces if t.success) / len(traces)


def mean_latency(traces: List[ExecutionTrace]) -> float:
    if not traces:
        return 0.0
    return sum(t.total_latency_ms for t in traces) / len(traces)


def mean_cost(traces: List[ExecutionTrace]) -> float:
    if not traces:
        return 0.0
    return sum(t.total_cost_usd for t in traces) / len(traces)


def routing_accuracy(predicted: List[str], reference: List[str]) -> float:
    if not reference:
        return 0.0
    matches = sum(p == r for p, r in zip(predicted, reference))
    return matches / len(reference)


def policy_comparison(
    policies: Dict[str, List[ExecutionTrace]],
    with_ci: bool = False,
    n_resamples: int = 2000,
    seed: int = 0,
) -> Dict:
    """Compare policies on summary metrics.

    When ``with_ci`` is True, each scalar metric gains a companion
    ``<metric>_ci`` key holding a ``(low, high)`` 95% bootstrap interval, so
    framework comparisons can report ``mean ± CI``. Defaults to
    ``with_ci=False`` so existing callers are unaffected.
    """
    ci_metrics = (
        ("success_rate", success_rate),
        ("mean_latency", mean_latency),
        ("mean_cost", mean_cost),
        ("orchestration_efficiency_score", orchestration_efficiency_score),
        ("task_dependency_score", task_dependency_score),
    )
    result = {}
    for name, traces in policies.items():
        metrics: Dict[str, object] = {
            "success_rate": success_rate(traces),
            "mean_latency": mean_latency(traces),
            "mean_cost": mean_cost(traces),
            "orchestration_efficiency_score": orchestration_efficiency_score(traces),
            "task_dependency_score": task_dependency_score(traces),
            "n_traces": len(traces),
        }
        if with_ci:
            for key, fn in ci_metrics:
                _, low, high = metric_ci(traces, fn, n_resamples=n_resamples, seed=seed)
                metrics[f"{key}_ci"] = (low, high)
        result[name] = metrics
    return result


def routing_distribution(traces: List[ExecutionTrace]) -> Dict[str, int]:
    counts: Dict[str, int] = {d.value: 0 for d in RoutingDecision}
    for trace in traces:
        for action in trace.actions:
            counts[action.decision.value] = counts.get(action.decision.value, 0) + 1
    return counts


def orchestration_efficiency_score(
    traces: List[ExecutionTrace],
    latency_budget_ms: float = 2000.0,
    cost_budget_usd: float = 0.05,
) -> float:
    """Composite score [0, 1] balancing success, latency, and cost.

    A trace is "efficient" when it succeeds within both the latency
    and cost budgets.  The final score is the fraction of efficient
    traces, penalised by the mean retry overhead.
    """
    if not traces:
        return 0.0
    efficient = 0
    for t in traces:
        if (
            t.success
            and t.total_latency_ms <= latency_budget_ms
            and t.total_cost_usd <= cost_budget_usd
        ):
            efficient += 1
    base = efficient / len(traces)
    # Penalty: mean fraction of retries relative to max_possible assumed 3
    max_assumed_retries = 3
    mean_retry_fraction = (
        sum(t.n_retries for t in traces) / (len(traces) * max_assumed_retries)
    )
    penalty = min(mean_retry_fraction, 1.0) * 0.2
    return max(0.0, base - penalty)


def task_dependency_score(traces: List[ExecutionTrace]) -> float:
    """Fraction of traces whose declared dependencies were fully resolved.

    A dependency is considered resolved when it appears in
    ``trace.dependencies_resolved``. Traces with no declared dependencies
    contribute a perfect score, while skipped or partially resolved traces are
    penalised proportionally.
    """
    if not traces:
        return 0.0
    scores: List[float] = []
    for trace in traces:
        declared = set(trace.dependencies_declared)
        if not declared:
            scores.append(1.0)
            continue
        resolved = len(declared.intersection(trace.dependencies_resolved))
        scores.append(resolved / len(declared))
    return sum(scores) / len(scores)


def mean_retry_rate(traces: List[ExecutionTrace]) -> float:
    """Mean number of retries per trace."""
    if not traces:
        return 0.0
    return sum(t.n_retries for t in traces) / len(traces)


def skip_rate(traces: List[ExecutionTrace]) -> float:
    """Fraction of traces that were skipped due to failed dependencies."""
    if not traces:
        return 0.0
    return sum(1 for t in traces if t.status == TaskStatus.SKIPPED) / len(traces)


def throughput_score(
    traces: List[ExecutionTrace],
    window_ms: float = 10_000.0,
) -> float:
    """Estimated tasks completed per *window_ms* assuming sequential execution."""
    if not traces:
        return 0.0
    total_latency = sum(t.total_latency_ms for t in traces if t.success)
    if total_latency == 0:
        return 0.0
    return sum(1 for t in traces if t.success) / total_latency * window_ms


def confidence_calibration(
    traces: List[ExecutionTrace],
) -> Optional[float]:
    """Mean confidence of the first action in each trace (proxy for calibration)."""
    values = [
        trace.actions[0].confidence
        for trace in traces
        if trace.actions
    ]
    if not values:
        return None
    return sum(values) / len(values)


# ---------------------------------------------------------------------------
# Per-class classification metrics for routing decisions
# ---------------------------------------------------------------------------
#
# `routing_accuracy` is a single number and silently hides poor performance on
# minority classes — a problem in orchestration benchmarks where one routing
# class (typically DIRECT_TOOL) dominates. The helpers below report per-class
# precision / recall / F1 and a confusion matrix so error analysis and paper
# figures can isolate behaviour on each class.


def _validate_label_pair(predicted: List[str], reference: List[str]) -> None:
    if len(predicted) != len(reference):
        raise ValueError(
            f"predicted and reference must be same length; "
            f"got {len(predicted)} vs {len(reference)}"
        )


def per_class_routing_metrics(
    predicted: List[str], reference: List[str]
) -> Dict[str, Dict[str, float]]:
    """Per-class precision / recall / F1 / support for routing decisions.

    Returns ``{class_name: {"precision", "recall", "f1", "support"}}``. Useful
    when routing classes are imbalanced — ``routing_accuracy`` alone hides
    poor performance on minority classes.

    Convention: classes are taken from the union of values seen in *predicted*
    and *reference*. ``support`` is the number of times the class appears in
    *reference* (true count). When a class has zero predicted-or-reference
    instances we report ``precision = recall = f1 = 0.0``.

    Raises ``ValueError`` if input lengths differ.
    """
    _validate_label_pair(predicted, reference)
    if not reference:
        return {}

    classes = sorted(set(predicted) | set(reference))
    result: Dict[str, Dict[str, float]] = {}
    for cls in classes:
        tp = sum(1 for p, r in zip(predicted, reference) if p == cls and r == cls)
        fp = sum(1 for p, r in zip(predicted, reference) if p == cls and r != cls)
        fn = sum(1 for p, r in zip(predicted, reference) if p != cls and r == cls)
        support = sum(1 for r in reference if r == cls)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        result[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(support),
        }
    return result


def routing_macro_f1(predicted: List[str], reference: List[str]) -> float:
    """Macro-averaged F1 across all routing classes (unweighted).

    Each class contributes equally regardless of support, which makes macro F1
    the right summary number when minority-class performance matters. For a
    support-weighted variant, multiply each class F1 by ``support`` and divide
    by ``len(reference)``.

    Returns ``0.0`` if *reference* is empty.
    """
    metrics = per_class_routing_metrics(predicted, reference)
    if not metrics:
        return 0.0
    return sum(m["f1"] for m in metrics.values()) / len(metrics)


def routing_confusion_matrix(
    predicted: List[str], reference: List[str]
) -> Dict[str, Dict[str, int]]:
    """Confusion matrix as ``{reference_class: {predicted_class: count}}``.

    Drop-in for paper Figures and per-class error analysis. Rows = ground
    truth, columns = model predictions; a perfect classifier produces a
    diagonal matrix.

    Raises ``ValueError`` if input lengths differ.
    """
    _validate_label_pair(predicted, reference)
    if not reference:
        return {}

    classes = sorted(set(predicted) | set(reference))
    matrix: Dict[str, Dict[str, int]] = {
        ref_cls: {pred_cls: 0 for pred_cls in classes} for ref_cls in classes
    }
    for p, r in zip(predicted, reference):
        matrix[r][p] += 1
    return matrix
