"""Evaluation metrics for orchestratebench."""

from __future__ import annotations

from typing import Dict, List, Optional

from .core import ExecutionTrace, RoutingDecision, TaskStatus


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


def policy_comparison(policies: Dict[str, List[ExecutionTrace]]) -> Dict:
    result = {}
    for name, traces in policies.items():
        result[name] = {
            "success_rate": success_rate(traces),
            "mean_latency": mean_latency(traces),
            "mean_cost": mean_cost(traces),
            "orchestration_efficiency_score": orchestration_efficiency_score(traces),
            "task_dependency_score": task_dependency_score(traces),
            "n_traces": len(traces),
        }
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
