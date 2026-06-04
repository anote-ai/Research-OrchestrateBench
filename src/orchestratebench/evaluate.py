"""Evaluation metrics for orchestratebench."""

from __future__ import annotations

from typing import Dict, List

from .core import ExecutionTrace, RoutingDecision


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
            "n_traces": len(traces),
        }
    return result


def routing_distribution(traces: List[ExecutionTrace]) -> Dict[str, int]:
    counts: Dict[str, int] = {d.value: 0 for d in RoutingDecision}
    for trace in traces:
        for action in trace.actions:
            counts[action.decision.value] = counts.get(action.decision.value, 0) + 1
    return counts
