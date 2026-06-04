"""Evaluation metrics for Orchestrate-Bench."""

from __future__ import annotations

from orchestratebench.core import ExecutionTrace


def success_rate(traces: list[ExecutionTrace]) -> float:
    """Fraction of traces that succeeded."""
    if not traces:
        return 0.0
    return sum(1 for t in traces if t.success) / len(traces)


def mean_latency(traces: list[ExecutionTrace]) -> float:
    """Mean total_latency_ms across traces."""
    if not traces:
        return 0.0
    return sum(t.total_latency_ms for t in traces) / len(traces)


def mean_cost(traces: list[ExecutionTrace]) -> float:
    """Mean total_cost_usd across traces."""
    if not traces:
        return 0.0
    return sum(t.total_cost_usd for t in traces) / len(traces)


def routing_accuracy(
    predicted_decisions: list[str],
    reference_decisions: list[str],
) -> float:
    """Fraction of routing decisions that match the reference."""
    if not reference_decisions:
        return 0.0
    if len(predicted_decisions) != len(reference_decisions):
        raise ValueError("predicted and reference lists must have the same length")
    matches = sum(p == r for p, r in zip(predicted_decisions, reference_decisions))
    return matches / len(reference_decisions)


def policy_comparison(
    policies: dict[str, list[ExecutionTrace]],
) -> dict[str, dict[str, float]]:
    """Return per-policy aggregated success/latency/cost."""
    comparison: dict[str, dict[str, float]] = {}
    for name, traces in policies.items():
        comparison[name] = {
            "success_rate": success_rate(traces),
            "mean_latency_ms": mean_latency(traces),
            "mean_cost_usd": mean_cost(traces),
        }
    return comparison
