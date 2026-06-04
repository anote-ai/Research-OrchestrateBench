"""Tests for orchestratebench.evaluate."""

import pytest
from orchestratebench.core import AgentTask, ExecutionTrace, OrchestratorAction, RoutingDecision
from orchestratebench.evaluate import (
    mean_cost,
    mean_latency,
    policy_comparison,
    routing_accuracy,
    routing_distribution,
    success_rate,
)


def _trace(success=True, latency=500.0, cost=0.01, decision=RoutingDecision.DIRECT_TOOL):
    action = OrchestratorAction(
        task_id="t1", decision=decision, confidence=0.9
    )
    return ExecutionTrace(
        task_id="t1",
        actions=[action],
        total_latency_ms=latency,
        total_cost_usd=cost,
        success=success,
    )


def test_success_rate():
    traces = [_trace(True), _trace(True), _trace(False)]
    assert success_rate(traces) == pytest.approx(2 / 3)


def test_mean_latency():
    traces = [_trace(latency=100), _trace(latency=300)]
    assert mean_latency(traces) == pytest.approx(200.0)


def test_mean_cost():
    traces = [_trace(cost=0.01), _trace(cost=0.03)]
    assert mean_cost(traces) == pytest.approx(0.02)


def test_routing_accuracy_perfect():
    assert routing_accuracy(["a", "b", "c"], ["a", "b", "c"]) == pytest.approx(1.0)


def test_routing_accuracy_zero():
    assert routing_accuracy(["a", "b"], ["c", "d"]) == pytest.approx(0.0)


def test_policy_comparison_structure():
    traces = [_trace()]
    result = policy_comparison({"p1": traces, "p2": traces})
    assert "p1" in result
    assert "success_rate" in result["p1"]
    assert "n_traces" in result["p1"]


def test_routing_distribution_counts():
    traces = [
        _trace(decision=RoutingDecision.DIRECT_TOOL),
        _trace(decision=RoutingDecision.DIRECT_TOOL),
        _trace(decision=RoutingDecision.DECOMPOSE),
    ]
    dist = routing_distribution(traces)
    assert dist["direct_tool"] == 2
    assert dist["decompose"] == 1
