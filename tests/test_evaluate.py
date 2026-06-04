"""Tests for orchestratebench.evaluate."""

import pytest

from orchestratebench.core import ExecutionTrace, OrchestratorAction, RoutingDecision, SubAgentType
from orchestratebench.evaluate import (
    mean_cost,
    mean_latency,
    policy_comparison,
    routing_accuracy,
    success_rate,
)


def _trace(success: bool, latency: float = 100.0, cost: float = 0.05) -> ExecutionTrace:
    action = OrchestratorAction(
        task_id="t1",
        decision=RoutingDecision.DIRECT_TOOL,
        selected_agent=SubAgentType.TOOL_CALL,
        reasoning="fixed",
    )
    return ExecutionTrace(
        task_id="t1",
        actions=[action],
        total_latency_ms=latency,
        total_cost_usd=cost,
        success=success,
    )


def test_success_rate():
    traces = [_trace(True), _trace(False), _trace(True)]
    assert success_rate(traces) == pytest.approx(2 / 3)


def test_success_rate_empty():
    assert success_rate([]) == 0.0


def test_mean_latency():
    traces = [_trace(True, latency=100.0), _trace(True, latency=200.0)]
    assert mean_latency(traces) == pytest.approx(150.0)


def test_mean_cost():
    traces = [_trace(True, cost=0.10), _trace(True, cost=0.20)]
    assert mean_cost(traces) == pytest.approx(0.15)


def test_routing_accuracy_perfect():
    preds = ["decompose", "direct_tool", "reason_only"]
    refs = ["decompose", "direct_tool", "reason_only"]
    assert routing_accuracy(preds, refs) == pytest.approx(1.0)


def test_routing_accuracy_partial():
    preds = ["decompose", "direct_tool"]
    refs = ["decompose", "reason_only"]
    assert routing_accuracy(preds, refs) == pytest.approx(0.5)


def test_policy_comparison():
    traces_a = [_trace(True), _trace(True)]
    traces_b = [_trace(False), _trace(False)]
    result = policy_comparison({"policy_a": traces_a, "policy_b": traces_b})
    assert result["policy_a"]["success_rate"] == pytest.approx(1.0)
    assert result["policy_b"]["success_rate"] == pytest.approx(0.0)
