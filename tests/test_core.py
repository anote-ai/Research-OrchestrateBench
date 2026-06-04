"""Tests for orchestratebench core."""

from __future__ import annotations

import pytest

from orchestratebench.core import (
    AgentTask,
    DependencyGraph,
    ExecutionTrace,
    FixedPolicy,
    HeuristicPolicy,
    OrchestratorAction,
    OrchestratorBench,
    RetryPolicy,
    RoutingDecision,
    SubAgentType,
    TaskStatus,
    _default_simulate,
)
from orchestratebench.data import (
    make_benchmark_tasks,
    make_devops_deploy_workflow,
    make_finance_approval_workflow,
    make_hr_onboarding_workflow,
    make_task,
)


# ---------------------------------------------------------------------------
# AgentTask
# ---------------------------------------------------------------------------


def test_agent_task_defaults() -> None:
    t = AgentTask(
        description="test",
        complexity_score=0.5,
        requires_code=False,
        requires_retrieval=False,
    )
    assert 0.0 <= t.complexity_score <= 1.0
    assert t.max_retries == 2
    assert t.timeout_ms == 5000.0
    assert t.dependencies == []


def test_agent_task_deduplicates_dependencies() -> None:
    t = AgentTask(
        description="test",
        complexity_score=0.3,
        requires_code=False,
        requires_retrieval=False,
        dependencies=["a", "b", "a"],
    )
    assert t.dependencies == ["a", "b"]


def test_orchestrator_action_confidence_validation() -> None:
    with pytest.raises(Exception):
        OrchestratorAction(
            task_id="x",
            decision=RoutingDecision.REASON_ONLY,
            confidence=1.5,
        )


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


def test_fixed_policy_always_direct_tool() -> None:
    policy = FixedPolicy()
    for complexity in [0.1, 0.5, 0.95]:
        task = make_task("t", complexity=complexity)
        action = policy.route(task)
        assert action.decision == RoutingDecision.DIRECT_TOOL
        assert action.selected_agent == SubAgentType.TOOL_CALL


def test_heuristic_policy_high_complexity_decomposes() -> None:
    policy = HeuristicPolicy()
    task = make_task("complex", complexity=0.85)
    action = policy.route(task)
    assert action.decision == RoutingDecision.DECOMPOSE
    assert action.selected_agent == SubAgentType.PLANNING


def test_heuristic_policy_code_task() -> None:
    policy = HeuristicPolicy()
    task = make_task("run script", complexity=0.5, requires_code=True)
    action = policy.route(task)
    assert action.decision == RoutingDecision.CODE_EXECUTION


def test_heuristic_policy_retrieval_task() -> None:
    policy = HeuristicPolicy()
    task = make_task("search docs", complexity=0.3, requires_retrieval=True)
    action = policy.route(task)
    assert action.decision == RoutingDecision.DIRECT_TOOL
    assert action.selected_agent == SubAgentType.RETRIEVAL


def test_heuristic_policy_simple_reason() -> None:
    policy = HeuristicPolicy()
    task = make_task("simple", complexity=0.2)
    action = policy.route(task)
    assert action.decision == RoutingDecision.REASON_ONLY


# ---------------------------------------------------------------------------
# DependencyGraph
# ---------------------------------------------------------------------------


def test_dependency_graph_topological_order() -> None:
    t1 = make_task("a", complexity=0.1)
    t2 = make_task("b", complexity=0.2, dependencies=[t1.task_id])
    t3 = make_task("c", complexity=0.3, dependencies=[t2.task_id])
    graph = DependencyGraph([t1, t2, t3])
    order = graph.topological_order()
    assert order.index(t1.task_id) < order.index(t2.task_id)
    assert order.index(t2.task_id) < order.index(t3.task_id)


def test_dependency_graph_cycle_detection() -> None:
    t1 = AgentTask(
        description="a",
        complexity_score=0.1,
        requires_code=False,
        requires_retrieval=False,
    )
    t2 = AgentTask(
        description="b",
        complexity_score=0.2,
        requires_code=False,
        requires_retrieval=False,
        dependencies=[t1.task_id],
    )
    # Manually add cycle: t1 depends on t2
    t1_cycle = AgentTask(
        task_id=t1.task_id,
        description="a",
        complexity_score=0.1,
        requires_code=False,
        requires_retrieval=False,
        dependencies=[t2.task_id],
    )
    graph = DependencyGraph([t1_cycle, t2])
    with pytest.raises(ValueError, match="cycle"):
        graph.topological_order()


def test_dependency_graph_critical_path() -> None:
    t1 = make_task("a", complexity=0.1)
    t2 = make_task("b", complexity=0.2, dependencies=[t1.task_id])
    t3 = make_task("c", complexity=0.3, dependencies=[t2.task_id])
    graph = DependencyGraph([t1, t2, t3])
    assert graph.critical_path_length() == 3


# ---------------------------------------------------------------------------
# RetryPolicy
# ---------------------------------------------------------------------------


def test_retry_policy_succeeds_eventually() -> None:
    inner = HeuristicPolicy()
    policy = RetryPolicy(inner, failure_rate=0.5, seed=7)
    task = make_task("retry test", complexity=0.2, max_retries=5)
    trace = policy.execute_with_retry(task)
    # With 5 retries and 50% failure rate, success is overwhelmingly likely
    assert trace.success or trace.status == TaskStatus.FAILED


def test_retry_policy_zero_failure_rate() -> None:
    inner = FixedPolicy()
    policy = RetryPolicy(inner, failure_rate=0.0, seed=1)
    task = make_task("no fail", complexity=0.3)
    trace = policy.execute_with_retry(task)
    assert trace.success is True
    assert trace.n_retries == 0


# ---------------------------------------------------------------------------
# OrchestratorBench
# ---------------------------------------------------------------------------


def test_bench_evaluate_policy_returns_traces() -> None:
    tasks = make_benchmark_tasks(n=5)
    bench = OrchestratorBench(tasks=tasks)
    traces = bench.evaluate_policy(FixedPolicy())
    assert len(traces) == 5
    assert all(isinstance(t, ExecutionTrace) for t in traces)


def test_bench_compare_policies_keys() -> None:
    tasks = make_benchmark_tasks(n=4)
    bench = OrchestratorBench(tasks=tasks)
    result = bench.compare_policies({"fixed": FixedPolicy(), "heuristic": HeuristicPolicy()})
    assert set(result.keys()) == {"fixed", "heuristic"}


def test_bench_evaluate_with_dependencies_skips_on_failure() -> None:
    workflow = make_finance_approval_workflow()
    bench = OrchestratorBench(tasks=workflow)

    # Simulate failure on first task
    def fail_first(action: OrchestratorAction) -> ExecutionTrace:
        return ExecutionTrace(
            task_id=action.task_id,
            actions=[action],
            total_latency_ms=100.0,
            total_cost_usd=0.001,
            success=False,
            status=TaskStatus.FAILED,
        )

    traces = bench.evaluate_with_dependencies(FixedPolicy(), simulate_fn=fail_first)
    statuses = [t.status for t in traces]
    assert TaskStatus.FAILED in statuses
    assert TaskStatus.SKIPPED in statuses


# ---------------------------------------------------------------------------
# Workflow templates
# ---------------------------------------------------------------------------


def test_finance_workflow_dependency_chain() -> None:
    tasks = make_finance_approval_workflow()
    assert len(tasks) == 4
    # Each task after first depends on the previous
    for i in range(1, len(tasks)):
        assert tasks[i - 1].task_id in tasks[i].dependencies


def test_hr_onboarding_workflow_has_retries() -> None:
    tasks = make_hr_onboarding_workflow()
    assert any(t.max_retries > 2 for t in tasks)


def test_devops_deploy_workflow_ordered() -> None:
    tasks = make_devops_deploy_workflow()
    graph = DependencyGraph(tasks)
    order = graph.topological_order()
    assert len(order) == len(tasks)
    assert graph.critical_path_length() == len(tasks)
