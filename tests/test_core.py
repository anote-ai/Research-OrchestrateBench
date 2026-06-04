"""Tests for orchestratebench.core."""

import pytest

from orchestratebench.core import (
    AgentTask,
    ExecutionTrace,
    FixedPolicy,
    OrchestratorAction,
    OrchestratorBench,
    RoutingDecision,
    SubAgentType,
)


def test_routing_decision_values():
    assert RoutingDecision.DECOMPOSE == "decompose"
    assert RoutingDecision.DIRECT_TOOL == "direct_tool"
    assert RoutingDecision.CODE_EXECUTION == "code_execution"
    assert RoutingDecision.REASON_ONLY == "reason_only"


def test_sub_agent_type_values():
    assert SubAgentType.RETRIEVAL == "retrieval"
    assert SubAgentType.CODE == "code"
    assert SubAgentType.TOOL_CALL == "tool_call"


def test_agent_task_construction():
    task = AgentTask(
        task_id="task-1",
        description="Summarise a document.",
        complexity_score=0.4,
        requires_code=False,
        requires_retrieval=True,
    )
    assert task.task_id == "task-1"
    assert task.requires_retrieval is True


def test_orchestrator_action_construction():
    action = OrchestratorAction(
        task_id="task-1",
        decision=RoutingDecision.DIRECT_TOOL,
        selected_agent=SubAgentType.TOOL_CALL,
        reasoning="Simple lookup.",
    )
    assert action.decision == RoutingDecision.DIRECT_TOOL
    assert action.selected_agent == SubAgentType.TOOL_CALL


def test_execution_trace_construction():
    action = OrchestratorAction(
        task_id="task-1",
        decision=RoutingDecision.REASON_ONLY,
        selected_agent=None,
        reasoning="No tool needed.",
    )
    trace = ExecutionTrace(
        task_id="task-1",
        actions=[action],
        total_latency_ms=120.5,
        total_cost_usd=0.02,
        success=True,
    )
    assert trace.success is True
    assert len(trace.actions) == 1


def test_fixed_policy_returns_orchestrator_action():
    policy = FixedPolicy()
    task = AgentTask(
        task_id="task-2",
        description="Run a SQL query.",
        complexity_score=0.6,
        requires_code=True,
        requires_retrieval=False,
    )
    action = policy.route(task)
    assert isinstance(action, OrchestratorAction)
    assert action.decision == RoutingDecision.DIRECT_TOOL


def test_fixed_policy_task_id_preserved():
    policy = FixedPolicy()
    task = AgentTask(
        task_id="xyz-99",
        description="Fetch data.",
        complexity_score=0.2,
        requires_code=False,
        requires_retrieval=True,
    )
    action = policy.route(task)
    assert action.task_id == "xyz-99"


def test_orchestrator_bench_returns_traces():
    bench = OrchestratorBench()
    tasks = [
        AgentTask(
            task_id=f"t{i}",
            description="task",
            complexity_score=0.5,
            requires_code=False,
            requires_retrieval=False,
        )
        for i in range(3)
    ]
    traces = bench.evaluate_policy(FixedPolicy(), tasks)
    assert len(traces) == 3
    assert all(isinstance(tr, ExecutionTrace) for tr in traces)
