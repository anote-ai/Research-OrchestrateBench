"""Tests for orchestratebench.core."""

import pytest
from orchestratebench.core import (
    AgentTask,
    ExecutionTrace,
    FixedPolicy,
    HeuristicPolicy,
    OrchestratorAction,
    OrchestratorBench,
    RoutingDecision,
    SubAgentType,
)


def _task(description="test", complexity=0.5, code=False, retrieval=False):
    return AgentTask(
        description=description,
        complexity_score=complexity,
        requires_code=code,
        requires_retrieval=retrieval,
    )


def test_routing_decision_enum():
    assert RoutingDecision.DECOMPOSE == "decompose"
    assert RoutingDecision.CODE_EXECUTION == "code_execution"


def test_sub_agent_type_enum():
    assert SubAgentType.CODE == "code"
    assert SubAgentType.RETRIEVAL == "retrieval"


def test_agent_task_construction():
    t = _task()
    assert t.complexity_score == 0.5
    assert isinstance(t.task_id, str)
    assert len(t.task_id) == 8


def test_orchestrator_action_confidence_validator():
    with pytest.raises(Exception):
        OrchestratorAction(
            task_id="t1",
            decision=RoutingDecision.REASON_ONLY,
            confidence=1.5,  # invalid
        )


def test_execution_trace_fields():
    action = OrchestratorAction(
        task_id="t1", decision=RoutingDecision.DIRECT_TOOL, confidence=0.9
    )
    trace = ExecutionTrace(
        task_id="t1",
        actions=[action],
        total_latency_ms=500.0,
        total_cost_usd=0.01,
        success=True,
        n_subagent_calls=1,
    )
    assert trace.success is True
    assert trace.n_subagent_calls == 1


def test_fixed_policy_always_direct_tool():
    policy = FixedPolicy()
    for complexity in [0.1, 0.5, 0.95]:
        action = policy.route(_task(complexity=complexity))
        assert action.decision == RoutingDecision.DIRECT_TOOL


def test_heuristic_policy_complex_to_decompose():
    policy = HeuristicPolicy()
    action = policy.route(_task(complexity=0.9))
    assert action.decision == RoutingDecision.DECOMPOSE


def test_heuristic_policy_code_to_code_execution():
    policy = HeuristicPolicy()
    action = policy.route(_task(complexity=0.5, code=True))
    assert action.decision == RoutingDecision.CODE_EXECUTION


def test_heuristic_policy_retrieval_to_direct_tool():
    policy = HeuristicPolicy()
    action = policy.route(_task(complexity=0.3, retrieval=True))
    assert action.decision == RoutingDecision.DIRECT_TOOL
    assert action.selected_agent == SubAgentType.RETRIEVAL


def test_orchestrator_bench_evaluate_policy_length():
    tasks = [_task(f"task {i}") for i in range(5)]
    bench = OrchestratorBench(tasks=tasks)
    traces = bench.evaluate_policy(FixedPolicy())
    assert len(traces) == 5


def test_compare_policies_keys():
    tasks = [_task()]
    bench = OrchestratorBench(tasks=tasks)
    result = bench.compare_policies({"fixed": FixedPolicy(), "heuristic": HeuristicPolicy()})
    assert set(result.keys()) == {"fixed", "heuristic"}
