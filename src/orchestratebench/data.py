"""Sample data and factory helpers for orchestratebench."""

from __future__ import annotations

import random
from typing import List, Optional

from .core import AgentTask, ExecutionTrace, OrchestratorAction, RoutingDecision, SubAgentType

SAMPLE_TASKS_RAW: List[dict] = [
    {"description": "Summarise a 10-page PDF report.", "complexity_score": 0.3, "requires_code": False, "requires_retrieval": False},
    {"description": "Write and execute a Python script to analyse CSV data.", "complexity_score": 0.6, "requires_code": True, "requires_retrieval": False},
    {"description": "Search the web and retrieve the latest ML papers.", "complexity_score": 0.4, "requires_code": False, "requires_retrieval": True},
    {"description": "Plan a multi-step research project with sub-tasks.", "complexity_score": 0.9, "requires_code": False, "requires_retrieval": False},
    {"description": "Answer a simple factual question.", "complexity_score": 0.1, "requires_code": False, "requires_retrieval": False},
    {"description": "Build and test a Flask REST API.", "complexity_score": 0.8, "requires_code": True, "requires_retrieval": False},
    {"description": "Fetch and parse JSON from an external API.", "complexity_score": 0.5, "requires_code": True, "requires_retrieval": True},
    {"description": "Write a literature review on transformer architectures.", "complexity_score": 0.75, "requires_code": False, "requires_retrieval": True},
]


def make_task(
    description: str,
    complexity: float = 0.5,
    requires_code: bool = False,
    requires_retrieval: bool = False,
) -> AgentTask:
    return AgentTask(
        description=description,
        complexity_score=complexity,
        requires_code=requires_code,
        requires_retrieval=requires_retrieval,
    )


def make_benchmark_tasks(n: int = 20, seed: int = 42) -> List[AgentTask]:
    rng = random.Random(seed)
    tasks = []
    for i in range(n):
        raw = rng.choice(SAMPLE_TASKS_RAW).copy()
        raw["complexity_score"] = rng.uniform(0.1, 1.0)
        tasks.append(AgentTask(**raw))
    return tasks


def make_execution_trace(
    task: AgentTask,
    policy_name: str = "fixed",
    success: bool = True,
    seed: int = 42,
) -> ExecutionTrace:
    rng = random.Random(seed)
    action = OrchestratorAction(
        task_id=task.task_id,
        decision=RoutingDecision.DIRECT_TOOL,
        selected_agent=SubAgentType.TOOL_CALL,
        reasoning=f"{policy_name} policy",
        confidence=0.9,
    )
    return ExecutionTrace(
        task_id=task.task_id,
        actions=[action],
        total_latency_ms=rng.uniform(100, 3000),
        total_cost_usd=rng.uniform(0.001, 0.1),
        success=success,
        n_subagent_calls=1,
    )
