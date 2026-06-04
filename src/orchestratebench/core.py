"""Core models and policies for orchestratebench."""

from __future__ import annotations

import random
import uuid
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class RoutingDecision(str, Enum):
    DECOMPOSE = "decompose"
    DIRECT_TOOL = "direct_tool"
    CODE_EXECUTION = "code_execution"
    REASON_ONLY = "reason_only"


class SubAgentType(str, Enum):
    RETRIEVAL = "retrieval"
    CODE = "code"
    PLANNING = "planning"
    SUMMARIZATION = "summarization"
    TOOL_CALL = "tool_call"


class AgentTask(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str
    complexity_score: float = Field(ge=0.0, le=1.0)
    requires_code: bool
    requires_retrieval: bool
    estimated_tokens: int = 500


class OrchestratorAction(BaseModel):
    task_id: str
    decision: RoutingDecision
    selected_agent: Optional[SubAgentType] = None
    reasoning: str = ""
    confidence: float = Field(default=0.9)

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {v}")
        return v


class ExecutionTrace(BaseModel):
    task_id: str
    actions: List[OrchestratorAction]
    total_latency_ms: float
    total_cost_usd: float
    success: bool
    n_subagent_calls: int = 0


class FixedPolicy:
    """Always routes to DIRECT_TOOL / TOOL_CALL regardless of task."""

    def route(self, task: AgentTask) -> OrchestratorAction:
        return OrchestratorAction(
            task_id=task.task_id,
            decision=RoutingDecision.DIRECT_TOOL,
            selected_agent=SubAgentType.TOOL_CALL,
            reasoning="Fixed policy: always use direct tool.",
            confidence=1.0,
        )


class HeuristicPolicy:
    """Routes based on task properties."""

    def route(self, task: AgentTask) -> OrchestratorAction:
        if task.complexity_score > 0.7:
            return OrchestratorAction(
                task_id=task.task_id,
                decision=RoutingDecision.DECOMPOSE,
                selected_agent=SubAgentType.PLANNING,
                reasoning="High complexity: decompose into sub-tasks.",
                confidence=0.85,
            )
        if task.requires_code:
            return OrchestratorAction(
                task_id=task.task_id,
                decision=RoutingDecision.CODE_EXECUTION,
                selected_agent=SubAgentType.CODE,
                reasoning="Task requires code execution.",
                confidence=0.9,
            )
        if task.requires_retrieval:
            return OrchestratorAction(
                task_id=task.task_id,
                decision=RoutingDecision.DIRECT_TOOL,
                selected_agent=SubAgentType.RETRIEVAL,
                reasoning="Task requires retrieval.",
                confidence=0.88,
            )
        return OrchestratorAction(
            task_id=task.task_id,
            decision=RoutingDecision.REASON_ONLY,
            selected_agent=None,
            reasoning="Simple reasoning task.",
            confidence=0.95,
        )


class OrchestratorBench:
    """Evaluates routing policies over a task list."""

    def __init__(self, tasks: Optional[List[AgentTask]] = None) -> None:
        self.tasks: List[AgentTask] = tasks or []

    def _default_simulate(self, action: OrchestratorAction, seed: int = 42) -> ExecutionTrace:
        rng = random.Random(seed + hash(action.task_id) % 10000)
        return ExecutionTrace(
            task_id=action.task_id,
            actions=[action],
            total_latency_ms=rng.uniform(100, 2000),
            total_cost_usd=rng.uniform(0.001, 0.05),
            success=True,
            n_subagent_calls=1,
        )

    def evaluate_policy(
        self,
        policy: Any,
        simulate_fn: Optional[Callable] = None,
    ) -> List[ExecutionTrace]:
        traces = []
        for i, task in enumerate(self.tasks):
            action = policy.route(task)
            if simulate_fn is not None:
                trace = simulate_fn(action)
            else:
                trace = self._default_simulate(action, seed=i)
            traces.append(trace)
        return traces

    def compare_policies(
        self, policies: Dict[str, Any]
    ) -> Dict[str, List[ExecutionTrace]]:
        return {name: self.evaluate_policy(p) for name, p in policies.items()}
