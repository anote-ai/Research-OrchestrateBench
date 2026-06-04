"""Core data models and policy primitives for Orchestrate-Bench."""

from __future__ import annotations

from enum import Enum
from typing import Optional, Protocol

from pydantic import BaseModel, Field


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
    task_id: str
    description: str
    complexity_score: float = Field(ge=0.0, le=1.0)
    requires_code: bool
    requires_retrieval: bool


class OrchestratorAction(BaseModel):
    task_id: str
    decision: RoutingDecision
    selected_agent: Optional[SubAgentType] = None
    reasoning: str


class ExecutionTrace(BaseModel):
    task_id: str
    actions: list[OrchestratorAction]
    total_latency_ms: float = Field(ge=0.0)
    total_cost_usd: float = Field(ge=0.0)
    success: bool


class PolicyProtocol(Protocol):
    def route(self, task: AgentTask) -> OrchestratorAction:
        ...


class FixedPolicy:
    """Baseline policy that always routes directly to a tool."""

    def route(self, task: AgentTask) -> OrchestratorAction:
        return OrchestratorAction(
            task_id=task.task_id,
            decision=RoutingDecision.DIRECT_TOOL,
            selected_agent=SubAgentType.TOOL_CALL,
            reasoning="Fixed policy: always use direct tool call.",
        )


class OrchestratorBench:
    """Benchmark harness for evaluating orchestration policies."""

    def evaluate_policy(
        self,
        policy: PolicyProtocol,
        tasks: list[AgentTask],
    ) -> list[ExecutionTrace]:
        """Run policy over tasks and return execution traces (stub).

        In a full implementation this would execute actions in a sandboxed
        environment and record real latency and cost.
        """
        traces: list[ExecutionTrace] = []
        for task in tasks:
            action = policy.route(task)
            traces.append(
                ExecutionTrace(
                    task_id=task.task_id,
                    actions=[action],
                    total_latency_ms=0.0,
                    total_cost_usd=0.0,
                    success=False,  # stub — replace with real execution
                )
            )
        return traces
