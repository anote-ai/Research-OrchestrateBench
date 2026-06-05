"""Core models and policies for orchestratebench."""

from __future__ import annotations

import hashlib
import random
import uuid
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

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


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class AgentTask(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str
    complexity_score: float = Field(ge=0.0, le=1.0)
    requires_code: bool
    requires_retrieval: bool
    estimated_tokens: int = 500
    timeout_ms: float = Field(default=5000.0, ge=0.0)
    max_retries: int = Field(default=2, ge=0)
    dependencies: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("dependencies", mode="before")
    @classmethod
    def deduplicate_deps(cls, v: List[str]) -> List[str]:
        seen: Set[str] = set()
        result = []
        for item in v:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result


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


class RetryRecord(BaseModel):
    attempt: int
    error: str
    latency_ms: float


class ExecutionTrace(BaseModel):
    task_id: str
    actions: List[OrchestratorAction]
    total_latency_ms: float
    total_cost_usd: float
    success: bool
    status: TaskStatus = TaskStatus.SUCCESS
    n_subagent_calls: int = 0
    retries: List[RetryRecord] = Field(default_factory=list)
    dependencies_declared: List[str] = Field(default_factory=list)
    dependencies_resolved: List[str] = Field(default_factory=list)

    @property
    def n_retries(self) -> int:
        return len(self.retries)


class DependencyGraph:
    """Topological ordering and cycle detection for task dependencies."""

    def __init__(self, tasks: List[AgentTask]) -> None:
        self._tasks: Dict[str, AgentTask] = {t.task_id: t for t in tasks}
        self._validate_dependencies()

    def _validate_dependencies(self) -> None:
        missing: Dict[str, List[str]] = {}
        for task in self._tasks.values():
            unknown = [dep for dep in task.dependencies if dep not in self._tasks]
            if unknown:
                missing[task.task_id] = unknown
        if missing:
            parts = [
                f"{task_id} -> {', '.join(deps)}"
                for task_id, deps in sorted(missing.items())
            ]
            raise ValueError(
                "Dependency graph contains unknown task IDs: " + "; ".join(parts)
            )

    def topological_order(self) -> List[str]:
        """Return task IDs in a valid execution order (Kahn's algorithm)."""
        in_degree: Dict[str, int] = {tid: 0 for tid in self._tasks}
        children: Dict[str, List[str]] = {tid: [] for tid in self._tasks}
        for task in self._tasks.values():
            for dep in task.dependencies:
                if dep in self._tasks:
                    in_degree[task.task_id] += 1
                    children[dep].append(task.task_id)
        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        order: List[str] = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for child in children[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        if len(order) != len(self._tasks):
            raise ValueError("Dependency graph contains a cycle.")
        return order

    def critical_path_length(self) -> int:
        """Return the length of the longest dependency chain."""
        order = self.topological_order()
        depth: Dict[str, int] = {}
        for tid in order:
            task = self._tasks[tid]
            dep_depths = [
                depth[dep] for dep in task.dependencies if dep in depth
            ]
            depth[tid] = (max(dep_depths) + 1) if dep_depths else 1
        return max(depth.values()) if depth else 0


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


class RetryPolicy:
    """Wraps another policy and simulates retry behaviour on simulated failures."""

    def __init__(self, inner: Any, failure_rate: float = 0.2, seed: int = 0) -> None:
        self._inner = inner
        self._failure_rate = failure_rate
        self._rng = random.Random(seed)

    def route(self, task: AgentTask) -> OrchestratorAction:
        return self._inner.route(task)

    def execute_with_retry(
        self,
        task: AgentTask,
        simulate_fn: Optional[Callable[[OrchestratorAction], ExecutionTrace]] = None,
    ) -> ExecutionTrace:
        """Execute task routing with retry logic, returning a final trace."""
        action = self.route(task)
        retries: List[RetryRecord] = []
        for attempt in range(task.max_retries + 1):
            failed = self._rng.random() < self._failure_rate and attempt < task.max_retries
            if not failed:
                if simulate_fn is not None:
                    trace = simulate_fn(action)
                else:
                    trace = _default_simulate(action, seed=attempt)
                trace.retries.extend(retries)
                trace.status = TaskStatus.SUCCESS
                return trace
            # Record failure and retry
            retries.append(
                RetryRecord(
                    attempt=attempt,
                    error="Simulated transient failure",
                    latency_ms=self._rng.uniform(50, 300),
                )
            )
        # All retries exhausted
        trace = ExecutionTrace(
            task_id=task.task_id,
            actions=[action],
            total_latency_ms=sum(r.latency_ms for r in retries),
            total_cost_usd=0.0,
            success=False,
            status=TaskStatus.FAILED,
            n_subagent_calls=0,
            retries=retries,
        )
        return trace


def _default_simulate(action: OrchestratorAction, seed: int = 42) -> ExecutionTrace:
    stable_task_seed = int(hashlib.sha256(action.task_id.encode("utf-8")).hexdigest()[:12], 16)
    rng = random.Random(seed + stable_task_seed)
    profile = {
        RoutingDecision.REASON_ONLY: {
            "latency": (100.0, 500.0),
            "cost": (0.001, 0.008),
            "subagent_calls": (0, 0),
        },
        RoutingDecision.DIRECT_TOOL: {
            "latency": (350.0, 1000.0),
            "cost": (0.004, 0.018),
            "subagent_calls": (1, 1),
        },
        RoutingDecision.CODE_EXECUTION: {
            "latency": (900.0, 1800.0),
            "cost": (0.010, 0.030),
            "subagent_calls": (1, 2),
        },
        RoutingDecision.DECOMPOSE: {
            "latency": (1500.0, 2600.0),
            "cost": (0.020, 0.055),
            "subagent_calls": (2, 4),
        },
    }[action.decision]
    return ExecutionTrace(
        task_id=action.task_id,
        actions=[action],
        total_latency_ms=rng.uniform(*profile["latency"]),
        total_cost_usd=rng.uniform(*profile["cost"]),
        success=True,
        status=TaskStatus.SUCCESS,
        n_subagent_calls=rng.randint(*profile["subagent_calls"]),
    )


class OrchestratorBench:
    """Evaluates routing policies over a task list."""

    def __init__(self, tasks: Optional[List[AgentTask]] = None) -> None:
        self.tasks: List[AgentTask] = tasks or []

    def _default_simulate(
        self, action: OrchestratorAction, seed: int = 42
    ) -> ExecutionTrace:
        return _default_simulate(action, seed=seed)

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
            trace.dependencies_declared = list(task.dependencies)
            traces.append(trace)
        return traces

    def compare_policies(
        self, policies: Dict[str, Any]
    ) -> Dict[str, List[ExecutionTrace]]:
        return {name: self.evaluate_policy(p) for name, p in policies.items()}

    def evaluate_with_dependencies(
        self,
        policy: Any,
        simulate_fn: Optional[Callable] = None,
    ) -> List[ExecutionTrace]:
        """Evaluate policy respecting task dependency order."""
        graph = DependencyGraph(self.tasks)
        order = graph.topological_order()
        task_map = {t.task_id: t for t in self.tasks}
        completed: Dict[str, ExecutionTrace] = {}
        traces: List[ExecutionTrace] = []
        for tid in order:
            task = task_map[tid]
            # Skip if any dependency failed
            dep_failed = any(
                completed[dep].success is False
                for dep in task.dependencies
                if dep in completed
            )
            if dep_failed:
                skipped = ExecutionTrace(
                    task_id=tid,
                    actions=[],
                    total_latency_ms=0.0,
                    total_cost_usd=0.0,
                    success=False,
                    status=TaskStatus.SKIPPED,
                    dependencies_declared=list(task.dependencies),
                    dependencies_resolved=[
                        dep
                        for dep in task.dependencies
                        if dep in completed and completed[dep].success
                    ],
                )
                completed[tid] = skipped
                traces.append(skipped)
                continue
            action = policy.route(task)
            if simulate_fn is not None:
                trace = simulate_fn(action)
            else:
                trace = self._default_simulate(action, seed=len(traces))
            trace.dependencies_declared = list(task.dependencies)
            trace.dependencies_resolved = list(task.dependencies)
            completed[tid] = trace
            traces.append(trace)
        return traces
