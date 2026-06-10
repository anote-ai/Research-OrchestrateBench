"""Failure injection and cascade measurement for orchestratebench.

Implements the failure-mode taxonomy benchmark (issue #4) and cascade
propagation measurement (issue #7). Failure modes are drawn from MAST
(2025, 14 modes across 1,642 traces) and adapted into controllable
injection points for multi-agent orchestration pipelines.
"""

from __future__ import annotations

import random
from enum import Enum
from typing import Callable, Dict, List, Optional

from .core import (
    AgentTask,
    DependencyGraph,
    ExecutionTrace,
    OrchestratorAction,
    RetryRecord,
    RoutingDecision,
    SubAgentType,
    TaskStatus,
)


class FailureMode(str, Enum):
    AMBIGUOUS_DELEGATION = "ambiguous_delegation"
    TOOL_INVOCATION_ERROR = "tool_invocation_error"
    CONTEXT_POLLUTION = "context_pollution"
    CONFLICTING_OUTPUTS = "conflicting_outputs"
    PREMATURE_ACTION = "premature_action"


class FailureInjector:
    """Inject controlled failures into orchestration traces.

    Each failure mode corrupts the trace in a specific, reproducible way so
    downstream metrics (recovery rate, cascade radius) are measured under
    known conditions.
    """

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def inject(
        self,
        trace: ExecutionTrace,
        mode: FailureMode,
    ) -> ExecutionTrace:
        fn = _INJECTORS.get(mode)
        if fn is None:
            raise ValueError(f"Unknown failure mode: {mode}")
        return fn(trace, self._rng)


def _inject_ambiguous_delegation(
    trace: ExecutionTrace, rng: random.Random
) -> ExecutionTrace:
    if not trace.actions:
        return trace
    corrupted = trace.actions[0].model_copy()
    options = [d for d in RoutingDecision if d != corrupted.decision]
    if options:
        corrupted.decision = rng.choice(options)
    corrupted.confidence = rng.uniform(0.3, 0.55)
    corrupted.reasoning = "Ambiguous delegation: unclear which sub-agent should handle this task."
    return trace.model_copy(
        update={
            "actions": [corrupted] + trace.actions[1:],
            "success": False,
            "status": TaskStatus.FAILED,
        }
    )


def _inject_tool_invocation_error(
    trace: ExecutionTrace, rng: random.Random
) -> ExecutionTrace:
    retry = RetryRecord(
        attempt=0,
        error="ToolInvocationError: tool returned non-zero exit code",
        latency_ms=rng.uniform(50, 500),
    )
    return trace.model_copy(
        update={
            "success": False,
            "status": TaskStatus.FAILED,
            "retries": trace.retries + [retry],
            "total_latency_ms": trace.total_latency_ms + retry.latency_ms,
        }
    )


def _inject_context_pollution(
    trace: ExecutionTrace, rng: random.Random
) -> ExecutionTrace:
    if not trace.actions:
        return trace
    corrupted = trace.actions[0].model_copy()
    corrupted.reasoning = (
        "Context pollution: irrelevant context from a prior agent "
        "corrupted this agent's input, producing a plausible but wrong answer."
    )
    corrupted.confidence = rng.uniform(0.6, 0.9)
    return trace.model_copy(
        update={
            "actions": [corrupted] + trace.actions[1:],
            "success": False,
            "status": TaskStatus.FAILED,
            "total_cost_usd": trace.total_cost_usd * rng.uniform(1.2, 1.8),
        }
    )


def _inject_conflicting_outputs(
    trace: ExecutionTrace, rng: random.Random
) -> ExecutionTrace:
    extra_action = OrchestratorAction(
        task_id=trace.task_id,
        decision=trace.actions[0].decision if trace.actions else RoutingDecision.DIRECT_TOOL,
        selected_agent=SubAgentType.RETRIEVAL,
        reasoning="Conflicting outputs: two sub-agents returned contradictory results.",
        confidence=rng.uniform(0.4, 0.6),
    )
    return trace.model_copy(
        update={
            "actions": trace.actions + [extra_action],
            "success": False,
            "status": TaskStatus.FAILED,
            "n_subagent_calls": trace.n_subagent_calls + 1,
            "total_latency_ms": trace.total_latency_ms + rng.uniform(200, 1000),
            "total_cost_usd": trace.total_cost_usd + rng.uniform(0.01, 0.05),
        }
    )


def _inject_premature_action(
    trace: ExecutionTrace, rng: random.Random
) -> ExecutionTrace:
    if not trace.actions:
        return trace
    corrupted = trace.actions[0].model_copy()
    corrupted.decision = RoutingDecision.REASON_ONLY
    corrupted.selected_agent = None
    corrupted.reasoning = "Premature action: agent responded before gathering sufficient context."
    corrupted.confidence = rng.uniform(0.7, 0.95)
    return trace.model_copy(
        update={
            "actions": [corrupted] + trace.actions[1:],
            "success": False,
            "status": TaskStatus.FAILED,
        }
    )


_INJECTORS: Dict[FailureMode, Callable[[ExecutionTrace, random.Random], ExecutionTrace]] = {
    FailureMode.AMBIGUOUS_DELEGATION: _inject_ambiguous_delegation,
    FailureMode.TOOL_INVOCATION_ERROR: _inject_tool_invocation_error,
    FailureMode.CONTEXT_POLLUTION: _inject_context_pollution,
    FailureMode.CONFLICTING_OUTPUTS: _inject_conflicting_outputs,
    FailureMode.PREMATURE_ACTION: _inject_premature_action,
}


# ---------------------------------------------------------------------------
# Cascade measurement (issue #7)
# ---------------------------------------------------------------------------


def measure_cascade(
    tasks: List[AgentTask],
    policy: object,
    injection_stage: int,
    failure_mode: FailureMode,
    simulate_fn: Optional[Callable] = None,
    seed: int = 0,
) -> Dict:
    """Run a workflow with a failure injected at *injection_stage* and measure cascade.

    Returns a dict with:
    - ``cascade_radius``: number of downstream tasks that failed due to the
      injected error (not counting the injection point itself).
    - ``total_tasks``: total number of tasks in the workflow.
    - ``recovery_completeness``: fraction of post-injection tasks that
      still succeeded despite the upstream failure.
    - ``traces``: the full list of execution traces for inspection.
    """
    graph = DependencyGraph(tasks)
    order = graph.topological_order()
    task_map = {t.task_id: t for t in tasks}

    rng = random.Random(seed)
    injector = FailureInjector(seed=seed)
    completed: Dict[str, ExecutionTrace] = {}
    traces: List[ExecutionTrace] = []

    for idx, tid in enumerate(order):
        task = task_map[tid]

        dep_failed = any(
            not completed[dep].success
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
                dependencies_resolved=list(task.dependencies),
            )
            completed[tid] = skipped
            traces.append(skipped)
            continue

        action = policy.route(task)  # type: ignore[union-attr]
        if simulate_fn is not None:
            trace = simulate_fn(action)
        else:
            trace = ExecutionTrace(
                task_id=action.task_id,
                actions=[action],
                total_latency_ms=rng.uniform(100, 2000),
                total_cost_usd=rng.uniform(0.001, 0.05),
                success=True,
                status=TaskStatus.SUCCESS,
                n_subagent_calls=1,
                dependencies_resolved=list(task.dependencies),
            )

        if idx == injection_stage:
            trace = injector.inject(trace, failure_mode)

        completed[tid] = trace
        traces.append(trace)

    injected_idx = min(injection_stage, len(traces) - 1)
    post_injection = traces[injected_idx + 1:]
    cascade_failures = sum(1 for t in post_injection if not t.success)
    post_count = len(post_injection)
    recovery = (post_count - cascade_failures) / post_count if post_count > 0 else 1.0

    return {
        "cascade_radius": cascade_failures,
        "total_tasks": len(tasks),
        "injection_stage": injected_idx,
        "failure_mode": failure_mode.value,
        "recovery_completeness": recovery,
        "traces": traces,
    }


def recovery_rate_by_mode(
    tasks: List[AgentTask],
    policy: object,
    modes: Optional[List[FailureMode]] = None,
    n_runs: int = 10,
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    """Run failure injection across all modes and compute per-mode recovery rate.

    Returns ``{mode_name: {"recovery_rate": float, "mean_cascade_radius": float}}``.
    """
    if modes is None:
        modes = list(FailureMode)

    rng = random.Random(seed)
    results: Dict[str, Dict[str, float]] = {}

    for mode in modes:
        recoveries: List[float] = []
        radii: List[int] = []
        for run in range(n_runs):
            stage = rng.randint(0, max(0, len(tasks) - 2))
            result = measure_cascade(
                tasks=tasks,
                policy=policy,
                injection_stage=stage,
                failure_mode=mode,
                seed=seed + run,
            )
            recoveries.append(result["recovery_completeness"])
            radii.append(result["cascade_radius"])

        results[mode.value] = {
            "recovery_rate": sum(recoveries) / len(recoveries) if recoveries else 0.0,
            "mean_cascade_radius": sum(radii) / len(radii) if radii else 0.0,
        }

    return results
