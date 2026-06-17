"""Sample data and factory helpers for orchestratebench."""

from __future__ import annotations

import random
from typing import List, Optional

from .core import (
    AgentTask,
    ExecutionTrace,
    OrchestratorAction,
    RoutingDecision,
    SubAgentType,
    TaskStatus,
)

SAMPLE_TASKS_RAW: List[dict] = [
    {
        "description": "Summarise a 10-page PDF report.",
        "complexity_score": 0.3,
        "requires_code": False,
        "requires_retrieval": False,
    },
    {
        "description": "Write and execute a Python script to analyse CSV data.",
        "complexity_score": 0.6,
        "requires_code": True,
        "requires_retrieval": False,
    },
    {
        "description": "Search the web and retrieve the latest ML papers.",
        "complexity_score": 0.4,
        "requires_code": False,
        "requires_retrieval": True,
    },
    {
        "description": "Plan a multi-step research project with sub-tasks.",
        "complexity_score": 0.9,
        "requires_code": False,
        "requires_retrieval": False,
    },
    {
        "description": "Answer a simple factual question.",
        "complexity_score": 0.1,
        "requires_code": False,
        "requires_retrieval": False,
    },
    {
        "description": "Build and test a Flask REST API.",
        "complexity_score": 0.8,
        "requires_code": True,
        "requires_retrieval": False,
    },
    {
        "description": "Fetch and parse JSON from an external API.",
        "complexity_score": 0.5,
        "requires_code": True,
        "requires_retrieval": True,
    },
    {
        "description": "Write a literature review on transformer architectures.",
        "complexity_score": 0.75,
        "requires_code": False,
        "requires_retrieval": True,
    },
]

# ---------------------------------------------------------------------------
# Realistic enterprise workflow templates
# ---------------------------------------------------------------------------


def make_finance_approval_workflow() -> List[AgentTask]:
    """Multi-stage purchase-order approval workflow (finance domain)."""
    t1 = AgentTask(
        description="Extract invoice fields (amount, vendor, due date) from PDF.",
        complexity_score=0.4,
        requires_code=False,
        requires_retrieval=True,
        timeout_ms=3000,
        metadata={"stage": "extraction", "domain": "finance"},
    )
    t2 = AgentTask(
        description="Validate invoice against purchase-order database.",
        complexity_score=0.5,
        requires_code=True,
        requires_retrieval=True,
        timeout_ms=4000,
        dependencies=[t1.task_id],
        metadata={"stage": "validation", "domain": "finance"},
    )
    t3 = AgentTask(
        description="Route to manager if amount exceeds approval threshold.",
        complexity_score=0.3,
        requires_code=False,
        requires_retrieval=False,
        timeout_ms=2000,
        dependencies=[t2.task_id],
        metadata={"stage": "routing", "domain": "finance"},
    )
    t4 = AgentTask(
        description="Post approved payment to ERP system and notify AP team.",
        complexity_score=0.6,
        requires_code=True,
        requires_retrieval=False,
        timeout_ms=5000,
        dependencies=[t3.task_id],
        metadata={"stage": "posting", "domain": "finance"},
    )
    return [t1, t2, t3, t4]


def make_hr_onboarding_workflow() -> List[AgentTask]:
    """New-employee onboarding workflow (HR domain)."""
    t1 = AgentTask(
        description="Collect signed offer letter and personal details from candidate.",
        complexity_score=0.2,
        requires_code=False,
        requires_retrieval=True,
        timeout_ms=2000,
        metadata={"stage": "intake", "domain": "hr"},
    )
    t2 = AgentTask(
        description="Run background check via third-party screening API.",
        complexity_score=0.5,
        requires_code=True,
        requires_retrieval=True,
        timeout_ms=8000,
        max_retries=3,
        dependencies=[t1.task_id],
        metadata={"stage": "screening", "domain": "hr"},
    )
    t3 = AgentTask(
        description="Provision SSO accounts, email, and Slack workspace.",
        complexity_score=0.6,
        requires_code=True,
        requires_retrieval=False,
        timeout_ms=6000,
        max_retries=2,
        dependencies=[t2.task_id],
        metadata={"stage": "provisioning", "domain": "hr"},
    )
    t4 = AgentTask(
        description="Schedule orientation sessions and assign onboarding buddy.",
        complexity_score=0.3,
        requires_code=False,
        requires_retrieval=False,
        timeout_ms=2000,
        dependencies=[t3.task_id],
        metadata={"stage": "scheduling", "domain": "hr"},
    )
    t5 = AgentTask(
        description="Send welcome email with first-week agenda to new hire.",
        complexity_score=0.1,
        requires_code=False,
        requires_retrieval=False,
        timeout_ms=1000,
        dependencies=[t4.task_id],
        metadata={"stage": "communication", "domain": "hr"},
    )
    return [t1, t2, t3, t4, t5]


def make_devops_deploy_workflow() -> List[AgentTask]:
    """CI/CD deployment pipeline workflow (DevOps domain)."""
    t1 = AgentTask(
        description="Run unit and integration test suite against PR branch.",
        complexity_score=0.7,
        requires_code=True,
        requires_retrieval=False,
        timeout_ms=30000,
        max_retries=1,
        metadata={"stage": "test", "domain": "devops"},
    )
    t2 = AgentTask(
        description="Build Docker image and push to container registry.",
        complexity_score=0.6,
        requires_code=True,
        requires_retrieval=False,
        timeout_ms=15000,
        max_retries=2,
        dependencies=[t1.task_id],
        metadata={"stage": "build", "domain": "devops"},
    )
    t3 = AgentTask(
        description="Run SAST and container vulnerability scan.",
        complexity_score=0.5,
        requires_code=True,
        requires_retrieval=True,
        timeout_ms=10000,
        dependencies=[t2.task_id],
        metadata={"stage": "security", "domain": "devops"},
    )
    t4 = AgentTask(
        description="Deploy to staging environment and run smoke tests.",
        complexity_score=0.8,
        requires_code=True,
        requires_retrieval=False,
        timeout_ms=20000,
        max_retries=1,
        dependencies=[t3.task_id],
        metadata={"stage": "staging", "domain": "devops"},
    )
    t5 = AgentTask(
        description="Promote image to production with canary rollout (5% traffic).",
        complexity_score=0.9,
        requires_code=True,
        requires_retrieval=False,
        timeout_ms=25000,
        dependencies=[t4.task_id],
        metadata={"stage": "production", "domain": "devops"},
    )
    return [t1, t2, t3, t4, t5]


def make_linear_pipeline(
    n_stages: int,
    domain: str = "generic",
    seed: int = 0,
) -> List[AgentTask]:
    """Generate an ``n_stages`` linear pipeline where each stage depends on the prior one.

    Used by Experiment 3 to measure cascade radius as a function of pipeline
    depth (e.g. 3-, 5-, 7-stage). Complexity / capability flags vary
    deterministically with ``seed`` so every run is reproducible.
    """
    if n_stages < 1:
        raise ValueError(f"n_stages must be >= 1, got {n_stages}")
    rng = random.Random(seed)
    tasks: List[AgentTask] = []
    prev_id: Optional[str] = None
    for i in range(n_stages):
        task = AgentTask(
            description=f"{domain} pipeline stage {i + 1}/{n_stages}",
            complexity_score=round(rng.uniform(0.2, 0.9), 3),
            requires_code=rng.random() < 0.5,
            requires_retrieval=rng.random() < 0.5,
            dependencies=[prev_id] if prev_id is not None else [],
            metadata={"stage": i, "domain": domain},
        )
        tasks.append(task)
        prev_id = task.task_id
    return tasks


def make_task(
    description: str,
    complexity: float = 0.5,
    requires_code: bool = False,
    requires_retrieval: bool = False,
    timeout_ms: float = 5000.0,
    max_retries: int = 2,
    dependencies: Optional[List[str]] = None,
) -> AgentTask:
    return AgentTask(
        description=description,
        complexity_score=complexity,
        requires_code=requires_code,
        requires_retrieval=requires_retrieval,
        timeout_ms=timeout_ms,
        max_retries=max_retries,
        dependencies=dependencies or [],
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
        status=TaskStatus.SUCCESS if success else TaskStatus.FAILED,
        n_subagent_calls=1,
    )
