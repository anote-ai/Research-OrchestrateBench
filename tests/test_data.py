"""Tests for orchestratebench data helpers."""

from __future__ import annotations

from orchestratebench.data import (
    make_benchmark_tasks,
    make_devops_deploy_workflow,
    make_finance_approval_workflow,
    make_hr_onboarding_workflow,
    make_task,
)


def test_make_task_defaults() -> None:
    t = make_task("hello")
    assert t.description == "hello"
    assert t.complexity_score == 0.5


def test_make_benchmark_tasks_count() -> None:
    tasks = make_benchmark_tasks(n=10)
    assert len(tasks) == 10


def test_make_benchmark_tasks_deterministic() -> None:
    t1 = make_benchmark_tasks(n=5, seed=99)
    t2 = make_benchmark_tasks(n=5, seed=99)
    assert [t.description for t in t1] == [t.description for t in t2]


def test_finance_approval_workflow_length() -> None:
    tasks = make_finance_approval_workflow()
    assert len(tasks) == 4


def test_hr_onboarding_workflow_length() -> None:
    tasks = make_hr_onboarding_workflow()
    assert len(tasks) == 5


def test_devops_deploy_workflow_length() -> None:
    tasks = make_devops_deploy_workflow()
    assert len(tasks) == 5


def test_workflow_dependencies_valid() -> None:
    """All dependency IDs in workflow tasks must refer to tasks in the workflow."""
    for make_fn in [
        make_finance_approval_workflow,
        make_hr_onboarding_workflow,
        make_devops_deploy_workflow,
    ]:
        tasks = make_fn()
        ids = {t.task_id for t in tasks}
        for task in tasks:
            for dep in task.dependencies:
                assert dep in ids, f"Unknown dependency {dep} in {make_fn.__name__}"
