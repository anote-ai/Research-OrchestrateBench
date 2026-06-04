"""Tests for orchestratebench.data."""

import pytest
from orchestratebench.core import AgentTask, ExecutionTrace
from orchestratebench.data import (
    SAMPLE_TASKS_RAW,
    make_benchmark_tasks,
    make_execution_trace,
    make_task,
)


def test_sample_tasks_raw_length():
    assert len(SAMPLE_TASKS_RAW) == 8


def test_make_task_returns_agent_task():
    t = make_task("do something", complexity=0.6)
    assert isinstance(t, AgentTask)
    assert t.complexity_score == pytest.approx(0.6)


def test_make_benchmark_tasks_count():
    tasks = make_benchmark_tasks(n=15, seed=1)
    assert len(tasks) == 15
    assert all(isinstance(t, AgentTask) for t in tasks)


def test_make_execution_trace_returns_trace():
    task = make_task("demo task")
    trace = make_execution_trace(task, success=True)
    assert isinstance(trace, ExecutionTrace)
    assert trace.success is True
