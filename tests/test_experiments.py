"""Tests for orchestratebench.experiments (Exp 2/3 harness)."""

from __future__ import annotations

from orchestratebench.core import FixedPolicy, HeuristicPolicy
from orchestratebench.data import make_linear_pipeline
from orchestratebench.experiments import (
    format_cascade_report,
    format_recovery_report,
    run_cascade_by_depth,
    run_failure_recovery,
)
from orchestratebench.failures import FailureMode

import pytest


class TestLinearPipeline:
    def test_depth_and_dependency_chain(self) -> None:
        tasks = make_linear_pipeline(5, seed=0)
        assert len(tasks) == 5
        assert tasks[0].dependencies == []
        for prev, cur in zip(tasks, tasks[1:]):
            assert cur.dependencies == [prev.task_id]

    def test_reproducible_for_seed(self) -> None:
        a = make_linear_pipeline(4, seed=7)
        b = make_linear_pipeline(4, seed=7)
        assert [t.complexity_score for t in a] == [t.complexity_score for t in b]

    def test_rejects_nonpositive_depth(self) -> None:
        with pytest.raises(ValueError):
            make_linear_pipeline(0)


class TestRunFailureRecovery:
    def test_structure_and_modes(self) -> None:
        policies = {"fixed": FixedPolicy(), "heuristic": HeuristicPolicy()}
        out = run_failure_recovery(policies, n_runs=3, seed=0)
        assert set(out) == {"fixed", "heuristic"}
        for by_mode in out.values():
            assert set(by_mode) == {m.value for m in FailureMode}
            for stats in by_mode.values():
                assert 0.0 <= stats["recovery_rate"] <= 1.0
                assert stats["mean_cascade_radius"] >= 0.0

    def test_deterministic(self) -> None:
        policies = {"fixed": FixedPolicy()}
        a = run_failure_recovery(policies, n_runs=3, seed=1)
        b = run_failure_recovery(policies, n_runs=3, seed=1)
        assert a == b


class TestRunCascadeByDepth:
    def test_structure_and_depths(self) -> None:
        policies = {"fixed": FixedPolicy()}
        out = run_cascade_by_depth(policies, depths=[3, 5, 7], n_runs=3, seed=0)
        assert set(out["fixed"]) == {3, 5, 7}
        for radius in out["fixed"].values():
            assert radius >= 0.0

    def test_deterministic(self) -> None:
        policies = {"heuristic": HeuristicPolicy()}
        a = run_cascade_by_depth(policies, n_runs=3, seed=2)
        b = run_cascade_by_depth(policies, n_runs=3, seed=2)
        assert a == b


class TestFormatters:
    def test_reports_render(self) -> None:
        policies = {"fixed": FixedPolicy()}
        rec = run_failure_recovery(policies, n_runs=2, seed=0)
        cas = run_cascade_by_depth(policies, n_runs=2, seed=0)
        assert "recovery rate" in format_recovery_report(rec)
        assert "cascade radius" in format_cascade_report(cas)
