"""Tests for orchestratebench.experiments (Exp 2/3 harness)."""

from __future__ import annotations

import json

import pytest

from orchestratebench.core import FixedPolicy, HeuristicPolicy, RetryPolicy
from orchestratebench.data import make_linear_pipeline
from orchestratebench.experiments import (
    CASCADE_PAIRWISE_METRICS,
    FAILURE_RECOVERY_PAIRWISE_METRICS,
    collect_cascade_records,
    collect_failure_recovery_records,
    compare_policy_pairs,
    format_cascade_report,
    format_cascade_sweep_report,
    format_pairwise_report,
    format_recovery_report,
    run_cascade_by_depth,
    run_cascade_diagnostics_by_depth,
    run_cascade_stage_sweep,
    run_failure_recovery,
    summarize_cascade_records,
    summarize_failure_recovery_records,
    write_json_file,
    write_records_csv,
)
from orchestratebench.failures import FailureMode


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
                assert 0.0 <= stats["final_task_success_rate"] <= 1.0
                assert stats["mean_cascade_radius"] >= 0.0
                assert stats["mean_time_to_detection_ms"] >= 0.0

    def test_deterministic(self) -> None:
        policies = {"fixed": FixedPolicy()}
        a = run_failure_recovery(policies, n_runs=3, seed=1)
        b = run_failure_recovery(policies, n_runs=3, seed=1)
        assert a == b

    def test_retry_policy_is_supported(self) -> None:
        policies = {
            "retry(heuristic)": RetryPolicy(HeuristicPolicy(), failure_rate=0.0, seed=0)
        }
        out = run_failure_recovery(
            policies,
            modes=[FailureMode.TOOL_INVOCATION_ERROR],
            n_runs=2,
            seed=0,
        )
        stats = out["retry(heuristic)"][FailureMode.TOOL_INVOCATION_ERROR.value]
        assert stats["recovery_rate"] >= 0.0

    def test_collect_records_shape(self) -> None:
        policies = {"fixed": FixedPolicy()}
        records = collect_failure_recovery_records(
            policies,
            modes=[FailureMode.TOOL_INVOCATION_ERROR],
            n_runs=2,
            seed=0,
        )
        assert records
        assert {"policy", "workflow", "failure_mode", "run", "injection_stage"} <= set(records[0])

    def test_summary_rows_can_include_ci(self) -> None:
        policies = {"fixed": FixedPolicy()}
        records = collect_failure_recovery_records(
            policies,
            modes=[FailureMode.TOOL_INVOCATION_ERROR],
            n_runs=2,
            seed=0,
        )
        summary = summarize_failure_recovery_records(
            records,
            with_ci=True,
            n_resamples=50,
            seed=0,
        )
        row = summary[0]
        assert "recovery_rate_ci_low" in row
        assert "recovery_rate_ci_high" in row

    def test_pairwise_comparisons_shape(self) -> None:
        policies = {
            "fixed": FixedPolicy(),
            "retry": RetryPolicy(HeuristicPolicy(), failure_rate=0.0, seed=0),
        }
        records = collect_failure_recovery_records(
            policies,
            modes=[FailureMode.TOOL_INVOCATION_ERROR],
            n_runs=2,
            seed=0,
        )
        rows = compare_policy_pairs(
            records,
            FAILURE_RECOVERY_PAIRWISE_METRICS,
            group_keys=("failure_mode",),
            scenario_keys=("workflow", "run", "injection_stage"),
            n_resamples=50,
            seed=0,
        )
        assert rows
        assert {"policy_a", "policy_b", "metric", "diff", "p_value"} <= set(rows[0])


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

    def test_diagnostics_include_recovery_and_detection_fields(self) -> None:
        policies = {"retry": RetryPolicy(HeuristicPolicy(), failure_rate=0.0, seed=0)}
        out = run_cascade_diagnostics_by_depth(policies, depths=[3], n_runs=2, seed=0)
        stats = out["retry"][3]
        assert stats["mean_cascade_radius"] >= 0.0
        assert 0.0 <= stats["mean_recovery_completeness"] <= 1.0
        assert 0.0 <= stats["final_task_success_rate"] <= 1.0
        assert stats["mean_time_to_detection_ms"] >= 0.0

    def test_stage_sweep_includes_multiple_injection_points(self) -> None:
        policies = {"fixed": FixedPolicy()}
        out = run_cascade_stage_sweep(
            policies,
            depths=[4],
            injection_stages=[0, 1, 2],
            n_runs=2,
            seed=0,
        )
        assert set(out["fixed"][4]) == {0, 1, 2}

    def test_collect_cascade_records_shape(self) -> None:
        policies = {"fixed": FixedPolicy()}
        records = collect_cascade_records(
            policies,
            depths=[3],
            injection_stages=[0, 1],
            n_runs=2,
            seed=0,
        )
        assert len(records) == 4
        assert {"policy", "depth", "injection_stage", "cascade_radius"} <= set(records[0])

    def test_summarize_cascade_records_with_ci(self) -> None:
        policies = {"fixed": FixedPolicy()}
        records = collect_cascade_records(
            policies,
            depths=[3],
            injection_stages=[0],
            n_runs=2,
            seed=0,
        )
        summary = summarize_cascade_records(
            records,
            with_ci=True,
            n_resamples=50,
            seed=0,
        )
        row = summary[0]
        assert "mean_cascade_radius_ci_low" in row
        assert "mean_cascade_radius_ci_high" in row

    def test_cascade_pairwise_comparisons_shape(self) -> None:
        policies = {
            "fixed": FixedPolicy(),
            "retry": RetryPolicy(HeuristicPolicy(), failure_rate=0.0, seed=0),
        }
        records = collect_cascade_records(
            policies,
            depths=[3],
            injection_stages=[0],
            n_runs=2,
            seed=0,
        )
        rows = compare_policy_pairs(
            records,
            CASCADE_PAIRWISE_METRICS,
            group_keys=("depth", "injection_stage"),
            scenario_keys=("run",),
            n_resamples=50,
            seed=0,
        )
        assert rows
        assert rows[0]["metric"] == "mean_cascade_radius"


class TestFormatters:
    def test_reports_render(self) -> None:
        policies = {"fixed": FixedPolicy()}
        rec = run_failure_recovery(policies, n_runs=2, seed=0)
        cas = run_cascade_diagnostics_by_depth(policies, n_runs=2, seed=0)
        assert "recovery rate" in format_recovery_report(rec)
        assert "cascade radius" in format_cascade_report(cas)

    def test_cascade_sweep_report_renders(self) -> None:
        policies = {"fixed": FixedPolicy()}
        cas = run_cascade_stage_sweep(
            policies,
            depths=[3],
            injection_stages=[0, 1],
            n_runs=2,
            seed=0,
        )
        text = format_cascade_sweep_report(cas)
        assert "inject-1" in text
        assert "cascade radius" in text

    def test_pairwise_report_renders(self) -> None:
        rows = [
            {
                "failure_mode": "tool_invocation_error",
                "policy_a": "retry",
                "policy_b": "heuristic",
                "metric": "recovery_rate",
                "diff": 1.0,
                "ci_low": 0.5,
                "ci_high": 1.0,
                "p_value": 0.01,
            }
        ]
        text = format_pairwise_report(
            rows,
            title="pairwise",
            group_keys=("failure_mode",),
        )
        assert "recovery_rate" in text
        assert "tool_invocation_error" in text


class TestArtifactWriters:
    def test_write_records_csv(self, tmp_path) -> None:
        records = [{"b": 2, "a": 1}]
        path = tmp_path / "records.csv"
        write_records_csv(records, path)
        text = path.read_text(encoding="utf-8")
        assert "a,b" in text
        assert "1,2" in text

    def test_write_json_file(self, tmp_path) -> None:
        path = tmp_path / "summary.json"
        write_json_file({"x": 1}, path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload == {"x": 1}
