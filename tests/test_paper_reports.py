"""Tests for publication-friendly Exp 2/3 report builders."""

from __future__ import annotations

from orchestratebench.paper_reports import (
    build_exp2_latex_tables,
    build_exp2_markdown_report,
    build_exp3_latex_tables,
    build_exp3_markdown_report,
    write_publication_artifacts,
)


def test_build_exp2_markdown_report_renders_tables_and_findings() -> None:
    summary_rows = [
        {
            "policy": "fixed",
            "failure_mode": "tool_invocation_error",
            "n_runs": 3,
            "recovery_rate": 0.0,
            "recovery_rate_ci_low": 0.0,
            "recovery_rate_ci_high": 0.0,
            "final_task_success_rate": 0.0,
            "mean_cascade_radius": 3.0,
            "mean_time_to_detection_ms": 410.0,
            "mean_escalation_latency_ms": 410.0,
        },
        {
            "policy": "retry(heuristic)",
            "failure_mode": "tool_invocation_error",
            "n_runs": 3,
            "recovery_rate": 1.0,
            "recovery_rate_ci_low": 1.0,
            "recovery_rate_ci_high": 1.0,
            "final_task_success_rate": 1.0,
            "mean_cascade_radius": 0.0,
            "mean_time_to_detection_ms": 60.0,
            "mean_escalation_latency_ms": 0.0,
        },
    ]
    pairwise_rows = [
        {
            "failure_mode": "tool_invocation_error",
            "policy_a": "fixed",
            "policy_b": "retry(heuristic)",
            "metric": "recovery_rate",
            "mean_a": 0.0,
            "mean_b": 1.0,
            "diff": -1.0,
            "ci_low": -1.0,
            "ci_high": -1.0,
            "p_value": 0.0,
        }
    ]
    text = build_exp2_markdown_report(
        summary_rows,
        pairwise_rows,
        config={"source_mode": "measured", "with_ci": True},
    )
    assert "# Experiment 2 Paper Summary" in text
    assert "Recovery Rate" in text
    assert "**1.000 [1.000, 1.000]**" in text
    assert "better policy: `retry(heuristic)`" in text


def test_build_exp2_latex_tables_renders_bold_best_cells() -> None:
    summary_rows = [
        {
            "policy": "fixed",
            "failure_mode": "tool_invocation_error",
            "recovery_rate": 0.0,
            "final_task_success_rate": 0.0,
            "mean_cascade_radius": 3.0,
            "mean_time_to_detection_ms": 410.0,
            "mean_escalation_latency_ms": 410.0,
        },
        {
            "policy": "retry(heuristic)",
            "failure_mode": "tool_invocation_error",
            "recovery_rate": 1.0,
            "final_task_success_rate": 1.0,
            "mean_cascade_radius": 0.0,
            "mean_time_to_detection_ms": 60.0,
            "mean_escalation_latency_ms": 0.0,
        },
    ]
    text = build_exp2_latex_tables(summary_rows)
    assert r"\begin{table}" in text
    assert r"\textbf{1.000}" in text


def test_build_exp3_markdown_report_renders_stage_and_depth_sections() -> None:
    stage_summary_rows = [
        {
            "policy": "fixed",
            "depth": 5,
            "injection_stage": 2,
            "mean_cascade_radius": 2.0,
            "mean_recovery_completeness": 0.0,
            "final_task_success_rate": 0.0,
            "mean_time_to_detection_ms": 820.0,
        },
        {
            "policy": "retry(heuristic)",
            "depth": 5,
            "injection_stage": 2,
            "mean_cascade_radius": 0.0,
            "mean_recovery_completeness": 1.0,
            "final_task_success_rate": 1.0,
            "mean_time_to_detection_ms": 205.0,
        },
    ]
    depth_summary_rows = [
        {
            "policy": "fixed",
            "depth": 5,
            "mean_cascade_radius": 2.0,
            "mean_recovery_completeness": 0.0,
            "final_task_success_rate": 0.0,
            "mean_time_to_detection_ms": 820.0,
        },
        {
            "policy": "retry(heuristic)",
            "depth": 5,
            "mean_cascade_radius": 0.0,
            "mean_recovery_completeness": 1.0,
            "final_task_success_rate": 1.0,
            "mean_time_to_detection_ms": 205.0,
        },
    ]
    pairwise_stage_rows = [
        {
            "depth": 5,
            "injection_stage": 2,
            "policy_a": "fixed",
            "policy_b": "retry(heuristic)",
            "metric": "mean_cascade_radius",
            "mean_a": 2.0,
            "mean_b": 0.0,
            "diff": 2.0,
            "ci_low": 2.0,
            "ci_high": 2.0,
            "p_value": 0.0,
        }
    ]
    pairwise_depth_rows = [
        {
            "depth": 5,
            "policy_a": "fixed",
            "policy_b": "retry(heuristic)",
            "metric": "mean_cascade_radius",
            "mean_a": 2.0,
            "mean_b": 0.0,
            "diff": 2.0,
            "ci_low": 2.0,
            "ci_high": 2.0,
            "p_value": 0.0,
        }
    ]
    text = build_exp3_markdown_report(
        stage_summary_rows,
        depth_summary_rows,
        pairwise_stage_rows,
        pairwise_depth_rows,
    )
    assert "Depth x Injection Stage Tables" in text
    assert "5-stage / inject-3" in text
    assert "Pairwise Findings by Depth" in text


def test_build_exp3_latex_tables_renders_both_table_blocks() -> None:
    stage_summary_rows = [
        {
            "policy": "fixed",
            "depth": 5,
            "injection_stage": 2,
            "mean_cascade_radius": 2.0,
            "mean_recovery_completeness": 0.0,
            "final_task_success_rate": 0.0,
            "mean_time_to_detection_ms": 820.0,
        }
    ]
    depth_summary_rows = [
        {
            "policy": "fixed",
            "depth": 5,
            "mean_cascade_radius": 2.0,
            "mean_recovery_completeness": 0.0,
            "final_task_success_rate": 0.0,
            "mean_time_to_detection_ms": 820.0,
        }
    ]
    text = build_exp3_latex_tables(stage_summary_rows, depth_summary_rows)
    assert text.count(r"\begin{table}") >= 2


def test_write_publication_artifacts_writes_both_files(tmp_path) -> None:
    write_publication_artifacts(
        tmp_path,
        markdown_text="# Summary\n",
        latex_text="% tables\n",
    )
    assert (tmp_path / "paper_summary.md").read_text(encoding="utf-8") == "# Summary\n"
    assert (tmp_path / "paper_tables.tex").read_text(encoding="utf-8") == "% tables\n"
