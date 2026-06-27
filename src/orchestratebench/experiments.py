"""Experiment runners for failure-recovery (Exp 2) and cascade-depth (Exp 3).

These tie the workflow suites (``data``), routing policies (``core``), and the
failure-injection primitives (``failures``) into reproducible, cross-policy
experiment runs — the harness behind issues #4 (failure taxonomy) and #7
(cascade propagation).

The harness itself runs **offline and deterministically from a seed** using
simulated execution traces; the measured paper numbers come from the
collaborative gold-labeled run. Treat the output of these runners as a
mechanism demo of the harness, not as the reportable results.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from .console_reports import (
    format_cascade_report,
    format_cascade_sweep_report,
    format_pairwise_report,
    format_recovery_report,
)
from .core import AgentTask
from .data import (
    make_devops_deploy_workflow,
    make_finance_approval_workflow,
    make_hr_onboarding_workflow,
    make_linear_pipeline,
)
from .experiment_analysis import (
    CASCADE_PAIRWISE_METRICS,
    FAILURE_RECOVERY_PAIRWISE_METRICS,
    compare_policy_pairs,
    summarize_cascade_records,
    summarize_failure_recovery_records,
)
from .experiment_artifacts import write_json_file, write_records_csv
from .failures import FailureMode, measure_cascade

__all__ = [
    "CASCADE_PAIRWISE_METRICS",
    "FAILURE_RECOVERY_PAIRWISE_METRICS",
    "collect_cascade_records",
    "collect_failure_recovery_records",
    "compare_policy_pairs",
    "default_workflow_suite",
    "format_cascade_report",
    "format_cascade_sweep_report",
    "format_pairwise_report",
    "format_recovery_report",
    "run_cascade_by_depth",
    "run_cascade_diagnostics_by_depth",
    "run_cascade_stage_sweep",
    "run_failure_recovery",
    "summarize_cascade_records",
    "summarize_failure_recovery_records",
    "write_json_file",
    "write_records_csv",
]


def default_workflow_suite() -> Dict[str, List[AgentTask]]:
    """The three enterprise workflow families used across experiments."""
    return {
        "finance_approval": make_finance_approval_workflow(),
        "hr_onboarding": make_hr_onboarding_workflow(),
        "devops_deploy": make_devops_deploy_workflow(),
    }


def collect_failure_recovery_records(
    policies: Dict[str, object],
    workflows: Optional[Dict[str, List[AgentTask]]] = None,
    modes: Optional[List[FailureMode]] = None,
    n_runs: int = 20,
    seed: int = 0,
) -> List[Dict[str, object]]:
    """Collect long-form Exp 2 records for every policy/workflow/mode/run."""
    if workflows is None:
        workflows = default_workflow_suite()
    if modes is None:
        modes = list(FailureMode)

    records: List[Dict[str, object]] = []
    sorted_workflows = sorted(workflows.items())
    for wi, (workflow_name, tasks) in enumerate(sorted_workflows):
        for mi, mode in enumerate(modes):
            stage_rng = random.Random(seed + wi * 1000 + mi * 100)
            for run in range(n_runs):
                injection_stage = stage_rng.randint(0, max(0, len(tasks) - 2))
                trial_seed = seed + wi * 10_000 + mi * 1000 + run
                for policy_name, policy in policies.items():
                    result = measure_cascade(
                        tasks=tasks,
                        policy=policy,
                        injection_stage=injection_stage,
                        failure_mode=mode,
                        seed=trial_seed,
                    )
                    records.append(
                        {
                            "policy": policy_name,
                            "workflow": workflow_name,
                            "failure_mode": mode.value,
                            "run": run,
                            "seed": trial_seed,
                            "injection_stage": injection_stage,
                            "injected_task_success": float(result["injected_task_success"]),
                            "final_task_success": float(result["final_task_success"]),
                            "cascade_radius": float(result["cascade_radius"]),
                            "recovery_completeness": float(result["recovery_completeness"]),
                            "time_to_detection_ms": float(result["time_to_detection_ms"]),
                            "escalated": float(result["escalated"]),
                            "escalation_latency_ms": float(result["escalation_latency_ms"]),
                        }
                    )
    return records

def _failure_summary_rows_to_nested(
    summary_rows: List[Dict[str, object]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    nested: Dict[str, Dict[str, Dict[str, float]]] = {}
    for row in summary_rows:
        policy = str(row["policy"])
        mode = str(row["failure_mode"])
        nested.setdefault(policy, {})
        nested[policy][mode] = {
            key: float(value)
            for key, value in row.items()
            if key not in {"policy", "failure_mode", "n_runs"}
        }
    return nested


def collect_cascade_records(
    policies: Dict[str, object],
    depths: Optional[List[int]] = None,
    injection_stages: Optional[List[int]] = None,
    mode: FailureMode = FailureMode.CONTEXT_POLLUTION,
    n_runs: int = 20,
    seed: int = 0,
) -> List[Dict[str, object]]:
    """Collect long-form Exp 3 records for every policy/depth/stage/run."""
    if depths is None:
        depths = [3, 5, 7]
    if injection_stages is None:
        injection_stages = [0]

    records: List[Dict[str, object]] = []
    for depth in depths:
        valid_stages = [stage for stage in injection_stages if 0 <= stage < depth]
        for stage in valid_stages:
            for run in range(n_runs):
                pipeline_seed = seed + run
                tasks = make_linear_pipeline(depth, seed=pipeline_seed)
                for policy_name, policy in policies.items():
                    result = measure_cascade(
                        tasks=tasks,
                        policy=policy,
                        injection_stage=stage,
                        failure_mode=mode,
                        seed=pipeline_seed,
                    )
                    records.append(
                        {
                            "policy": policy_name,
                            "depth": depth,
                            "injection_stage": stage,
                            "failure_mode": mode.value,
                            "run": run,
                            "seed": pipeline_seed,
                            "injected_task_success": float(result["injected_task_success"]),
                            "final_task_success": float(result["final_task_success"]),
                            "cascade_radius": float(result["cascade_radius"]),
                            "recovery_completeness": float(result["recovery_completeness"]),
                            "time_to_detection_ms": float(result["time_to_detection_ms"]),
                            "escalated": float(result["escalated"]),
                            "escalation_latency_ms": float(result["escalation_latency_ms"]),
                        }
                    )
    return records

def _cascade_summary_rows_to_nested(
    summary_rows: List[Dict[str, object]],
    *,
    group_by_stage: bool,
) -> Dict[str, Any]:
    nested: Dict[str, Any] = {}
    for row in summary_rows:
        policy = str(row["policy"])
        depth = int(row["depth"])
        payload = {
            key: float(value)
            for key, value in row.items()
            if key not in {"policy", "depth", "injection_stage", "n_runs"}
        }
        nested.setdefault(policy, {})
        if group_by_stage:
            stage = int(row["injection_stage"])
            nested[policy].setdefault(depth, {})
            nested[policy][depth][stage] = payload
        else:
            nested[policy][depth] = payload
    return nested


def run_failure_recovery(
    policies: Dict[str, object],
    workflows: Optional[Dict[str, List[AgentTask]]] = None,
    modes: Optional[List[FailureMode]] = None,
    n_runs: int = 20,
    seed: int = 0,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Experiment 2 — per-failure-mode recovery rate, compared across policies.

    For every policy and failure mode, injects the mode across all workflow
    suites and averages the recovery, end-to-end success, detection, and
    cascade metrics. Returns
    ``{policy_name: {mode_name: {"recovery_rate", "final_task_success_rate",``
    ``"mean_cascade_radius", "mean_recovery_completeness",``
    ``"mean_time_to_detection_ms", "escalation_rate",``
    ``"mean_escalation_latency_ms"}}}``. Deterministic for a fixed ``seed``.
    """
    records = collect_failure_recovery_records(
        policies=policies,
        workflows=workflows,
        modes=modes,
        n_runs=n_runs,
        seed=seed,
    )
    summary_rows = summarize_failure_recovery_records(records)
    return _failure_summary_rows_to_nested(summary_rows)


def run_cascade_by_depth(
    policies: Dict[str, object],
    depths: Optional[List[int]] = None,
    mode: FailureMode = FailureMode.CONTEXT_POLLUTION,
    n_runs: int = 20,
    injection_stage: int = 0,
    seed: int = 0,
) -> Dict[str, Dict[int, float]]:
    """Backward-compatible wrapper returning mean cascade radius only."""
    detailed = run_cascade_diagnostics_by_depth(
        policies=policies,
        depths=depths,
        mode=mode,
        n_runs=n_runs,
        injection_stage=injection_stage,
        seed=seed,
    )
    return {
        pname: {
            depth: stats["mean_cascade_radius"]
            for depth, stats in by_depth.items()
        }
        for pname, by_depth in detailed.items()
    }


def run_cascade_diagnostics_by_depth(
    policies: Dict[str, object],
    depths: Optional[List[int]] = None,
    mode: FailureMode = FailureMode.CONTEXT_POLLUTION,
    n_runs: int = 20,
    injection_stage: int = 0,
    seed: int = 0,
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Experiment 3 — mean cascade radius as a function of pipeline depth.

    For every policy and depth, injects ``mode`` at the first stage of an
    n-stage linear pipeline and averages the resulting cascade, recovery, and
    detection metrics over ``n_runs`` reproducible pipelines. Returns
    ``{policy_name: {depth: {"mean_cascade_radius",``
    ``"mean_recovery_completeness", "final_task_success_rate",``
    ``"mean_time_to_detection_ms", "mean_escalation_latency_ms"}}}``.
    Deterministic for a fixed ``seed``.
    """
    records = collect_cascade_records(
        policies=policies,
        depths=depths,
        injection_stages=[injection_stage],
        mode=mode,
        n_runs=n_runs,
        seed=seed,
    )
    summary_rows = summarize_cascade_records(
        records,
        group_keys=("policy", "depth"),
    )
    return _cascade_summary_rows_to_nested(summary_rows, group_by_stage=False)


def run_cascade_stage_sweep(
    policies: Dict[str, object],
    depths: Optional[List[int]] = None,
    injection_stages: Optional[List[int]] = None,
    mode: FailureMode = FailureMode.CONTEXT_POLLUTION,
    n_runs: int = 20,
    seed: int = 0,
) -> Dict[str, Dict[int, Dict[int, Dict[str, float]]]]:
    """Experiment 3 sweep across both pipeline depth and injection stage."""
    records = collect_cascade_records(
        policies=policies,
        depths=depths,
        injection_stages=injection_stages,
        mode=mode,
        n_runs=n_runs,
        seed=seed,
    )
    summary_rows = summarize_cascade_records(records)
    return _cascade_summary_rows_to_nested(summary_rows, group_by_stage=True)
