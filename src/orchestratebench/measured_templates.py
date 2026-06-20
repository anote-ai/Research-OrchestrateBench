"""Scaffold builders for measured Exp 2/3 long-form input files."""

from __future__ import annotations

import random
from typing import Dict, List, Sequence

from .data import (
    make_devops_deploy_workflow,
    make_finance_approval_workflow,
    make_hr_onboarding_workflow,
)
from .failures import FailureMode

DEFAULT_MEASURED_POLICIES: tuple[str, ...] = (
    "fixed",
    "heuristic",
    "retry(heuristic)",
)

EXP2_SKELETON_FIELDS: tuple[str, ...] = (
    "policy",
    "workflow",
    "failure_mode",
    "run",
    "seed",
    "injection_stage",
    "scenario_id",
    "injected_task_success",
    "final_task_success",
    "cascade_radius",
    "recovery_completeness",
    "time_to_detection_ms",
    "escalated",
    "escalation_latency_ms",
    "annotator",
    "notes",
)

EXP3_SKELETON_FIELDS: tuple[str, ...] = (
    "policy",
    "depth",
    "failure_mode",
    "run",
    "seed",
    "injection_stage",
    "scenario_id",
    "injected_task_success",
    "final_task_success",
    "cascade_radius",
    "recovery_completeness",
    "time_to_detection_ms",
    "escalated",
    "escalation_latency_ms",
    "annotator",
    "notes",
)


def _blank_metric_fields() -> Dict[str, str]:
    return {
        "injected_task_success": "",
        "final_task_success": "",
        "cascade_radius": "",
        "recovery_completeness": "",
        "time_to_detection_ms": "",
        "escalated": "",
        "escalation_latency_ms": "",
        "annotator": "",
        "notes": "",
    }


def _default_workflow_lengths() -> Dict[str, int]:
    return {
        "finance_approval": len(make_finance_approval_workflow()),
        "hr_onboarding": len(make_hr_onboarding_workflow()),
        "devops_deploy": len(make_devops_deploy_workflow()),
    }


def build_exp2_measured_skeleton(
    *,
    policies: Sequence[str] = DEFAULT_MEASURED_POLICIES,
    workflows: Sequence[str] = ("finance_approval", "hr_onboarding", "devops_deploy"),
    modes: Sequence[FailureMode] = tuple(FailureMode),
    n_runs: int = 5,
    seed: int = 0,
) -> List[Dict[str, object]]:
    """Build an Exp 2 long-form CSV scaffold with blank metric cells."""
    if n_runs < 1:
        raise ValueError(f"n_runs must be >= 1, got {n_runs}")

    workflow_lengths = _default_workflow_lengths()
    rows: List[Dict[str, object]] = []
    for wi, workflow_name in enumerate(workflows):
        if workflow_name not in workflow_lengths:
            raise ValueError(f"Unknown workflow: {workflow_name}")
        workflow_len = workflow_lengths[workflow_name]
        for mi, mode in enumerate(modes):
            stage_rng = random.Random(seed + wi * 1000 + mi * 100)
            for run in range(n_runs):
                injection_stage = stage_rng.randint(0, max(0, workflow_len - 2))
                trial_seed = seed + wi * 10_000 + mi * 1000 + run
                scenario_id = (
                    f"{workflow_name}__{mode.value}__inj{injection_stage}__run{run:04d}"
                )
                for policy in policies:
                    rows.append(
                        {
                            "policy": policy,
                            "workflow": workflow_name,
                            "failure_mode": mode.value,
                            "run": run,
                            "seed": trial_seed,
                            "injection_stage": injection_stage,
                            "scenario_id": scenario_id,
                            **_blank_metric_fields(),
                        }
                    )
    return rows


def build_exp3_measured_skeleton(
    *,
    policies: Sequence[str] = DEFAULT_MEASURED_POLICIES,
    depths: Sequence[int] = (3, 5, 7),
    injection_stages: Sequence[int] = (0, 1, 2),
    mode: FailureMode = FailureMode.CONTEXT_POLLUTION,
    n_runs: int = 5,
    seed: int = 0,
) -> List[Dict[str, object]]:
    """Build an Exp 3 long-form CSV scaffold with blank metric cells."""
    if n_runs < 1:
        raise ValueError(f"n_runs must be >= 1, got {n_runs}")

    rows: List[Dict[str, object]] = []
    for depth in depths:
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        valid_stages = [stage for stage in injection_stages if 0 <= stage < depth]
        for stage in valid_stages:
            for run in range(n_runs):
                trial_seed = seed + run
                scenario_id = f"depth{depth}__{mode.value}__inj{stage}__run{run:04d}"
                for policy in policies:
                    rows.append(
                        {
                            "policy": policy,
                            "depth": depth,
                            "failure_mode": mode.value,
                            "run": run,
                            "seed": trial_seed,
                            "injection_stage": stage,
                            "scenario_id": scenario_id,
                            **_blank_metric_fields(),
                        }
                    )
    return rows
