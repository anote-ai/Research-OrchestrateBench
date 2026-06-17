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

from typing import Dict, List, Optional

from .core import AgentTask
from .data import (
    make_devops_deploy_workflow,
    make_finance_approval_workflow,
    make_hr_onboarding_workflow,
    make_linear_pipeline,
)
from .failures import FailureMode, measure_cascade, recovery_rate_by_mode


def default_workflow_suite() -> Dict[str, List[AgentTask]]:
    """The three enterprise workflow families used across experiments."""
    return {
        "finance_approval": make_finance_approval_workflow(),
        "hr_onboarding": make_hr_onboarding_workflow(),
        "devops_deploy": make_devops_deploy_workflow(),
    }


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_failure_recovery(
    policies: Dict[str, object],
    workflows: Optional[Dict[str, List[AgentTask]]] = None,
    modes: Optional[List[FailureMode]] = None,
    n_runs: int = 20,
    seed: int = 0,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Experiment 2 — per-failure-mode recovery rate, compared across policies.

    For every policy and failure mode, injects the mode across all workflow
    suites and averages the recovery rate and cascade radius. Returns
    ``{policy_name: {mode_name: {"recovery_rate", "mean_cascade_radius"}}}``.
    Deterministic for a fixed ``seed``.
    """
    if workflows is None:
        workflows = default_workflow_suite()
    if modes is None:
        modes = list(FailureMode)

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for pname, policy in policies.items():
        per_mode_recovery: Dict[str, List[float]] = {m.value: [] for m in modes}
        per_mode_radius: Dict[str, List[float]] = {m.value: [] for m in modes}
        for wi, (_, tasks) in enumerate(sorted(workflows.items())):
            by_mode = recovery_rate_by_mode(
                tasks=tasks,
                policy=policy,
                modes=modes,
                n_runs=n_runs,
                seed=seed + wi,
            )
            for mname, stats in by_mode.items():
                per_mode_recovery[mname].append(stats["recovery_rate"])
                per_mode_radius[mname].append(stats["mean_cascade_radius"])
        results[pname] = {
            mname: {
                "recovery_rate": _mean(per_mode_recovery[mname]),
                "mean_cascade_radius": _mean(per_mode_radius[mname]),
            }
            for mname in per_mode_recovery
        }
    return results


def run_cascade_by_depth(
    policies: Dict[str, object],
    depths: Optional[List[int]] = None,
    mode: FailureMode = FailureMode.CONTEXT_POLLUTION,
    n_runs: int = 20,
    seed: int = 0,
) -> Dict[str, Dict[int, float]]:
    """Experiment 3 — mean cascade radius as a function of pipeline depth.

    For every policy and depth, injects ``mode`` at the first stage of an
    n-stage linear pipeline and averages the resulting cascade radius over
    ``n_runs`` reproducible pipelines. Returns
    ``{policy_name: {depth: mean_cascade_radius}}``. Deterministic for a fixed
    ``seed``.
    """
    if depths is None:
        depths = [3, 5, 7]

    results: Dict[str, Dict[int, float]] = {}
    for pname, policy in policies.items():
        by_depth: Dict[int, float] = {}
        for depth in depths:
            radii: List[float] = []
            for run in range(n_runs):
                tasks = make_linear_pipeline(depth, seed=seed + run)
                out = measure_cascade(
                    tasks=tasks,
                    policy=policy,
                    injection_stage=0,
                    failure_mode=mode,
                    seed=seed + run,
                )
                radii.append(float(out["cascade_radius"]))
            by_depth[depth] = _mean(radii)
        results[pname] = by_depth
    return results


def format_recovery_report(results: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    """Pretty-print the Exp 2 recovery table (recovery rate per mode per policy)."""
    modes = sorted({m for p in results.values() for m in p})
    lines = ["Experiment 2 - per-failure-mode recovery rate (higher = better)"]
    header = f"{'failure mode':<24}" + "".join(f"{p:>14}" for p in results)
    lines.append(header)
    lines.append("-" * len(header))
    for m in modes:
        row = f"{m:<24}" + "".join(f"{results[p][m]['recovery_rate']:>14.3f}" for p in results)
        lines.append(row)
    return "\n".join(lines)


def format_cascade_report(results: Dict[str, Dict[int, float]]) -> str:
    """Pretty-print the Exp 3 cascade-radius table (radius per depth per policy)."""
    depths = sorted({d for p in results.values() for d in p})
    lines = ["Experiment 3 - mean cascade radius by pipeline depth (lower = better)"]
    header = f"{'pipeline depth':<24}" + "".join(f"{p:>14}" for p in results)
    lines.append(header)
    lines.append("-" * len(header))
    for d in depths:
        row = f"{str(d) + '-stage':<24}" + "".join(f"{results[p][d]:>14.2f}" for p in results)
        lines.append(row)
    return "\n".join(lines)
