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

import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .core import AgentTask
from .data import (
    make_devops_deploy_workflow,
    make_finance_approval_workflow,
    make_hr_onboarding_workflow,
    make_linear_pipeline,
)
from .failures import FailureMode, measure_cascade
from .statistics import bootstrap_ci, paired_bootstrap_values

FAILURE_RECOVERY_METRICS: Dict[str, Dict[str, Any]] = {
    "recovery_rate": {"source": "injected_task_success"},
    "final_task_success_rate": {"source": "final_task_success"},
    "mean_cascade_radius": {"source": "cascade_radius"},
    "mean_recovery_completeness": {"source": "recovery_completeness"},
    "mean_time_to_detection_ms": {"source": "time_to_detection_ms"},
    "escalation_rate": {"source": "escalated"},
    "mean_escalation_latency_ms": {
        "source": "escalation_latency_ms",
        "positive_only": True,
    },
}

CASCADE_METRICS: Dict[str, Dict[str, Any]] = {
    "mean_cascade_radius": {"source": "cascade_radius"},
    "mean_recovery_completeness": {"source": "recovery_completeness"},
    "final_task_success_rate": {"source": "final_task_success"},
    "mean_time_to_detection_ms": {"source": "time_to_detection_ms"},
    "mean_escalation_latency_ms": {
        "source": "escalation_latency_ms",
        "positive_only": True,
    },
}

FAILURE_RECOVERY_PAIRWISE_METRICS: Dict[str, Dict[str, Any]] = {
    name: spec
    for name, spec in FAILURE_RECOVERY_METRICS.items()
    if not bool(spec.get("positive_only", False))
}

CASCADE_PAIRWISE_METRICS: Dict[str, Dict[str, Any]] = {
    name: spec
    for name, spec in CASCADE_METRICS.items()
    if not bool(spec.get("positive_only", False))
}


def default_workflow_suite() -> Dict[str, List[AgentTask]]:
    """The three enterprise workflow families used across experiments."""
    return {
        "finance_approval": make_finance_approval_workflow(),
        "hr_onboarding": make_hr_onboarding_workflow(),
        "devops_deploy": make_devops_deploy_workflow(),
    }


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mean_positive(values: List[float]) -> float:
    positive = [v for v in values if v > 0.0]
    return _mean(positive)


def _metric_values(
    rows: List[Dict[str, object]],
    source_key: str,
    *,
    positive_only: bool = False,
) -> List[float]:
    values = [float(row[source_key]) for row in rows]
    if positive_only:
        values = [v for v in values if v > 0.0]
    return values


def _summarize_group(
    rows: List[Dict[str, object]],
    metric_specs: Dict[str, Dict[str, Any]],
    *,
    with_ci: bool = False,
    n_resamples: int = 2000,
    seed: int = 0,
) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    for i, (metric_name, spec) in enumerate(metric_specs.items()):
        values = _metric_values(
            rows,
            str(spec["source"]),
            positive_only=bool(spec.get("positive_only", False)),
        )
        summary[metric_name] = _mean(values)
        if with_ci:
            low, high = bootstrap_ci(values, n_resamples=n_resamples, seed=seed + i)
            summary[f"{metric_name}_ci_low"] = low
            summary[f"{metric_name}_ci_high"] = high
    return summary


def _group_rows(
    rows: List[Dict[str, object]],
    keys: Sequence[str],
) -> Dict[tuple[object, ...], List[Dict[str, object]]]:
    grouped: Dict[tuple[object, ...], List[Dict[str, object]]] = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        grouped.setdefault(key, []).append(row)
    return grouped


def _policy_pairs(
    policy_names: Sequence[str],
    baseline_policy: Optional[str] = None,
) -> List[tuple[str, str]]:
    names = sorted(policy_names)
    if baseline_policy is not None:
        if baseline_policy not in names:
            return []
        return [(name, baseline_policy) for name in names if name != baseline_policy]
    pairs: List[tuple[str, str]] = []
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            pairs.append((left, right))
    return pairs


def compare_policy_pairs(
    records: List[Dict[str, object]],
    metric_specs: Dict[str, Dict[str, Any]],
    *,
    group_keys: Sequence[str],
    scenario_keys: Sequence[str],
    baseline_policy: Optional[str] = None,
    n_resamples: int = 2000,
    seed: int = 0,
) -> List[Dict[str, object]]:
    """Compare policies on paired scenario-level records with bootstrap tests."""
    grouped = _group_rows(records, group_keys)
    comparison_rows: List[Dict[str, object]] = []
    for gi, group in enumerate(sorted(grouped)):
        rows = grouped[group]
        by_policy: Dict[str, Dict[tuple[object, ...], Dict[str, object]]] = {}
        for row in rows:
            policy = str(row["policy"])
            scenario = tuple(row[key] for key in scenario_keys)
            by_policy.setdefault(policy, {})[scenario] = row

        for pair_i, (policy_a, policy_b) in enumerate(
            _policy_pairs(by_policy.keys(), baseline_policy=baseline_policy)
        ):
            common = sorted(set(by_policy[policy_a]).intersection(by_policy[policy_b]))
            if not common:
                continue
            base_row = {
                group_key: value
                for group_key, value in zip(group_keys, group)
            }
            base_row["policy_a"] = policy_a
            base_row["policy_b"] = policy_b
            base_row["n_pairs"] = len(common)

            for metric_j, (metric_name, spec) in enumerate(metric_specs.items()):
                values_a = [
                    float(by_policy[policy_a][scenario][str(spec["source"])])
                    for scenario in common
                ]
                values_b = [
                    float(by_policy[policy_b][scenario][str(spec["source"])])
                    for scenario in common
                ]
                result = paired_bootstrap_values(
                    values_a,
                    values_b,
                    n_resamples=n_resamples,
                    seed=seed + gi * 1000 + pair_i * 100 + metric_j,
                )
                comparison_rows.append(
                    {
                        **base_row,
                        "metric": metric_name,
                        "mean_a": float(result["mean_a"]),
                        "mean_b": float(result["mean_b"]),
                        "diff": float(result["diff"]),
                        "ci_low": float(result["ci"][0]),
                        "ci_high": float(result["ci"][1]),
                        "p_value": float(result["p_value"]),
                    }
                )
    return comparison_rows


def write_records_csv(records: List[Dict[str, object]], path: str | Path) -> None:
    """Write long-form experiment records to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for record in records for key in record})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_json_file(payload: object, path: str | Path) -> None:
    """Write experiment artifacts as pretty JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


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


def summarize_failure_recovery_records(
    records: List[Dict[str, object]],
    *,
    group_keys: Sequence[str] = ("policy", "failure_mode"),
    with_ci: bool = False,
    n_resamples: int = 2000,
    seed: int = 0,
) -> List[Dict[str, object]]:
    """Summarize Exp 2 long-form records into grouped metrics."""
    grouped = _group_rows(records, group_keys)
    summary_rows: List[Dict[str, object]] = []
    for gi, key in enumerate(sorted(grouped)):
        rows = grouped[key]
        summary_row: Dict[str, object] = {
            group_key: value for group_key, value in zip(group_keys, key)
        }
        summary_row["n_runs"] = len(rows)
        summary_row.update(
            _summarize_group(
                rows,
                FAILURE_RECOVERY_METRICS,
                with_ci=with_ci,
                n_resamples=n_resamples,
                seed=seed + gi * 100,
            )
        )
        summary_rows.append(summary_row)
    return summary_rows


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


def summarize_cascade_records(
    records: List[Dict[str, object]],
    *,
    group_keys: Sequence[str] = ("policy", "depth", "injection_stage"),
    with_ci: bool = False,
    n_resamples: int = 2000,
    seed: int = 0,
) -> List[Dict[str, object]]:
    """Summarize Exp 3 long-form records into grouped metrics."""
    grouped = _group_rows(records, group_keys)
    summary_rows: List[Dict[str, object]] = []
    for gi, key in enumerate(sorted(grouped)):
        rows = grouped[key]
        summary_row: Dict[str, object] = {
            group_key: value for group_key, value in zip(group_keys, key)
        }
        summary_row["n_runs"] = len(rows)
        summary_row.update(
            _summarize_group(
                rows,
                CASCADE_METRICS,
                with_ci=with_ci,
                n_resamples=n_resamples,
                seed=seed + gi * 100,
            )
        )
        summary_rows.append(summary_row)
    return summary_rows


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


def format_recovery_report(results: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    """Pretty-print the Exp 2 recovery tables."""
    modes = sorted({m for p in results.values() for m in p})
    lines = ["Experiment 2 - per-failure-mode recovery diagnostics"]
    metric_blocks = [
        ("recovery_rate", "Injected-stage recovery rate (higher = better)"),
        ("final_task_success_rate", "Final-task success rate (higher = better)"),
        ("mean_cascade_radius", "Mean cascade radius (lower = better)"),
        ("mean_time_to_detection_ms", "Mean time to detection, ms (lower = better)"),
        ("mean_escalation_latency_ms", "Mean escalation latency, ms (lower = better)"),
    ]
    for metric, title in metric_blocks:
        lines.append("")
        lines.append(title)
        header = f"{'failure mode':<24}" + "".join(f"{p:>18}" for p in results)
        lines.append(header)
        lines.append("-" * len(header))
        for m in modes:
            row = f"{m:<24}" + "".join(
                f"{results[p][m][metric]:>18.3f}"
                for p in results
            )
            lines.append(row)
    return "\n".join(lines)


def format_cascade_report(
    results: Dict[str, Dict[int, float]] | Dict[str, Dict[int, Dict[str, float]]]
) -> str:
    """Pretty-print the Exp 3 cascade report.

    Accepts either the legacy ``{policy: {depth: mean_cascade_radius}}`` shape
    or the richer diagnostics from ``run_cascade_diagnostics_by_depth``.
    """
    depths = sorted({d for p in results.values() for d in p})
    sample_policy = next(iter(results))
    sample_value = results[sample_policy][depths[0]]

    if not isinstance(sample_value, dict):
        lines = ["Experiment 3 - mean cascade radius by pipeline depth (lower = better)"]
        header = f"{'pipeline depth':<24}" + "".join(f"{p:>14}" for p in results)
        lines.append(header)
        lines.append("-" * len(header))
        for d in depths:
            row = f"{str(d) + '-stage':<24}" + "".join(
                f"{results[p][d]:>14.2f}"  # type: ignore[index]
                for p in results
            )
            lines.append(row)
        return "\n".join(lines)

    lines = ["Experiment 3 - cascade diagnostics by pipeline depth"]
    metric_blocks = [
        ("mean_cascade_radius", "Mean cascade radius (lower = better)"),
        ("mean_recovery_completeness", "Mean recovery completeness (higher = better)"),
        ("final_task_success_rate", "Final-task success rate (higher = better)"),
        ("mean_time_to_detection_ms", "Mean time to detection, ms (lower = better)"),
    ]
    for metric, title in metric_blocks:
        lines.append("")
        lines.append(title)
        header = f"{'pipeline depth':<24}" + "".join(f"{p:>18}" for p in results)
        lines.append(header)
        lines.append("-" * len(header))
        for d in depths:
            row = f"{str(d) + '-stage':<24}" + "".join(
                f"{results[p][d][metric]:>18.3f}"  # type: ignore[index]
                for p in results
            )
            lines.append(row)
    return "\n".join(lines)


def format_cascade_sweep_report(
    results: Dict[str, Dict[int, Dict[int, Dict[str, float]]]]
) -> str:
    """Pretty-print Exp 3 results for a depth x injection-stage sweep."""
    coordinates = sorted(
        {
            (depth, stage)
            for by_depth in results.values()
            for depth, by_stage in by_depth.items()
            for stage in by_stage
        }
    )
    lines = ["Experiment 3 - cascade diagnostics by depth and injection stage"]
    metric_blocks = [
        ("mean_cascade_radius", "Mean cascade radius (lower = better)"),
        ("mean_recovery_completeness", "Mean recovery completeness (higher = better)"),
        ("final_task_success_rate", "Final-task success rate (higher = better)"),
        ("mean_time_to_detection_ms", "Mean time to detection, ms (lower = better)"),
    ]
    for metric, title in metric_blocks:
        lines.append("")
        lines.append(title)
        header = f"{'depth / stage':<24}" + "".join(f"{p:>18}" for p in results)
        lines.append(header)
        lines.append("-" * len(header))
        for depth, stage in coordinates:
            row_label = f"{depth}-stage / inject-{stage + 1}"
            row = f"{row_label:<24}" + "".join(
                f"{results[p][depth][stage][metric]:>18.3f}"
                for p in results
            )
            lines.append(row)
    return "\n".join(lines)


def format_pairwise_report(
    rows: List[Dict[str, object]],
    *,
    title: str,
    group_keys: Sequence[str],
) -> str:
    """Pretty-print pairwise policy comparisons from long-form comparison rows."""
    if not rows:
        return f"{title}\n(no rows)"

    metrics = sorted({str(row["metric"]) for row in rows})
    lines = [title]
    for metric in metrics:
        lines.append("")
        lines.append(metric)
        header = (
            f"{'group':<28}{'policy_a':<18}{'policy_b':<18}"
            f"{'diff':>10}{'ci_low':>10}{'ci_high':>10}{'p':>10}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        metric_rows = [row for row in rows if str(row["metric"]) == metric]
        for row in metric_rows:
            group_label = ", ".join(str(row[key]) for key in group_keys)
            lines.append(
                f"{group_label:<28}"
                f"{str(row['policy_a']):<18}"
                f"{str(row['policy_b']):<18}"
                f"{float(row['diff']):>10.3f}"
                f"{float(row['ci_low']):>10.3f}"
                f"{float(row['ci_high']):>10.3f}"
                f"{float(row['p_value']):>10.3f}"
            )
    return "\n".join(lines)
