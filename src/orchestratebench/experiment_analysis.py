"""Shared experiment summarization and pairwise-comparison helpers."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

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


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _metric_values(
    rows: list[dict[str, object]],
    source_key: str,
    *,
    positive_only: bool = False,
) -> list[float]:
    values = [float(row[source_key]) for row in rows]
    if positive_only:
        values = [value for value in values if value > 0.0]
    return values


def _summarize_group(
    rows: list[dict[str, object]],
    metric_specs: Dict[str, Dict[str, Any]],
    *,
    with_ci: bool = False,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    summary: dict[str, float] = {}
    for index, (metric_name, spec) in enumerate(metric_specs.items()):
        values = _metric_values(
            rows,
            str(spec["source"]),
            positive_only=bool(spec.get("positive_only", False)),
        )
        summary[metric_name] = _mean(values)
        if with_ci:
            low, high = bootstrap_ci(values, n_resamples=n_resamples, seed=seed + index)
            summary[f"{metric_name}_ci_low"] = low
            summary[f"{metric_name}_ci_high"] = high
    return summary


def _group_rows(
    rows: list[dict[str, object]],
    keys: Sequence[str],
) -> dict[tuple[object, ...], list[dict[str, object]]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(row[group_key] for group_key in keys)
        grouped.setdefault(key, []).append(row)
    return grouped


def _policy_pairs(
    policy_names: Sequence[str],
    baseline_policy: Optional[str] = None,
) -> list[tuple[str, str]]:
    names = sorted(policy_names)
    if baseline_policy is not None:
        if baseline_policy not in names:
            return []
        return [(name, baseline_policy) for name in names if name != baseline_policy]
    pairs: list[tuple[str, str]] = []
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            pairs.append((left, right))
    return pairs


def compare_policy_pairs(
    records: list[dict[str, object]],
    metric_specs: Dict[str, Dict[str, Any]],
    *,
    group_keys: Sequence[str],
    scenario_keys: Sequence[str],
    baseline_policy: Optional[str] = None,
    n_resamples: int = 2000,
    seed: int = 0,
) -> list[dict[str, object]]:
    """Compare policies on paired scenario-level records with bootstrap tests."""
    grouped = _group_rows(records, group_keys)
    comparison_rows: list[dict[str, object]] = []
    for group_index, group in enumerate(sorted(grouped)):
        rows = grouped[group]
        by_policy: dict[str, dict[tuple[object, ...], dict[str, object]]] = {}
        for row in rows:
            policy = str(row["policy"])
            scenario = tuple(row[key] for key in scenario_keys)
            by_policy.setdefault(policy, {})[scenario] = row

        for pair_index, (policy_a, policy_b) in enumerate(
            _policy_pairs(by_policy.keys(), baseline_policy=baseline_policy)
        ):
            common = sorted(set(by_policy[policy_a]).intersection(by_policy[policy_b]))
            if not common:
                continue
            base_row = {
                group_key: value for group_key, value in zip(group_keys, group)
            }
            base_row["policy_a"] = policy_a
            base_row["policy_b"] = policy_b
            base_row["n_pairs"] = len(common)

            for metric_index, (metric_name, spec) in enumerate(metric_specs.items()):
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
                    seed=seed + group_index * 1000 + pair_index * 100 + metric_index,
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


def summarize_failure_recovery_records(
    records: list[dict[str, object]],
    *,
    group_keys: Sequence[str] = ("policy", "failure_mode"),
    with_ci: bool = False,
    n_resamples: int = 2000,
    seed: int = 0,
) -> list[dict[str, object]]:
    """Summarize Exp 2 long-form records into grouped metrics."""
    grouped = _group_rows(records, group_keys)
    summary_rows: list[dict[str, object]] = []
    for group_index, key in enumerate(sorted(grouped)):
        rows = grouped[key]
        summary_row: dict[str, object] = {
            group_key: value for group_key, value in zip(group_keys, key)
        }
        summary_row["n_runs"] = len(rows)
        summary_row.update(
            _summarize_group(
                rows,
                FAILURE_RECOVERY_METRICS,
                with_ci=with_ci,
                n_resamples=n_resamples,
                seed=seed + group_index * 100,
            )
        )
        summary_rows.append(summary_row)
    return summary_rows


def summarize_cascade_records(
    records: list[dict[str, object]],
    *,
    group_keys: Sequence[str] = ("policy", "depth", "injection_stage"),
    with_ci: bool = False,
    n_resamples: int = 2000,
    seed: int = 0,
) -> list[dict[str, object]]:
    """Summarize Exp 3 long-form records into grouped metrics."""
    grouped = _group_rows(records, group_keys)
    summary_rows: list[dict[str, object]] = []
    for group_index, key in enumerate(sorted(grouped)):
        rows = grouped[key]
        summary_row: dict[str, object] = {
            group_key: value for group_key, value in zip(group_keys, key)
        }
        summary_row["n_runs"] = len(rows)
        summary_row.update(
            _summarize_group(
                rows,
                CASCADE_METRICS,
                with_ci=with_ci,
                n_resamples=n_resamples,
                seed=seed + group_index * 100,
            )
        )
        summary_rows.append(summary_row)
    return summary_rows
