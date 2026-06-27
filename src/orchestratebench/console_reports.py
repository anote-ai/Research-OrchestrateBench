"""Console-friendly text renderers for experiment outputs."""

from __future__ import annotations

from typing import Sequence


def format_recovery_report(results: dict[str, dict[str, dict[str, float]]]) -> str:
    """Pretty-print the Exp 2 recovery tables."""
    modes = sorted({mode for policy in results.values() for mode in policy})
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
        header = f"{'failure mode':<24}" + "".join(f"{policy:>18}" for policy in results)
        lines.append(header)
        lines.append("-" * len(header))
        for mode in modes:
            row = f"{mode:<24}" + "".join(
                f"{results[policy][mode][metric]:>18.3f}"
                for policy in results
            )
            lines.append(row)
    return "\n".join(lines)


def format_cascade_report(
    results: dict[str, dict[int, float]] | dict[str, dict[int, dict[str, float]]]
) -> str:
    """Pretty-print the Exp 3 cascade report."""
    depths = sorted({depth for policy in results.values() for depth in policy})
    sample_policy = next(iter(results))
    sample_value = results[sample_policy][depths[0]]

    if not isinstance(sample_value, dict):
        lines = ["Experiment 3 - mean cascade radius by pipeline depth (lower = better)"]
        header = f"{'pipeline depth':<24}" + "".join(f"{policy:>14}" for policy in results)
        lines.append(header)
        lines.append("-" * len(header))
        for depth in depths:
            row = f"{str(depth) + '-stage':<24}" + "".join(
                f"{results[policy][depth]:>14.2f}"  # type: ignore[index]
                for policy in results
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
        header = f"{'pipeline depth':<24}" + "".join(f"{policy:>18}" for policy in results)
        lines.append(header)
        lines.append("-" * len(header))
        for depth in depths:
            row = f"{str(depth) + '-stage':<24}" + "".join(
                f"{results[policy][depth][metric]:>18.3f}"  # type: ignore[index]
                for policy in results
            )
            lines.append(row)
    return "\n".join(lines)


def format_cascade_sweep_report(
    results: dict[str, dict[int, dict[int, dict[str, float]]]]
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
        header = f"{'depth / stage':<24}" + "".join(f"{policy:>18}" for policy in results)
        lines.append(header)
        lines.append("-" * len(header))
        for depth, stage in coordinates:
            row_label = f"{depth}-stage / inject-{stage + 1}"
            row = f"{row_label:<24}" + "".join(
                f"{results[policy][depth][stage][metric]:>18.3f}"
                for policy in results
            )
            lines.append(row)
    return "\n".join(lines)


def format_pairwise_report(
    rows: list[dict[str, object]],
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
