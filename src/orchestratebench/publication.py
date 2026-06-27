"""Publication-friendly report builders for Exp 2/3 artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Sequence

METRIC_METADATA: Dict[str, Dict[str, object]] = {
    "recovery_rate": {
        "label": "Recovery Rate",
        "direction": "higher",
        "decimals": 3,
    },
    "final_task_success_rate": {
        "label": "Final Task Success Rate",
        "direction": "higher",
        "decimals": 3,
    },
    "mean_cascade_radius": {
        "label": "Mean Cascade Radius",
        "direction": "lower",
        "decimals": 3,
    },
    "mean_recovery_completeness": {
        "label": "Mean Recovery Completeness",
        "direction": "higher",
        "decimals": 3,
    },
    "mean_time_to_detection_ms": {
        "label": "Mean Time to Detection (ms)",
        "direction": "lower",
        "decimals": 1,
    },
    "mean_escalation_latency_ms": {
        "label": "Mean Escalation Latency (ms)",
        "direction": "lower",
        "decimals": 1,
    },
    "escalation_rate": {
        "label": "Escalation Rate",
        "direction": "lower",
        "decimals": 3,
    },
}

EXP2_PRIMARY_METRICS = (
    "recovery_rate",
    "final_task_success_rate",
    "mean_cascade_radius",
    "mean_time_to_detection_ms",
    "mean_escalation_latency_ms",
)

EXP3_PRIMARY_METRICS = (
    "mean_cascade_radius",
    "mean_recovery_completeness",
    "final_task_success_rate",
    "mean_time_to_detection_ms",
)


def _group_rows(
    rows: Sequence[Dict[str, object]],
    keys: Sequence[str],
) -> Dict[tuple[object, ...], Dict[str, Dict[str, object]]]:
    grouped: Dict[tuple[object, ...], Dict[str, Dict[str, object]]] = {}
    for row in rows:
        group = tuple(row[key] for key in keys)
        grouped.setdefault(group, {})
        grouped[group][str(row["policy"])] = row
    return grouped


def _format_numeric(value: float, decimals: int) -> str:
    return f"{value:.{decimals}f}"


def _escape_latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    escaped = text
    for source, target in replacements.items():
        escaped = escaped.replace(source, target)
    return escaped


def _format_config_lines(config: Dict[str, object] | None) -> list[str]:
    if not config:
        return []
    keys = ("source_mode", "input_file", "with_ci", "n_resamples", "seed")
    lines: list[str] = []
    for key in keys:
        value = config.get(key)
        if value is None:
            continue
        lines.append(f"- `{key}`: `{value}`")
    return lines


def _metric_label(metric: str) -> str:
    meta = METRIC_METADATA.get(metric, {})
    return str(meta.get("label", metric))


def _metric_decimals(metric: str) -> int:
    meta = METRIC_METADATA.get(metric, {})
    return int(meta.get("decimals", 3))


def _metric_direction(metric: str) -> str:
    meta = METRIC_METADATA.get(metric, {})
    return str(meta.get("direction", "higher"))


def _group_label_exp2(values: tuple[object, ...]) -> str:
    return str(values[0]).replace("_", " ")


def _group_label_exp3_stage(values: tuple[object, ...]) -> str:
    depth = int(values[0])
    stage = int(values[1]) + 1
    return f"{depth}-stage / inject-{stage}"


def _group_label_exp3_depth(values: tuple[object, ...]) -> str:
    depth = int(values[0])
    return f"{depth}-stage"


def _best_policies(
    rows_by_policy: Dict[str, Dict[str, object]],
    metric: str,
) -> set[str]:
    direction = _metric_direction(metric)
    values = {policy: float(row[metric]) for policy, row in rows_by_policy.items()}
    if not values:
        return set()
    best_value = max(values.values()) if direction == "higher" else min(values.values())
    return {
        policy
        for policy, value in values.items()
        if abs(value - best_value) <= 1e-12
    }


def _render_metric_cell(
    row: Dict[str, object],
    metric: str,
    *,
    style: str,
    bold: bool,
) -> str:
    decimals = _metric_decimals(metric)
    value = _format_numeric(float(row[metric]), decimals)
    low_key = f"{metric}_ci_low"
    high_key = f"{metric}_ci_high"
    if low_key in row and high_key in row:
        ci_low = _format_numeric(float(row[low_key]), decimals)
        ci_high = _format_numeric(float(row[high_key]), decimals)
        text = f"{value} [{ci_low}, {ci_high}]"
    else:
        text = value

    if bold:
        if style == "latex":
            return r"\textbf{" + _escape_latex(text) + "}"
        return f"**{text}**"

    if style == "latex":
        return _escape_latex(text)
    return text


def _render_markdown_tables(
    rows: Sequence[Dict[str, object]],
    *,
    group_keys: Sequence[str],
    metrics: Sequence[str],
    group_labeler: Callable[[tuple[object, ...]], str],
    section_title: str,
) -> str:
    if not rows:
        return f"## {section_title}\n\nNo rows.\n"

    grouped = _group_rows(rows, group_keys)
    policies = sorted({str(row["policy"]) for row in rows})
    lines = [f"## {section_title}"]
    for metric in metrics:
        lines.extend(
            [
                "",
                f"### {_metric_label(metric)}",
                "",
                "| Group | " + " | ".join(policies) + " |",
                "|---|" + "|".join("---" for _ in policies) + "|",
            ]
        )
        for group in sorted(grouped):
            rows_by_policy = grouped[group]
            best = _best_policies(rows_by_policy, metric)
            cells = []
            for policy in policies:
                row = rows_by_policy.get(policy)
                if row is None:
                    cells.append("—")
                    continue
                cells.append(
                    _render_metric_cell(
                        row,
                        metric,
                        style="markdown",
                        bold=policy in best,
                    )
                )
            lines.append("| " + group_labeler(group) + " | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _render_latex_tables(
    rows: Sequence[Dict[str, object]],
    *,
    group_keys: Sequence[str],
    metrics: Sequence[str],
    group_labeler: Callable[[tuple[object, ...]], str],
    caption_prefix: str,
) -> str:
    if not rows:
        return "% no rows\n"

    grouped = _group_rows(rows, group_keys)
    policies = sorted({str(row["policy"]) for row in rows})
    chunks: list[str] = []
    for metric in metrics:
        chunks.extend(
            [
                r"\begin{table}[t]",
                r"\centering",
                rf"\caption{{{_escape_latex(caption_prefix)}: {_escape_latex(_metric_label(metric))}}}",
                r"\begin{tabular}{" + "l" + "c" * len(policies) + "}",
                r"\hline",
                "Group & " + " & ".join(_escape_latex(policy) for policy in policies) + r" \\",
                r"\hline",
            ]
        )
        for group in sorted(grouped):
            rows_by_policy = grouped[group]
            best = _best_policies(rows_by_policy, metric)
            cells = []
            for policy in policies:
                row = rows_by_policy.get(policy)
                if row is None:
                    cells.append("--")
                    continue
                cells.append(
                    _render_metric_cell(
                        row,
                        metric,
                        style="latex",
                        bold=policy in best,
                    )
                )
            chunks.append(
                _escape_latex(group_labeler(group)) + " & " + " & ".join(cells) + r" \\"
            )
        chunks.extend(
            [
                r"\hline",
                r"\end{tabular}",
                r"\end{table}",
                "",
            ]
        )
    return "\n".join(chunks).rstrip() + "\n"


def _better_policy(row: Dict[str, object]) -> str | None:
    metric = str(row["metric"])
    diff = float(row["diff"])
    direction = _metric_direction(metric)
    if abs(diff) <= 1e-12:
        return None
    if direction == "higher":
        return str(row["policy_a"]) if diff > 0 else str(row["policy_b"])
    return str(row["policy_a"]) if diff < 0 else str(row["policy_b"])


def _format_pairwise_findings(
    rows: Sequence[Dict[str, object]],
    *,
    group_keys: Sequence[str],
    group_labeler: Callable[[tuple[object, ...]], str],
    section_title: str,
    alpha: float = 0.05,
    max_findings: int = 8,
) -> str:
    lines = [f"## {section_title}"]
    if not rows:
        lines.append("")
        lines.append("No pairwise rows.")
        return "\n".join(lines) + "\n"

    significant: list[Dict[str, object]] = []
    win_counts: Dict[str, Dict[str, int]] = {}
    for row in rows:
        p_value = float(row["p_value"])
        ci_low = float(row["ci_low"])
        ci_high = float(row["ci_high"])
        if p_value > alpha or (ci_low <= 0.0 <= ci_high):
            continue
        better = _better_policy(row)
        if better is not None:
            metric = str(row["metric"])
            win_counts.setdefault(metric, {})
            win_counts[metric][better] = win_counts[metric].get(better, 0) + 1
        significant.append(row)

    lines.append("")
    if not significant:
        lines.append(f"No statistically significant pairwise differences at alpha={alpha:.2f}.")
        return "\n".join(lines) + "\n"

    lines.append("### Significant Win Counts")
    for metric in sorted(win_counts):
        counts = ", ".join(
            f"{policy}: {count}"
            for policy, count in sorted(
                win_counts[metric].items(),
                key=lambda item: (-item[1], item[0]),
            )
        )
        lines.append(f"- {_metric_label(metric)}: {counts}")

    lines.append("")
    lines.append("### Top Significant Findings")
    ranked = sorted(significant, key=lambda row: abs(float(row["diff"])), reverse=True)
    for row in ranked[:max_findings]:
        group = tuple(row[key] for key in group_keys)
        better = _better_policy(row) or "tie"
        metric = str(row["metric"])
        decimals = _metric_decimals(metric)
        lines.append(
            "- "
            f"{group_labeler(group)} | {_metric_label(metric)} | better policy: `{better}` | "
            f"{row['policy_a']}={_format_numeric(float(row['mean_a']), decimals)} vs "
            f"{row['policy_b']}={_format_numeric(float(row['mean_b']), decimals)} | "
            f"diff={_format_numeric(float(row['diff']), decimals)} | "
            f"95% CI [{_format_numeric(float(row['ci_low']), decimals)}, "
            f"{_format_numeric(float(row['ci_high']), decimals)}] | "
            f"p={float(row['p_value']):.3f}"
        )

    return "\n".join(lines) + "\n"


def build_exp2_markdown_report(
    summary_rows: Sequence[Dict[str, object]],
    pairwise_rows: Sequence[Dict[str, object]],
    *,
    config: Dict[str, object] | None = None,
) -> str:
    lines = ["# Experiment 2 Paper Summary"]
    config_lines = _format_config_lines(config)
    if config_lines:
        lines.extend(["", "## Provenance", *config_lines])
    lines.extend(
        [
            "",
            _render_markdown_tables(
                summary_rows,
                group_keys=("failure_mode",),
                metrics=EXP2_PRIMARY_METRICS,
                group_labeler=_group_label_exp2,
                section_title="Primary Tables",
            ).rstrip(),
            "",
            _format_pairwise_findings(
                pairwise_rows,
                group_keys=("failure_mode",),
                group_labeler=_group_label_exp2,
                section_title="Pairwise Findings",
            ).rstrip(),
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def build_exp2_latex_tables(
    summary_rows: Sequence[Dict[str, object]],
) -> str:
    return _render_latex_tables(
        summary_rows,
        group_keys=("failure_mode",),
        metrics=EXP2_PRIMARY_METRICS,
        group_labeler=_group_label_exp2,
        caption_prefix="Experiment 2",
    )


def build_exp3_markdown_report(
    stage_summary_rows: Sequence[Dict[str, object]],
    depth_summary_rows: Sequence[Dict[str, object]],
    pairwise_stage_rows: Sequence[Dict[str, object]],
    pairwise_depth_rows: Sequence[Dict[str, object]],
    *,
    config: Dict[str, object] | None = None,
) -> str:
    lines = ["# Experiment 3 Paper Summary"]
    config_lines = _format_config_lines(config)
    if config_lines:
        lines.extend(["", "## Provenance", *config_lines])
    lines.extend(
        [
            "",
            _render_markdown_tables(
                stage_summary_rows,
                group_keys=("depth", "injection_stage"),
                metrics=EXP3_PRIMARY_METRICS,
                group_labeler=_group_label_exp3_stage,
                section_title="Depth x Injection Stage Tables",
            ).rstrip(),
            "",
            _render_markdown_tables(
                depth_summary_rows,
                group_keys=("depth",),
                metrics=EXP3_PRIMARY_METRICS,
                group_labeler=_group_label_exp3_depth,
                section_title="Depth-Aggregated Tables",
            ).rstrip(),
            "",
            _format_pairwise_findings(
                pairwise_stage_rows,
                group_keys=("depth", "injection_stage"),
                group_labeler=_group_label_exp3_stage,
                section_title="Pairwise Findings by Depth x Stage",
            ).rstrip(),
            "",
            _format_pairwise_findings(
                pairwise_depth_rows,
                group_keys=("depth",),
                group_labeler=_group_label_exp3_depth,
                section_title="Pairwise Findings by Depth",
            ).rstrip(),
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def build_exp3_latex_tables(
    stage_summary_rows: Sequence[Dict[str, object]],
    depth_summary_rows: Sequence[Dict[str, object]],
) -> str:
    stage_tables = _render_latex_tables(
        stage_summary_rows,
        group_keys=("depth", "injection_stage"),
        metrics=EXP3_PRIMARY_METRICS,
        group_labeler=_group_label_exp3_stage,
        caption_prefix="Experiment 3 depth-stage",
    )
    depth_tables = _render_latex_tables(
        depth_summary_rows,
        group_keys=("depth",),
        metrics=EXP3_PRIMARY_METRICS,
        group_labeler=_group_label_exp3_depth,
        caption_prefix="Experiment 3 depth-aggregated",
    )
    return stage_tables.rstrip() + "\n\n" + depth_tables


def write_publication_artifacts(
    output_dir: str | Path,
    *,
    markdown_text: str,
    latex_text: str,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "paper_summary.md").write_text(markdown_text, encoding="utf-8")
    (output / "paper_tables.tex").write_text(latex_text, encoding="utf-8")
