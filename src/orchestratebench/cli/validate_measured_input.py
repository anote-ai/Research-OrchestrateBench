"""Validate collaborative measured Exp 2/3 inputs before analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

from orchestratebench.measured_runs import (
    analyze_pairwise_compatibility,
    load_exp2_measured_records,
    load_exp3_measured_records,
    prefer_scenario_id,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        required=True,
        choices=["2", "3", "exp2", "exp3"],
        help="Measured dataset type to validate.",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="Measured input file (.csv, .jsonl, or .json).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero on warnings, not just schema errors.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _format_values(values: Iterable[object]) -> str:
    items = [str(value) for value in values]
    return ", ".join(items) if items else "-"


def _format_group(payload: Dict[str, Any], keys: Sequence[str]) -> str:
    return ", ".join(f"{key}={payload[key]}" for key in keys)


def _format_example_list(
    items: Sequence[Dict[str, Any]],
    *,
    formatter,
    limit: int = 3,
) -> str:
    if not items:
        return "-"
    return " | ".join(formatter(item) for item in items[:limit])


def _scenario_id_coverage(records: Sequence[Dict[str, Any]]) -> tuple[int, int]:
    present = sum(1 for record in records if str(record.get("scenario_id") or "").strip())
    return present, len(records)


def _pairwise_checks(
    experiment: str,
    records: Sequence[Dict[str, Any]],
) -> list[dict[str, Any]]:
    if experiment == "2":
        return [
            {
                "label": "by failure_mode",
                "group_keys": ("failure_mode",),
                "scenario_keys": prefer_scenario_id(
                    records,
                    fallback=("workflow", "run", "injection_stage"),
                ),
            },
            {
                "label": "by workflow,failure_mode",
                "group_keys": ("workflow", "failure_mode"),
                "scenario_keys": prefer_scenario_id(
                    records,
                    fallback=("run", "injection_stage"),
                ),
            },
        ]
    return [
        {
            "label": "by depth,injection_stage",
            "group_keys": ("depth", "injection_stage"),
            "scenario_keys": prefer_scenario_id(records, fallback=("run",)),
        },
        {
            "label": "by depth",
            "group_keys": ("depth",),
            "scenario_keys": prefer_scenario_id(records, fallback=("injection_stage", "run")),
        },
    ]


def _dataset_summary_lines(
    experiment: str,
    records: Sequence[Dict[str, Any]],
) -> list[str]:
    policies = sorted({str(record["policy"]) for record in records})
    failure_modes = sorted({str(record["failure_mode"]) for record in records})
    injection_stages = sorted({int(record["injection_stage"]) for record in records})
    present, total = _scenario_id_coverage(records)
    coverage_pct = (present / total * 100.0) if total else 0.0

    lines = [
        f"Rows: {len(records)}",
        f"Policies ({len(policies)}): {_format_values(policies)}",
        f"Failure modes ({len(failure_modes)}): {_format_values(failure_modes)}",
        f"Injection stages: {_format_values(injection_stages)}",
        f"scenario_id coverage: {present}/{total} ({coverage_pct:.1f}%)",
    ]
    if experiment == "2":
        workflows = sorted({str(record["workflow"]) for record in records})
        lines.insert(3, f"Workflows ({len(workflows)}): {_format_values(workflows)}")
    else:
        depths = sorted({int(record["depth"]) for record in records})
        lines.insert(3, f"Depths ({len(depths)}): {_format_values(depths)}")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    experiment = "2" if args.experiment in {"2", "exp2"} else "3"

    if experiment == "2":
        records = load_exp2_measured_records(args.input_file)
    else:
        records = load_exp3_measured_records(args.input_file)

    warnings: list[str] = []
    errors: list[str] = []

    print(f"Validated measured Exp {experiment} input: {args.input_file}")
    for line in _dataset_summary_lines(experiment, records):
        print(line)

    if experiment == "3":
        failure_modes = sorted({str(record["failure_mode"]) for record in records})
        if len(failure_modes) != 1:
            errors.append(
                "Exp 3 analyzer requires exactly one failure_mode per file; "
                f"found {len(failure_modes)} ({_format_values(failure_modes)})."
            )

    scenario_present, scenario_total = _scenario_id_coverage(records)
    if scenario_present < scenario_total:
        warnings.append(
            "scenario_id is incomplete; pairwise comparisons will fall back to run-based keys. "
            "This is valid, but cross-policy pairing is less explicit."
        )

    print("\nPairwise preflight:")
    for check in _pairwise_checks(experiment, records):
        analysis = analyze_pairwise_compatibility(
            records,
            group_keys=check["group_keys"],
            scenario_keys=check["scenario_keys"],
        )
        duplicate_count = len(analysis["duplicate_scenarios"])
        missing_group_count = len(analysis["missing_policy_groups"])
        no_overlap_count = len(analysis["no_overlap_pairs"])

        print(
            f"- {check['label']}: scenario_keys={_format_values(check['scenario_keys'])}; "
            f"groups={analysis['group_count']}; shared_pairs_min={analysis['min_shared_pairs']}; "
            f"shared_pairs_max={analysis['max_shared_pairs']}; duplicates={duplicate_count}; "
            f"missing_policy_groups={missing_group_count}; no_overlap_pairs={no_overlap_count}"
        )

        if duplicate_count:
            warnings.append(
                f"{check['label']} has {duplicate_count} duplicate policy/scenario rows. "
                "The pairwise comparator keeps only one row per policy/scenario, so duplicates "
                "should be deduplicated before analysis."
            )
            example_text = _format_example_list(
                analysis["duplicate_scenarios"],
                formatter=lambda item: (
                    f"{_format_group(item, check['group_keys'])}, policy={item['policy']}, "
                    f"scenario={item['scenario']}, count={item['count']}"
                ),
            )
            warnings.append(f"{check['label']} duplicate examples: {example_text}")

        if missing_group_count:
            warnings.append(
                f"{check['label']} is missing at least one policy in {missing_group_count} group(s), "
                "so some pairwise comparisons will be omitted."
            )
            example_text = _format_example_list(
                analysis["missing_policy_groups"],
                formatter=lambda item: (
                    f"{_format_group(item, check['group_keys'])}, "
                    f"missing={_format_values(item['missing_policies'])}"
                ),
            )
            warnings.append(f"{check['label']} missing-policy examples: {example_text}")

        if no_overlap_count:
            warnings.append(
                f"{check['label']} has {no_overlap_count} policy-pair group(s) with zero shared "
                "scenarios, so paired bootstrap rows will be missing there."
            )
            example_text = _format_example_list(
                analysis["no_overlap_pairs"],
                formatter=lambda item: (
                    f"{_format_group(item, check['group_keys'])}, "
                    f"{item['policy_a']} vs {item['policy_b']}"
                ),
            )
            warnings.append(f"{check['label']} no-overlap examples: {example_text}")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"- {warning}")

    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"- {error}")
        print("\nValidation result: FAIL")
        return 1

    if warnings and args.strict:
        print("\nValidation result: WARNINGS (strict mode -> non-zero exit)")
        return 1

    if warnings:
        print("\nValidation result: WARNINGS")
        return 0

    print("\nValidation result: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
