"""CLI for Experiment 3 analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from orchestratebench.automation import build_named_policies
from orchestratebench.cli._common import parse_int_list
from orchestratebench.console_reports import (
    format_cascade_report,
    format_cascade_sweep_report,
    format_pairwise_report,
)
from orchestratebench.experiment_analysis import (
    CASCADE_PAIRWISE_METRICS,
    compare_policy_pairs,
    summarize_cascade_records,
)
from orchestratebench.experiment_artifacts import write_json_file, write_records_csv
from orchestratebench.experiments import (
    collect_cascade_records,
)
from orchestratebench.failures import FailureMode
from orchestratebench.measured_runs import load_exp3_measured_records, prefer_scenario_id
from orchestratebench.publication import (
    build_exp3_latex_tables,
    build_exp3_markdown_report,
    write_publication_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-runs", type=int, default=20, help="Runs per depth x injection stage.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed for deterministic runs.")
    parser.add_argument(
        "--depths",
        default="3,5,7",
        help="Comma-separated pipeline depths to evaluate.",
    )
    parser.add_argument(
        "--injection-stages",
        default="0,1,2",
        help="Comma-separated zero-based injection stages to sweep.",
    )
    parser.add_argument(
        "--mode",
        default=FailureMode.CONTEXT_POLLUTION.value,
        choices=[mode.value for mode in FailureMode],
        help="Injected failure mode for the cascade sweep.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/exp3"),
        help="Directory for raw and summarized artifacts.",
    )
    parser.add_argument(
        "--with-ci",
        action="store_true",
        help="Include bootstrap confidence intervals in summary artifacts.",
    )
    parser.add_argument(
        "--n-resamples",
        type=int,
        default=2000,
        help="Bootstrap resamples when --with-ci is enabled.",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        help="Optional measured Exp 3 CSV / JSONL / JSON file. When provided, skips simulation.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _summary_rows_to_depth_nested(
    rows: list[dict[str, object]],
) -> dict[str, dict[int, dict[str, float]]]:
    nested: dict[str, dict[int, dict[str, float]]] = {}
    for row in rows:
        policy = str(row["policy"])
        depth = int(row["depth"])
        nested.setdefault(policy, {})
        nested[policy][depth] = {
            key: float(value)
            for key, value in row.items()
            if key not in {"policy", "depth", "n_runs"}
        }
    return nested


def _summary_rows_to_stage_nested(
    rows: list[dict[str, object]],
) -> dict[str, dict[int, dict[int, dict[str, float]]]]:
    nested: dict[str, dict[int, dict[int, dict[str, float]]]] = {}
    for row in rows:
        policy = str(row["policy"])
        depth = int(row["depth"])
        stage = int(row["injection_stage"])
        nested.setdefault(policy, {})
        nested[policy].setdefault(depth, {})
        nested[policy][depth][stage] = {
            key: float(value)
            for key, value in row.items()
            if key not in {"policy", "depth", "injection_stage", "n_runs"}
        }
    return nested


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.input_file is not None:
        records = load_exp3_measured_records(args.input_file)
        policies = sorted({str(row["policy"]) for row in records})
        failure_modes = sorted({str(row["failure_mode"]) for row in records})
        if len(failure_modes) > 1:
            raise ValueError(
                "Measured Exp 3 input must contain exactly one failure_mode per file. "
                f"Got: {failure_modes}"
            )
        mode = FailureMode(failure_modes[0]) if failure_modes else FailureMode(args.mode)
        depths = sorted({int(row["depth"]) for row in records})
        injection_stages = sorted({int(row["injection_stage"]) for row in records})
        source_mode = "measured"
    else:
        policy_map = build_named_policies()
        policies = list(policy_map)
        mode = FailureMode(args.mode)
        depths = parse_int_list(args.depths)
        injection_stages = parse_int_list(args.injection_stages)
        records = collect_cascade_records(
            policy_map,
            depths=depths,
            injection_stages=injection_stages,
            mode=mode,
            n_runs=args.n_runs,
            seed=args.seed,
        )
        source_mode = "simulated"

    stage_summary_rows = summarize_cascade_records(
        records,
        with_ci=args.with_ci,
        n_resamples=args.n_resamples,
        seed=args.seed,
    )
    depth_summary_rows = summarize_cascade_records(
        records,
        group_keys=("policy", "depth"),
        with_ci=args.with_ci,
        n_resamples=args.n_resamples,
        seed=args.seed,
    )
    pairwise_by_depth_stage_rows = compare_policy_pairs(
        records,
        CASCADE_PAIRWISE_METRICS,
        group_keys=("depth", "injection_stage"),
        scenario_keys=prefer_scenario_id(records, fallback=("run",)),
        n_resamples=args.n_resamples,
        seed=args.seed,
    )
    pairwise_by_depth_rows = compare_policy_pairs(
        records,
        CASCADE_PAIRWISE_METRICS,
        group_keys=("depth",),
        scenario_keys=prefer_scenario_id(records, fallback=("injection_stage", "run")),
        n_resamples=args.n_resamples,
        seed=args.seed,
    )

    sweep_results = _summary_rows_to_stage_nested(stage_summary_rows)
    depth_results = _summary_rows_to_depth_nested(depth_summary_rows)
    sweep_text = format_cascade_sweep_report(sweep_results)
    depth_text = format_cascade_report(depth_results)
    pairwise_text = format_pairwise_report(
        pairwise_by_depth_stage_rows,
        title="Experiment 3 - pairwise policy comparisons by depth and injection stage",
        group_keys=("depth", "injection_stage"),
    )
    print(sweep_text)
    print()
    print(depth_text)
    print()
    print(pairwise_text)

    write_records_csv(records, args.output_dir / "raw_runs.csv")
    write_records_csv(stage_summary_rows, args.output_dir / "summary_by_depth_stage.csv")
    write_records_csv(depth_summary_rows, args.output_dir / "summary_by_depth.csv")
    write_records_csv(
        pairwise_by_depth_stage_rows,
        args.output_dir / "pairwise_by_depth_stage.csv",
    )
    write_records_csv(
        pairwise_by_depth_rows,
        args.output_dir / "pairwise_by_depth.csv",
    )
    config = {
        "source_mode": source_mode,
        "n_runs": args.n_runs,
        "seed": args.seed,
        "depths": depths,
        "injection_stages": injection_stages,
        "mode": mode.value,
        "policies": list(policies),
        "with_ci": args.with_ci,
        "n_resamples": args.n_resamples,
        "input_file": str(args.input_file) if args.input_file is not None else None,
    }
    write_json_file(
        {
            "config": config,
            "summary_by_depth_stage": stage_summary_rows,
            "summary_by_depth": depth_summary_rows,
            "pairwise_by_depth_stage": pairwise_by_depth_stage_rows,
            "pairwise_by_depth": pairwise_by_depth_rows,
        },
        args.output_dir / "summary.json",
    )
    (args.output_dir / "report.txt").write_text(
        sweep_text + "\n\n" + depth_text + "\n\n" + pairwise_text + "\n",
        encoding="utf-8",
    )
    write_publication_artifacts(
        args.output_dir,
        markdown_text=build_exp3_markdown_report(
            stage_summary_rows,
            depth_summary_rows,
            pairwise_by_depth_stage_rows,
            pairwise_by_depth_rows,
            config=config,
        ),
        latex_text=build_exp3_latex_tables(
            stage_summary_rows,
            depth_summary_rows,
        ),
    )
    print(f"\n[artifact] wrote Exp 3 artifacts to {args.output_dir}")
    if source_mode == "simulated":
        print(
            "\n[note] simulated-trace harness demo (#7). Retry-wrapped policies in this "
            "offline runner only help on retryable injected failures; latent failures "
            "can still cascade. Measured paper numbers come from the collaborative "
            "gold-labeled run; do not cite these as final results."
        )
    else:
        print(
            "\n[note] analyzed measured Exp 3 input. Review the record schema carefully "
            "before citing results, especially scenario pairing and injection-stage labels."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
