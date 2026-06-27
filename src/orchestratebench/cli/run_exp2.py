"""CLI for Experiment 2 analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from orchestratebench.automation import build_named_policies
from orchestratebench.experiments import (
    FAILURE_RECOVERY_PAIRWISE_METRICS,
    compare_policy_pairs,
    collect_failure_recovery_records,
    format_pairwise_report,
    format_recovery_report,
    summarize_failure_recovery_records,
    write_json_file,
    write_records_csv,
)
from orchestratebench.measured_runs import load_exp2_measured_records, prefer_scenario_id
from orchestratebench.paper_reports import (
    build_exp2_latex_tables,
    build_exp2_markdown_report,
    write_publication_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-runs", type=int, default=20, help="Runs per workflow x failure mode.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed for deterministic runs.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/exp2"),
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
        help="Optional measured Exp 2 CSV / JSONL / JSON file. When provided, skips simulation.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _summary_rows_to_nested(
    rows: list[dict[str, object]],
) -> dict[str, dict[str, dict[str, float]]]:
    nested: dict[str, dict[str, dict[str, float]]] = {}
    for row in rows:
        policy = str(row["policy"])
        mode = str(row["failure_mode"])
        nested.setdefault(policy, {})
        nested[policy][mode] = {
            key: float(value)
            for key, value in row.items()
            if key not in {"policy", "failure_mode", "n_runs"}
        }
    return nested


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.input_file is not None:
        records = load_exp2_measured_records(args.input_file)
        policies = sorted({str(row["policy"]) for row in records})
        source_mode = "measured"
    else:
        policy_map = build_named_policies()
        policies = list(policy_map)
        records = collect_failure_recovery_records(
            policy_map,
            n_runs=args.n_runs,
            seed=args.seed,
        )
        source_mode = "simulated"

    summary_rows = summarize_failure_recovery_records(
        records,
        with_ci=args.with_ci,
        n_resamples=args.n_resamples,
        seed=args.seed,
    )
    workflow_summary_rows = summarize_failure_recovery_records(
        records,
        group_keys=("policy", "workflow", "failure_mode"),
        with_ci=False,
        seed=args.seed,
    )
    pairwise_by_mode_rows = compare_policy_pairs(
        records,
        FAILURE_RECOVERY_PAIRWISE_METRICS,
        group_keys=("failure_mode",),
        scenario_keys=prefer_scenario_id(
            records,
            fallback=("workflow", "run", "injection_stage"),
        ),
        n_resamples=args.n_resamples,
        seed=args.seed,
    )
    pairwise_by_workflow_mode_rows = compare_policy_pairs(
        records,
        FAILURE_RECOVERY_PAIRWISE_METRICS,
        group_keys=("workflow", "failure_mode"),
        scenario_keys=prefer_scenario_id(
            records,
            fallback=("run", "injection_stage"),
        ),
        n_resamples=args.n_resamples,
        seed=args.seed,
    )

    results = _summary_rows_to_nested(summary_rows)
    summary_text = format_recovery_report(results)
    pairwise_text = format_pairwise_report(
        pairwise_by_mode_rows,
        title="Experiment 2 - pairwise policy comparisons by failure mode",
        group_keys=("failure_mode",),
    )
    print(summary_text)
    print()
    print(pairwise_text)

    write_records_csv(records, args.output_dir / "raw_runs.csv")
    write_records_csv(summary_rows, args.output_dir / "summary_by_mode.csv")
    write_records_csv(
        workflow_summary_rows,
        args.output_dir / "summary_by_workflow_mode.csv",
    )
    write_records_csv(
        pairwise_by_mode_rows,
        args.output_dir / "pairwise_by_mode.csv",
    )
    write_records_csv(
        pairwise_by_workflow_mode_rows,
        args.output_dir / "pairwise_by_workflow_mode.csv",
    )
    config = {
        "source_mode": source_mode,
        "n_runs": args.n_runs,
        "seed": args.seed,
        "policies": list(policies),
        "with_ci": args.with_ci,
        "n_resamples": args.n_resamples,
        "input_file": str(args.input_file) if args.input_file is not None else None,
    }
    write_json_file(
        {
            "config": config,
            "summary_by_mode": summary_rows,
            "summary_by_workflow_mode": workflow_summary_rows,
            "pairwise_by_mode": pairwise_by_mode_rows,
            "pairwise_by_workflow_mode": pairwise_by_workflow_mode_rows,
        },
        args.output_dir / "summary.json",
    )
    (args.output_dir / "report.txt").write_text(
        summary_text + "\n\n" + pairwise_text + "\n",
        encoding="utf-8",
    )
    write_publication_artifacts(
        args.output_dir,
        markdown_text=build_exp2_markdown_report(
            summary_rows,
            pairwise_by_mode_rows,
            config=config,
        ),
        latex_text=build_exp2_latex_tables(summary_rows),
    )
    print(f"\n[artifact] wrote Exp 2 artifacts to {args.output_dir}")
    if source_mode == "simulated":
        print(
            "\n[note] simulated-trace harness demo (#4). Retry-wrapped policies in this "
            "offline runner only help on retryable injected failures (for example "
            "tool_invocation_error). Measured paper numbers still come from the "
            "collaborative gold-labeled run; do not cite these as final results."
        )
    else:
        print(
            "\n[note] analyzed measured Exp 2 input. Review the record schema carefully "
            "before citing results, especially scenario pairing and failure-mode labels."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
