"""Run Exp 2/3 end-to-end from one command."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from orchestratebench.automation import (
    build_auto_exp2_measured_records,
    build_auto_exp3_measured_records,
    write_auto_measured_records,
)
from orchestratebench.cli._common import parse_int_list, parse_list, run_step
from orchestratebench.cli.run_exp2 import main as run_exp2_main
from orchestratebench.cli.run_exp3 import main as run_exp3_main
from orchestratebench.cli.validate_measured_input import main as validate_measured_input_main
from orchestratebench.experiment_artifacts import write_json_file
from orchestratebench.failures import FailureMode
from orchestratebench.measured_templates import DEFAULT_MEASURED_POLICIES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/exp23_pipeline"),
        help="Root directory for measured inputs, analysis artifacts, and manifest.",
    )
    parser.add_argument(
        "--policies",
        default=",".join(DEFAULT_MEASURED_POLICIES),
        help="Comma-separated supported policy names.",
    )
    parser.add_argument(
        "--exp2-runs",
        type=int,
        default=5,
        help="Runs per Exp 2 workflow x failure mode.",
    )
    parser.add_argument(
        "--exp3-runs",
        type=int,
        default=5,
        help="Runs per Exp 3 depth x injection stage.",
    )
    parser.add_argument(
        "--depths",
        default="3,5,7",
        help="Comma-separated Exp 3 depths.",
    )
    parser.add_argument(
        "--injection-stages",
        default="0,1,2",
        help="Comma-separated zero-based injection stages for Exp 3.",
    )
    parser.add_argument(
        "--exp3-modes",
        default=FailureMode.CONTEXT_POLLUTION.value,
        help="Comma-separated Exp 3 failure modes. One analyzed file is produced per mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed for deterministic scenario generation and harness collection.",
    )
    parser.add_argument(
        "--with-ci",
        action="store_true",
        help="Enable bootstrap confidence intervals in Exp 2/3 analyzer outputs.",
    )
    parser.add_argument(
        "--n-resamples",
        type=int,
        default=2000,
        help="Bootstrap resamples when --with-ci is enabled.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = args.output_root
    measured_dir = output_root / "measured_inputs"
    analysis_root = output_root / "analysis"

    policies = parse_list(args.policies)
    depths = parse_int_list(args.depths)
    injection_stages = parse_int_list(args.injection_stages)
    exp3_modes = [FailureMode(mode) for mode in parse_list(args.exp3_modes)]

    measured_dir.mkdir(parents=True, exist_ok=True)
    analysis_root.mkdir(parents=True, exist_ok=True)

    exp2_rows = build_auto_exp2_measured_records(
        policy_names=policies,
        n_runs=args.exp2_runs,
        seed=args.seed,
    )
    exp2_input = measured_dir / "exp2_measured_auto.csv"
    write_auto_measured_records(exp2_rows, exp2_input)
    print(f"[artifact] wrote {exp2_input} ({len(exp2_rows)} rows)")

    run_step(
        "orbench-validate-measured",
        validate_measured_input_main,
        [
            "--experiment",
            "2",
            "--input-file",
            str(exp2_input),
            "--strict",
        ],
    )
    exp2_output_dir = analysis_root / "exp2"
    exp2_argv = [
        "--input-file",
        str(exp2_input),
        "--output-dir",
        str(exp2_output_dir),
        "--n-resamples",
        str(args.n_resamples),
    ]
    if args.with_ci:
        exp2_argv.append("--with-ci")
    run_step("orbench-run-exp2", run_exp2_main, exp2_argv)

    exp3_outputs: list[dict[str, object]] = []
    for mode in exp3_modes:
        exp3_rows = build_auto_exp3_measured_records(
            policy_names=policies,
            depths=depths,
            injection_stages=injection_stages,
            mode=mode,
            n_runs=args.exp3_runs,
            seed=args.seed,
        )
        exp3_input = measured_dir / f"exp3_{mode.value}_auto.csv"
        write_auto_measured_records(exp3_rows, exp3_input)
        print(f"[artifact] wrote {exp3_input} ({len(exp3_rows)} rows)")

        run_step(
            "orbench-validate-measured",
            validate_measured_input_main,
            [
                "--experiment",
                "3",
                "--input-file",
                str(exp3_input),
                "--strict",
            ],
        )
        exp3_output_dir = analysis_root / "exp3" / mode.value
        exp3_argv = [
            "--input-file",
            str(exp3_input),
            "--output-dir",
            str(exp3_output_dir),
            "--n-resamples",
            str(args.n_resamples),
        ]
        if args.with_ci:
            exp3_argv.append("--with-ci")
        run_step("orbench-run-exp3", run_exp3_main, exp3_argv)
        exp3_outputs.append(
            {
                "mode": mode.value,
                "input_file": str(exp3_input),
                "output_dir": str(exp3_output_dir),
                "n_rows": len(exp3_rows),
            }
        )

    manifest = {
        "pipeline": "run_exp23_pipeline",
        "config": {
            "policies": policies,
            "exp2_runs": args.exp2_runs,
            "exp3_runs": args.exp3_runs,
            "depths": depths,
            "injection_stages": injection_stages,
            "exp3_modes": [mode.value for mode in exp3_modes],
            "seed": args.seed,
            "with_ci": args.with_ci,
            "n_resamples": args.n_resamples,
        },
        "artifacts": {
            "measured_dir": str(measured_dir),
            "analysis_root": str(analysis_root),
            "exp2": {
                "input_file": str(exp2_input),
                "output_dir": str(exp2_output_dir),
                "n_rows": len(exp2_rows),
            },
            "exp3": exp3_outputs,
        },
    }
    write_json_file(manifest, output_root / "pipeline_manifest.json")
    print(f"[artifact] wrote {output_root / 'pipeline_manifest.json'}")
    print("\nPipeline complete.")
    print(f"- Measured inputs: {measured_dir}")
    print(f"- Analysis artifacts: {analysis_root}")
    print("- Manual filling: not required on this auto-harness path")
    print(
        "- For real collaborative measured runs, use "
        "scripts/scaffold_measured_inputs.py and fill the blank metrics yourself."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
