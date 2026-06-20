#!/usr/bin/env python3
"""Run Exp 2/3 end-to-end from one command.

This automation pipeline:
1. auto-collects measured-style long-form CSVs from the current harness
2. validates those CSVs
3. runs Exp 2/3 analyzers
4. emits summary, pairwise, and paper-facing artifacts

Examples:

    python scripts/run_exp23_pipeline.py
    python scripts/run_exp23_pipeline.py --exp2-runs 20 --exp3-runs 20 --exp3-modes context_pollution,tool_invocation_error --with-ci
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orchestratebench import (  # noqa: E402
    DEFAULT_MEASURED_POLICIES,
    FailureMode,
    build_auto_exp2_measured_records,
    build_auto_exp3_measured_records,
    write_auto_measured_records,
    write_json_file,
)


def _parse_list(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


def _run_subprocess(command: list[str], cwd: Path) -> None:
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_root = args.output_root
    measured_dir = output_root / "measured_inputs"
    analysis_root = output_root / "analysis"
    python_exe = sys.executable

    policies = _parse_list(args.policies)
    depths = _parse_int_list(args.depths)
    injection_stages = _parse_int_list(args.injection_stages)
    exp3_modes = [FailureMode(mode) for mode in _parse_list(args.exp3_modes)]

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

    _run_subprocess(
        [
            python_exe,
            "scripts/validate_measured_input.py",
            "--experiment",
            "2",
            "--input-file",
            str(exp2_input),
            "--strict",
        ],
        cwd=repo_root,
    )
    exp2_output_dir = analysis_root / "exp2"
    exp2_command = [
        python_exe,
        "scripts/run_exp2.py",
        "--input-file",
        str(exp2_input),
        "--output-dir",
        str(exp2_output_dir),
        "--n-resamples",
        str(args.n_resamples),
    ]
    if args.with_ci:
        exp2_command.append("--with-ci")
    _run_subprocess(exp2_command, cwd=repo_root)

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

        _run_subprocess(
            [
                python_exe,
                "scripts/validate_measured_input.py",
                "--experiment",
                "3",
                "--input-file",
                str(exp3_input),
                "--strict",
            ],
            cwd=repo_root,
        )
        exp3_output_dir = analysis_root / "exp3" / mode.value
        exp3_command = [
            python_exe,
            "scripts/run_exp3.py",
            "--input-file",
            str(exp3_input),
            "--output-dir",
            str(exp3_output_dir),
            "--n-resamples",
            str(args.n_resamples),
        ]
        if args.with_ci:
            exp3_command.append("--with-ci")
        _run_subprocess(exp3_command, cwd=repo_root)
        exp3_outputs.append(
            {
                "mode": mode.value,
                "input_file": str(exp3_input),
                "output_dir": str(exp3_output_dir),
                "n_rows": len(exp3_rows),
            }
        )

    manifest = {
        "pipeline": "run_exp23_pipeline.py",
        "python_executable": python_exe,
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


if __name__ == "__main__":
    main()
