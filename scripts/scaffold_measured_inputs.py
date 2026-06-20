#!/usr/bin/env python3
"""Create blank measured long-form files for Exp 2/3.

Examples:

    python scripts/scaffold_measured_inputs.py --output-dir data/measured
    python scripts/scaffold_measured_inputs.py --output-dir data/measured --exp2-runs 20 --exp3-runs 20
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orchestratebench import (  # noqa: E402
    DEFAULT_MEASURED_POLICIES,
    EXP2_SKELETON_FIELDS,
    EXP3_SKELETON_FIELDS,
    FailureMode,
    build_exp2_measured_skeleton,
    build_exp3_measured_skeleton,
)


def _parse_list(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/measured"),
        help="Directory for generated measured-input skeleton files.",
    )
    parser.add_argument(
        "--policies",
        default=",".join(DEFAULT_MEASURED_POLICIES),
        help="Comma-separated policy names to include in each skeleton row.",
    )
    parser.add_argument(
        "--exp2-runs",
        type=int,
        default=5,
        help="Scenario runs per Exp 2 workflow x failure mode.",
    )
    parser.add_argument(
        "--exp3-runs",
        type=int,
        default=5,
        help="Scenario runs per Exp 3 depth x injection stage.",
    )
    parser.add_argument(
        "--exp3-depths",
        default="3,5,7",
        help="Comma-separated Exp 3 depths.",
    )
    parser.add_argument(
        "--exp3-injection-stages",
        default="0,1,2",
        help="Comma-separated zero-based Exp 3 injection stages.",
    )
    parser.add_argument(
        "--exp3-modes",
        default=FailureMode.CONTEXT_POLLUTION.value,
        help="Comma-separated Exp 3 failure modes. One CSV scaffold is written per mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed used to deterministically scaffold scenario ids and stage assignments.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    policies = _parse_list(args.policies)
    exp3_depths = _parse_int_list(args.exp3_depths)
    exp3_stages = _parse_int_list(args.exp3_injection_stages)
    exp3_modes = [FailureMode(mode) for mode in _parse_list(args.exp3_modes)]

    exp2_rows = build_exp2_measured_skeleton(
        policies=policies,
        n_runs=args.exp2_runs,
        seed=args.seed,
    )
    exp2_path = output_dir / "exp2_measured_skeleton.csv"
    _write_csv(exp2_path, EXP2_SKELETON_FIELDS, exp2_rows)
    print(f"[artifact] wrote {exp2_path} ({len(exp2_rows)} rows)")

    for mode in exp3_modes:
        exp3_rows = build_exp3_measured_skeleton(
            policies=policies,
            depths=exp3_depths,
            injection_stages=exp3_stages,
            mode=mode,
            n_runs=args.exp3_runs,
            seed=args.seed,
        )
        exp3_path = output_dir / f"exp3_{mode.value}_skeleton.csv"
        _write_csv(exp3_path, EXP3_SKELETON_FIELDS, exp3_rows)
        print(f"[artifact] wrote {exp3_path} ({len(exp3_rows)} rows)")

    print("\nNext steps:")
    print("1. Copy each *_skeleton.csv file to a measured working file name.")
    print("2. Fill in the blank metric columns from your real system runs.")
    print("3. Validate with scripts/validate_measured_input.py.")
    print("4. Analyze with scripts/run_exp2.py or scripts/run_exp3.py.")


if __name__ == "__main__":
    main()
