#!/usr/bin/env python3
"""Standalone drift-check between PAPER.md's headline numbers and the
committed measured CSVs under data/measured/.

This script intentionally does NOT import the orchestratebench package or
its statistics/analysis modules. It re-derives a handful of headline figures
directly from the raw CSVs with only the Python standard library, so it acts
as an independent cross-check against scripts/analyze_measured.py (which
uses the package's own analyzers). If both paths agree, that's a much
stronger reproducibility signal than either one alone.

It does not run any experiment and does not write or modify any file under
data/measured/ -- it only reads the committed CSVs and prints a report.

Usage:
    python3 scripts/audit_paper_claims.py
    python3 scripts/audit_paper_claims.py --strict   # exit 1 on any drift

What it checks today (extend CLAIMS as the paper grows):
  - Experiment 4 (data/measured/exp4_real.csv): mean delegation_fidelity for
    the `monolithic` and `decompose` policies, against the values reported
    in PAPER.md "5.4 Experiment 4 -- Decomposition quality":
      monolithic ~ 0.37 (paper states 95% CI [0.33, 0.43])
      decompose  ~ 1.00
  - Experiment 3 (data/measured/exp3_real.csv): mean cascade_radius by depth
    for the pooled latent/semantic failure modes (everything except
    tool_invocation_error), against PAPER.md section 5.3:
      depth 3 ~ 1.0
      depth 5 ~ 2.9 (paper reports [2.6, 3.0] 95% CI bucket -> ~2.88)
      depth 7 ~ 5.0
    and tool_invocation_error cascade_radius == 0 at every depth.

This is a lightweight guard, not a replacement for
scripts/analyze_measured.py or the bootstrap-CI statistics it reports --
it exists to catch *gross* drift (e.g. someone edits a CSV and forgets to
update the paper, or vice versa) cheaply and without any dependencies.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "measured"

# Tolerance for "this still matches the paper" -- generous on purpose since
# this is a coarse sanity check, not a statistical test.
TOLERANCE = 0.03

# (description, expected_value) pairs. Source: PAPER.md sections 5.3 and 5.4,
# and DESIGN_DOC.md section 3 (Experiment 3 / Experiment 4), as of the v0.6
# draft committed to this repo. Update this table if those sections change.
CLAIMS = {
    "exp4.monolithic.mean_delegation_fidelity": 0.37,
    "exp4.decompose.mean_delegation_fidelity": 1.00,
    "exp3.depth3.latent_mean_cascade_radius": 1.0,
    "exp3.depth5.latent_mean_cascade_radius": 2.88,
    "exp3.depth7.latent_mean_cascade_radius": 5.0,
    "exp3.tool_invocation_error.mean_cascade_radius": 0.0,
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _exp4_claims() -> dict[str, float]:
    path = DATA_DIR / "exp4_real.csv"
    if not path.exists():
        return {}
    rows = _read_csv(path)
    out: dict[str, float] = {}
    for policy in ("monolithic", "decompose"):
        values = [
            float(row["delegation_fidelity"])
            for row in rows
            if row.get("policy") == policy
        ]
        if values:
            out[f"exp4.{policy}.mean_delegation_fidelity"] = mean(values)
    return out


def _exp3_claims() -> dict[str, float]:
    path = DATA_DIR / "exp3_real.csv"
    if not path.exists():
        return {}
    rows = _read_csv(path)
    out: dict[str, float] = {}

    # Cascade radius by depth, pooled over the latent/semantic failure modes
    # (everything that is not tool_invocation_error), matching PAPER.md 5.3.
    for depth in (3, 5, 7):
        values = [
            float(row["cascade_radius"])
            for row in rows
            if row.get("failure_mode") != "tool_invocation_error"
            and row.get("depth") == str(depth)
        ]
        if values:
            out[f"exp3.depth{depth}.latent_mean_cascade_radius"] = mean(values)

    tool_values = [
        float(row["cascade_radius"])
        for row in rows
        if row.get("failure_mode") == "tool_invocation_error"
    ]
    if tool_values:
        out["exp3.tool_invocation_error.mean_cascade_radius"] = mean(tool_values)

    return out


def collect_observed() -> dict[str, float]:
    observed: dict[str, float] = {}
    observed.update(_exp4_claims())
    observed.update(_exp3_claims())
    return observed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with status 1 if any claim drifts beyond tolerance.",
    )
    args = parser.parse_args(argv)

    if not DATA_DIR.exists():
        print(f"data/measured/ not found at {DATA_DIR}; nothing to audit.")
        return 0

    observed = collect_observed()
    drifted = []
    missing = []

    print("PAPER.md claim audit (independent of orchestratebench package)\n")
    print(f"{'claim':<48} {'paper':>8} {'observed':>10} {'status':>10}")
    print("-" * 80)

    for key, expected in CLAIMS.items():
        if key not in observed:
            missing.append(key)
            print(f"{key:<48} {expected:>8.3f} {'(no data)':>10} {'SKIPPED':>10}")
            continue
        actual = observed[key]
        delta = abs(actual - expected)
        status = "OK" if delta <= TOLERANCE else "DRIFT"
        if status == "DRIFT":
            drifted.append((key, expected, actual, delta))
        print(f"{key:<48} {expected:>8.3f} {actual:>10.3f} {status:>10}")

    print()
    if drifted:
        print(f"{len(drifted)} claim(s) drifted beyond tolerance ({TOLERANCE}):")
        for key, expected, actual, delta in drifted:
            print(f"  - {key}: paper={expected:.3f} observed={actual:.3f} (delta={delta:.3f})")
            print("    -> update PAPER.md/DESIGN_DOC.md, or this script's CLAIMS table,")
            print("       whichever is stale.")
    else:
        print("No drift detected for the claims this script checks.")

    if missing:
        print(
            f"\n{len(missing)} claim(s) could not be checked (missing CSV or columns): "
            + ", ".join(missing)
        )

    if args.strict and drifted:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
