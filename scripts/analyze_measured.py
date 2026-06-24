#!/usr/bin/env python3
"""Recompute every measured number reported in PAPER.md (Experiments 2/3/4)
directly from the committed CSVs, with bootstrap 95% CIs and the paired
significance tests the draft currently omits (Exp 4 decompose vs monolithic).

Pure offline: reads ``data/measured/*.csv`` only -- no API key, no network. This
is the Exp 2/3/4 counterpart to ``scripts/reproduce_exp1.py``, so the §9 claim
"every reported number regenerable" holds for the measured experiments too.

Usage:
    python scripts/analyze_measured.py
    python scripts/analyze_measured.py --data-dir data/measured --seed 0
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orchestratebench.statistics import (  # noqa: E402
    bootstrap_ci,
    paired_bootstrap_values,
)

LATENT_MODES = (
    "ambiguous_delegation",
    "context_pollution",
    "conflicting_outputs",
    "premature_action",
)
TOOL_MODE = "tool_invocation_error"


def _load(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _bool(cell: object) -> float:
    return 1.0 if str(cell).strip().lower() in ("true", "1") else 0.0


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _stat(values: list[float], seed: int, decimals: int = 3) -> str:
    lo, hi = bootstrap_ci(values, seed=seed)
    return f"{_mean(values):.{decimals}f} [{lo:.{decimals}f}, {hi:.{decimals}f}]"


def analyze_exp2(rows: list[dict], title: str, seed: int) -> None:
    print(f"\n### {title}  (N={len(rows)})")
    fts, cr = defaultdict(list), defaultdict(list)
    for r in rows:
        fts[r["failure_mode"]].append(_bool(r["final_task_success"]))
        cr[r["failure_mode"]].append(float(r["cascade_radius"]))
    print(f"    {'mode':24s}  {'final-task success [95% CI]':28s}  mean cascade radius [95% CI]")
    for m in (TOOL_MODE, *LATENT_MODES):
        print(f"    {m:24s}  {_stat(fts[m], seed):28s}  {_stat(cr[m], seed)}   (n={len(fts[m])})")
    latent = [v for m in LATENT_MODES for v in fts[m]]
    print(f"    {'LATENT (pooled)':24s}  {_stat(latent, seed):28s}   (n={len(latent)})")


def analyze_exp3(rows: list[dict], seed: int) -> None:
    print(f"\n### Exp 3 -- cascade radius by depth  (N={len(rows)})")
    depths = sorted({int(r["depth"]) for r in rows})
    print(f"    {'depth':6s}  {'latent cascade radius [95% CI]':32s}  tool cascade radius [95% CI]")
    for d in depths:
        latent = [float(r["cascade_radius"]) for r in rows
                  if int(r["depth"]) == d and r["failure_mode"] in LATENT_MODES]
        tool = [float(r["cascade_radius"]) for r in rows
                if int(r["depth"]) == d and r["failure_mode"] == TOOL_MODE]
        print(f"    {d:<6d}  {_stat(latent, seed):32s}  {_stat(tool, seed)}   (n_latent={len(latent)})")
    print("    per-mode cascade radius by depth (surfaces the ambiguous@depth-5 noise):")
    for m in LATENT_MODES:
        cells = []
        for d in depths:
            vals = [float(r["cascade_radius"]) for r in rows
                    if int(r["depth"]) == d and r["failure_mode"] == m]
            cells.append(f"d{d}={_mean(vals):.3f}")
        print(f"      {m:24s} " + "  ".join(cells))


def analyze_exp4(rows: list[dict], seed: int) -> None:
    print(f"\n### Exp 4 -- decomposition quality  (N={len(rows)})")
    by: dict[str, dict[str, dict]] = {"decompose": {}, "monolithic": {}}
    for r in rows:
        by[r["policy"]][r["task_id"]] = r
    tasks = sorted(by["decompose"], key=int)
    for metric in ("delegation_fidelity", "granularity_error"):
        for p in ("decompose", "monolithic"):
            vals = [float(by[p][t][metric]) for t in tasks]
            print(f"    {p:10s}  {metric:20s}  {_stat(vals, seed)}")
        a = [float(by["decompose"][t][metric]) for t in tasks]
        b = [float(by["monolithic"][t][metric]) for t in tasks]
        res = paired_bootstrap_values(a, b, seed=seed)
        lo, hi = res["ci"]  # type: ignore[misc]
        print(f"      paired decompose-monolithic: diff={res['diff']:.3f} "
              f"95% CI [{lo:.3f}, {hi:.3f}]  p={res['p_value']:.4f}  (n_pairs={len(a)})")
    for p in ("decompose", "monolithic"):
        fc = [_bool(by[p][t]["final_correct"]) for t in tasks]
        print(f"    {p:10s}  {'final_correct':20s}  {_stat(fc, seed)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path,
                    default=Path(__file__).resolve().parents[1] / "data" / "measured")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    d = args.data_dir

    print("=" * 78)
    print("OrchestraBench -- measured-experiment statistics (regenerated from CSV)")
    print(f"data-dir={d}")
    print(f"seed={args.seed}  bootstrap=10000 resamples, 95% percentile CI")
    print("=" * 78)

    analyze_exp2(_load(d / "exp2_real.csv"), "Exp 2 -- arithmetic chain", args.seed)
    analyze_exp2(_load(d / "exp2_domain_real.csv"), "Exp 2 -- loan-approval domain", args.seed)
    analyze_exp3(_load(d / "exp3_real.csv"), args.seed)
    analyze_exp4(_load(d / "exp4_real.csv"), args.seed)
    print()


if __name__ == "__main__":
    main()
