#!/usr/bin/env python3
"""Run Experiment 3 — cascade radius vs. pipeline depth, across policies.

    python scripts/run_exp3.py

Offline and deterministic. Uses simulated execution traces, so the numbers are
a *mechanism demo* of the harness — the measured paper numbers come from the
collaborative Exp 3 run (#7). Do not cite these as results.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orchestratebench import FixedPolicy, HeuristicPolicy  # noqa: E402
from orchestratebench.experiments import (  # noqa: E402
    format_cascade_report,
    run_cascade_by_depth,
)


def main() -> None:
    policies: dict[str, object] = {"fixed": FixedPolicy(), "heuristic": HeuristicPolicy()}
    results = run_cascade_by_depth(policies, depths=[3, 5, 7], n_runs=20, seed=0)
    print(format_cascade_report(results))
    print(
        "\n[note] simulated-trace harness demo (#7). Measured numbers come from "
        "the collaborative gold-labeled run; do not cite these as results."
    )


if __name__ == "__main__":
    main()
