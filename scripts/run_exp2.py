#!/usr/bin/env python3
"""Run Experiment 2 — failure injection and per-mode recovery, across policies.

    python scripts/run_exp2.py

Offline and deterministic (no API key / network). Uses simulated execution
traces, so the numbers are a *mechanism demo* of the harness — the measured
paper numbers come from the collaborative Exp 2 run (#4). Replace the simulated
traces with the agreed gold-labeled run before citing in the paper.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orchestratebench import FixedPolicy, HeuristicPolicy  # noqa: E402
from orchestratebench.experiments import (  # noqa: E402
    format_recovery_report,
    run_failure_recovery,
)


def main() -> None:
    policies: dict[str, object] = {"fixed": FixedPolicy(), "heuristic": HeuristicPolicy()}
    results = run_failure_recovery(policies, n_runs=20, seed=0)
    print(format_recovery_report(results))
    print(
        "\n[note] simulated-trace harness demo (#4). Measured numbers come from "
        "the collaborative gold-labeled run; do not cite these as results."
    )


if __name__ == "__main__":
    main()
