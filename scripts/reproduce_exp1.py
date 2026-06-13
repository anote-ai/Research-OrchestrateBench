#!/usr/bin/env python3
"""Reproduce Experiment 1 — the routing-policy comparison.

Regenerates the headline routing-accuracy table from the committed 26-case gold
set, so every number in the paper/design doc is reproducible from one command:

    python scripts/reproduce_exp1.py

The offline baselines (Fixed / Heuristic / Oracle) need no API key or network.
The model-driven LLM row runs only when ``ANTHROPIC_API_KEY`` is set; the model
is configurable via ``ORCHESTRATEBENCH_LLM_MODEL`` (default: cost-efficient
Sonnet 4.6). Expected offline result: Fixed 23% / Heuristic 62% (adversarial 0%)
/ Oracle 100%; LLM (Sonnet 4.6) saturates at 100% incl. adversarial when run
with a valid Anthropic key.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orchestratebench import (  # noqa: E402
    FixedPolicy,
    HeuristicPolicy,
    LLMPolicy,
    OrchestratorAction,
    RoutingDecision,
    format_comparison_report,
    make_routing_eval_set,
    run_routing_comparison,
)

MODEL = os.getenv("ORCHESTRATEBENCH_LLM_MODEL", "claude-sonnet-4-6")


class OraclePolicy:
    """Routes to the gold label — a perfect upper bound (free, deterministic)."""

    def route(self, task: object) -> OrchestratorAction:
        return OrchestratorAction(
            task_id=task.task_id,  # type: ignore[attr-defined]
            decision=RoutingDecision(task.metadata["gold"]),  # type: ignore[attr-defined]
            confidence=1.0,
        )


def main() -> None:
    eval_set = make_routing_eval_set()
    policies: dict[str, object] = {
        "fixed": FixedPolicy(),
        "heuristic": HeuristicPolicy(),
        "oracle": OraclePolicy(),
    }
    if os.getenv("ANTHROPIC_API_KEY"):
        policies[f"llm({MODEL})"] = LLMPolicy(model=MODEL)
    else:
        print(
            "[note] ANTHROPIC_API_KEY not set — running offline baselines only "
            "(set it to reproduce the LLM-as-Router row).\n"
        )

    report = run_routing_comparison(policies, eval_set)
    print(format_comparison_report(report))

    print("\naligned / adversarial split:")
    for name, r in report["policies"].items():
        by_case = r["accuracy_by_case"]
        print(
            f"  {name:>26}: overall={r['routing_accuracy']:.3f} "
            f"aligned={by_case.get('aligned', 0.0):.3f} "
            f"adversarial={by_case.get('adversarial', 0.0):.3f}"
        )


if __name__ == "__main__":
    main()
