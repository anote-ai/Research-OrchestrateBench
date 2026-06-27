"""CLI for reproducing Experiment 1 routing comparisons."""

from __future__ import annotations

import os

from orchestratebench.core import FixedPolicy, HeuristicPolicy, OrchestratorAction, RoutingDecision
from orchestratebench.routing_comparison import (
    LLMPolicy,
    format_comparison_report,
    make_routing_eval_set,
    run_routing_comparison,
)

MODEL = os.getenv("ORCHESTRATEBENCH_LLM_MODEL", "claude-sonnet-4-6")


class OraclePolicy:
    """Routes to the gold label as a deterministic upper bound."""

    def route(self, task: object) -> OrchestratorAction:
        return OrchestratorAction(
            task_id=task.task_id,  # type: ignore[attr-defined]
            decision=RoutingDecision(task.metadata["gold"]),  # type: ignore[attr-defined]
            confidence=1.0,
        )


def main(argv: list[str] | None = None) -> int:
    del argv
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
    for name, result in report["policies"].items():
        by_case = result["accuracy_by_case"]
        print(
            f"  {name:>26}: overall={result['routing_accuracy']:.3f} "
            f"aligned={by_case.get('aligned', 0.0):.3f} "
            f"adversarial={by_case.get('adversarial', 0.0):.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
