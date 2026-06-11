"""Routing-quality comparison: heuristic (rule/keyword) vs LLM (model-driven).

This is the measurable instantiation of the reasoning trade-off Natan raised
(6/9): Panacea routes with a fixed rule/keyword scheme (see
``HeuristicPolicy``), while AICT lets the model decide per turn. This module
adds an ``LLMPolicy`` that routes by reading the *task intent* with Claude, and
a gold-labelled eval set + harness so the two can be compared on a single,
reproducible number — ``routing_accuracy`` — rather than vibes.

It is also the concrete Experiment 1 (routing policy comparison) from the
OrchestraBench design doc: Fixed / Heuristic / LLM on the same tasks.

The eval set deliberately contains two kinds of cases:

* **aligned** — the task's surface flags (``requires_code`` /
  ``requires_retrieval`` / ``complexity_score``) match its true intent, so a
  rule-based router and a model-based router should both get them right.
* **adversarial** — the surface flags are missing or misleading, but the
  *description* makes the right route obvious. This is exactly where keyword /
  flag routing misfires in production and where model-driven routing should
  win.

Running the LLM policy needs an Anthropic API key (``ANTHROPIC_API_KEY``); the
Fixed/Heuristic baselines and the whole harness run offline.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from .core import AgentTask, OrchestratorAction, RoutingDecision, SubAgentType
from .evaluate import (
    per_class_routing_metrics,
    routing_accuracy,
    routing_macro_f1,
)

# Default to the most capable model so the comparison reflects the *best* the
# model-driven approach can do (override via env without touching code).
DEFAULT_LLM_MODEL = os.getenv("ORCHESTRATEBENCH_LLM_MODEL", "claude-opus-4-8")

_DECISION_GUIDE = {
    RoutingDecision.DIRECT_TOOL: (
        "a single lookup/fetch/retrieval that one tool answers directly"
    ),
    RoutingDecision.CODE_EXECUTION: (
        "the task needs code to be written and/or run (compute, parse, script, chart)"
    ),
    RoutingDecision.DECOMPOSE: (
        "a multi-step goal that must be broken into ordered sub-tasks"
    ),
    RoutingDecision.REASON_ONLY: (
        "the model can answer from its own knowledge; no tool or decomposition needed"
    ),
}

_ROUTE_TOOL = {
    "name": "route_task",
    "description": "Decide how the orchestrator should handle the task.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "decision": {
                "type": "string",
                "enum": [d.value for d in RoutingDecision],
                "description": "The routing decision for this task.",
            },
            "reasoning": {
                "type": "string",
                "description": "One sentence: why this route fits the task's intent.",
            },
            "confidence": {
                "type": "number",
                "description": "Confidence in the decision, 0.0-1.0.",
            },
        },
        "required": ["decision", "reasoning", "confidence"],
        "additionalProperties": False,
    },
}

_AGENT_FOR_DECISION = {
    RoutingDecision.CODE_EXECUTION: SubAgentType.CODE,
    RoutingDecision.DECOMPOSE: SubAgentType.PLANNING,
    RoutingDecision.REASON_ONLY: None,
}


def _clamp01(value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, v))


class LLMPolicy:
    """Model-driven router: Claude reads the task and picks the route.

    Mirrors AICT's Agents-SDK style (the model decides) rather than Panacea's
    keyword/flag scheme. Uses a single forced tool call so the output is always
    a structured ``{decision, reasoning, confidence}``.

    The Anthropic client is injectable (``client=``) so tests run without a key
    or network; import of the ``anthropic`` package is lazy so the module loads
    even where it isn't installed.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        client: Any = None,
        max_tokens: int = 1024,
    ) -> None:
        self.model = model or DEFAULT_LLM_MODEL
        self._client = client
        self._max_tokens = max_tokens

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic  # lazy: only needed when actually routing

            self._client = anthropic.Anthropic()
        return self._client

    def _build_prompt(self, task: AgentTask) -> str:
        options = "\n".join(
            f"- {d.value}: {desc}" for d, desc in _DECISION_GUIDE.items()
        )
        return (
            "You are the orchestrator for a multi-agent system. Choose the single "
            "best route for the task below by reasoning about what it actually "
            "requires — judge the intent of the description, not just the metadata "
            "flags (they may be missing or wrong).\n\n"
            f"Task: {task.description}\n"
            f"Metadata (may be noisy): complexity_score={task.complexity_score}, "
            f"requires_code={task.requires_code}, "
            f"requires_retrieval={task.requires_retrieval}\n\n"
            f"Routes:\n{options}\n\n"
            "Call route_task with your decision."
        )

    def route(self, task: AgentTask) -> OrchestratorAction:
        # NOTE: no temperature/top_p/top_k — removed on opus-4-8 (would 400).
        response = self._get_client().messages.create(
            model=self.model,
            max_tokens=self._max_tokens,
            tools=[_ROUTE_TOOL],
            tool_choice={"type": "tool", "name": "route_task"},
            messages=[{"role": "user", "content": self._build_prompt(task)}],
        )
        return self._parse(task, response)

    def _parse(self, task: AgentTask, response: Any) -> OrchestratorAction:
        payload = self._extract_tool_input(response)
        try:
            decision = RoutingDecision(payload["decision"])
        except (KeyError, ValueError, TypeError):
            # Defensive: never crash a benchmark run on a malformed reply.
            return OrchestratorAction(
                task_id=task.task_id,
                decision=RoutingDecision.REASON_ONLY,
                selected_agent=None,
                reasoning="LLMPolicy: unparseable response, fell back to reason_only.",
                confidence=0.3,
            )
        agent = _AGENT_FOR_DECISION.get(decision, SubAgentType.TOOL_CALL)
        if decision == RoutingDecision.DIRECT_TOOL and task.requires_retrieval:
            agent = SubAgentType.RETRIEVAL
        return OrchestratorAction(
            task_id=task.task_id,
            decision=decision,
            selected_agent=agent,
            reasoning=str(payload.get("reasoning", "")) or "LLM routing decision.",
            confidence=_clamp01(payload.get("confidence", 0.8)),
        )

    @staticmethod
    def _extract_tool_input(response: Any) -> Dict[str, Any]:
        for block in getattr(response, "content", []) or []:
            if getattr(block, "type", None) == "tool_use":
                value = getattr(block, "input", None)
                if isinstance(value, str):
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        return {}
                return value if isinstance(value, dict) else {}
        return {}


# --------------------------------------------------------------------------- #
# Gold-labelled routing eval set
# --------------------------------------------------------------------------- #

GoldCase = Tuple[AgentTask, RoutingDecision]


def _case(
    description: str,
    gold: RoutingDecision,
    *,
    kind: str,
    complexity: float,
    requires_code: bool = False,
    requires_retrieval: bool = False,
) -> GoldCase:
    task = AgentTask(
        description=description,
        complexity_score=complexity,
        requires_code=requires_code,
        requires_retrieval=requires_retrieval,
        metadata={"case": kind, "gold": gold.value},
    )
    return task, gold


def make_routing_eval_set() -> List[GoldCase]:
    """Return (task, gold_decision) pairs: 16 aligned + 10 adversarial."""
    D = RoutingDecision
    aligned: List[GoldCase] = [
        _case("Look up the current account balance for customer #4821.",
              D.DIRECT_TOOL, kind="aligned", complexity=0.2, requires_retrieval=True),
        _case("Fetch the latest invoice PDF for order 9931.",
              D.DIRECT_TOOL, kind="aligned", complexity=0.25, requires_retrieval=True),
        _case("Retrieve the contract section covering termination clauses.",
              D.DIRECT_TOOL, kind="aligned", complexity=0.3, requires_retrieval=True),
        _case("Pull the employee's remaining PTO balance for this year.",
              D.DIRECT_TOOL, kind="aligned", complexity=0.2, requires_retrieval=True),
        _case("Compute the 30-day moving average from the attached sales CSV.",
              D.CODE_EXECUTION, kind="aligned", complexity=0.5, requires_code=True),
        _case("Parse the server log file and count error codes by hour.",
              D.CODE_EXECUTION, kind="aligned", complexity=0.45, requires_code=True),
        _case("Run the unit-test suite and report which tests fail.",
              D.CODE_EXECUTION, kind="aligned", complexity=0.5, requires_code=True),
        _case("Generate a bar chart from the quarterly revenue table.",
              D.CODE_EXECUTION, kind="aligned", complexity=0.5, requires_code=True),
        _case("Plan and execute a full quarterly financial close across AP, AR, "
              "and reconciliation.",
              D.DECOMPOSE, kind="aligned", complexity=0.85),
        _case("Onboard a new enterprise client: provision accounts, configure SSO, "
              "migrate data, and schedule training.",
              D.DECOMPOSE, kind="aligned", complexity=0.9),
        _case("Design, draft, review, and submit an RFP response with three "
              "distinct technical subsections.",
              D.DECOMPOSE, kind="aligned", complexity=0.8),
        _case("Coordinate the product launch: marketing, engineering sign-off, "
              "and a staged rollout plan.",
              D.DECOMPOSE, kind="aligned", complexity=0.88),
        _case("Explain the difference between gross margin and net margin.",
              D.REASON_ONLY, kind="aligned", complexity=0.2),
        _case("Summarize the single biggest risk here in one sentence.",
              D.REASON_ONLY, kind="aligned", complexity=0.15),
        _case("Which of these two options is riskier, and why?",
              D.REASON_ONLY, kind="aligned", complexity=0.25),
        _case("Rephrase this paragraph to be more concise.",
              D.REASON_ONLY, kind="aligned", complexity=0.2),
    ]
    # Adversarial: flags/score mislead the rule-based router; intent is clear.
    adversarial: List[GoldCase] = [
        _case("Break this initiative into sub-tasks, assign owners, and sequence "
              "them by dependency.",
              D.DECOMPOSE, kind="adversarial", complexity=0.6),  # heuristic->reason_only
        _case("Write a Python script to deduplicate the customer list and run it.",
              D.CODE_EXECUTION, kind="adversarial", complexity=0.5),  # code flag missing
        _case("Just look up the CEO's name in the latest company filing.",
              D.DIRECT_TOOL, kind="adversarial", complexity=0.82),  # high score -> decompose
        _case("What does the acronym 'EBITDA' stand for?",
              D.REASON_ONLY, kind="adversarial", complexity=0.3, requires_retrieval=True),
        _case("Create a multi-step migration plan: audit, map, transfer, validate.",
              D.DECOMPOSE, kind="adversarial", complexity=0.65),
        _case("Calculate the correlation matrix for these five metrics programmatically.",
              D.CODE_EXECUTION, kind="adversarial", complexity=0.4),
        _case("Find the phone number listed on page 2 of the document.",
              D.DIRECT_TOOL, kind="adversarial", complexity=0.9),
        _case("Is 17 a prime number?",
              D.REASON_ONLY, kind="adversarial", complexity=0.2, requires_retrieval=True),
        _case("Plan the team offsite: venue, agenda, travel, and budget, in order.",
              D.DECOMPOSE, kind="adversarial", complexity=0.55),
        _case("Run a regex over the dataset to extract every email address.",
              D.CODE_EXECUTION, kind="adversarial", complexity=0.5),
    ]
    return aligned + adversarial


# --------------------------------------------------------------------------- #
# Comparison harness
# --------------------------------------------------------------------------- #

def run_routing_comparison(
    policies: Dict[str, Any],
    eval_set: Optional[List[GoldCase]] = None,
) -> Dict[str, Any]:
    """Score each policy on the gold set; reuses the existing routing metrics."""
    eval_set = eval_set or make_routing_eval_set()
    tasks = [t for t, _ in eval_set]
    gold = [g.value for _, g in eval_set]
    report: Dict[str, Any] = {"n": len(tasks), "policies": {}}
    for name, policy in policies.items():
        preds = [policy.route(t).decision.value for t in tasks]
        report["policies"][name] = {
            "routing_accuracy": routing_accuracy(preds, gold),
            "macro_f1": routing_macro_f1(preds, gold),
            "accuracy_by_case": _accuracy_by_case(preds, eval_set),
            "per_class": per_class_routing_metrics(preds, gold),
            "predictions": preds,
        }
    return report


def _accuracy_by_case(
    preds: List[str], eval_set: List[GoldCase]
) -> Dict[str, float]:
    buckets: Dict[str, List[int]] = {}
    for pred, (task, gold) in zip(preds, eval_set):
        kind = str(task.metadata.get("case", "unknown"))
        buckets.setdefault(kind, []).append(int(pred == gold.value))
    return {
        kind: (sum(hits) / len(hits) if hits else 0.0)
        for kind, hits in buckets.items()
    }


def format_comparison_report(report: Dict[str, Any]) -> str:
    """Render a compact text table (for standup / paper display)."""
    lines = [f"Routing comparison on {report['n']} gold tasks:", ""]
    header = f"{'policy':<14}{'accuracy':>10}{'macro_f1':>10}{'aligned':>10}{'adversarial':>13}"
    lines.append(header)
    lines.append("-" * len(header))
    for name, m in report["policies"].items():
        by_case = m["accuracy_by_case"]
        lines.append(
            f"{name:<14}{m['routing_accuracy']:>10.3f}{m['macro_f1']:>10.3f}"
            f"{by_case.get('aligned', 0.0):>10.3f}{by_case.get('adversarial', 0.0):>13.3f}"
        )
    return "\n".join(lines)
