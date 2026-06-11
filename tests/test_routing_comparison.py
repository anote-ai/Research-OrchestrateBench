"""Tests for routing_comparison: gold set, LLMPolicy, and the harness.

LLMPolicy is tested with an injected fake client — no API key or network.
"""

import math

import pytest

from orchestratebench import (
    FixedPolicy,
    HeuristicPolicy,
    LLMPolicy,
    OrchestratorAction,
    RoutingDecision,
    SubAgentType,
    format_comparison_report,
    make_routing_eval_set,
    run_routing_comparison,
)


# --------------------------------------------------------------------------- #
# Fakes for the Anthropic client (so LLMPolicy needs no key/network)
# --------------------------------------------------------------------------- #

class _FakeBlock:
    def __init__(self, tool_input):
        self.type = "tool_use"
        self.input = tool_input


class _FakeResponse:
    def __init__(self, content):
        self.content = content


class _FakeMessages:
    def __init__(self, content):
        self._content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse(self._content)


class _FakeClient:
    """Returns a fixed `content` for every messages.create call."""

    def __init__(self, content):
        self.messages = _FakeMessages(content)


def _client_returning(tool_input):
    return _FakeClient([_FakeBlock(tool_input)])


class _OraclePolicy:
    """Routes to the task's gold label — a perfect upper bound for the harness."""

    def route(self, task):
        return OrchestratorAction(
            task_id=task.task_id,
            decision=RoutingDecision(task.metadata["gold"]),
            confidence=1.0,
        )


# --------------------------------------------------------------------------- #
# Gold eval set
# --------------------------------------------------------------------------- #

def test_eval_set_has_both_case_kinds_and_valid_golds():
    eval_set = make_routing_eval_set()
    assert len(eval_set) == 26
    kinds = {t.metadata["case"] for t, _ in eval_set}
    assert kinds == {"aligned", "adversarial"}
    for task, gold in eval_set:
        assert isinstance(gold, RoutingDecision)
        assert task.metadata["gold"] == gold.value
        assert 0.0 <= task.complexity_score <= 1.0


def test_all_four_routes_appear_in_gold():
    golds = {g for _, g in make_routing_eval_set()}
    assert golds == set(RoutingDecision)


# --------------------------------------------------------------------------- #
# Heuristic baseline — the gap the experiment is designed to expose
# --------------------------------------------------------------------------- #

def test_heuristic_aces_aligned_and_misses_all_adversarial():
    report = run_routing_comparison({"heuristic": HeuristicPolicy()})
    by_case = report["policies"]["heuristic"]["accuracy_by_case"]
    # By construction: rule routing is right when flags match intent, wrong when not.
    assert by_case["aligned"] == 1.0
    assert by_case["adversarial"] == 0.0


def test_heuristic_beats_fixed_but_is_not_perfect():
    report = run_routing_comparison(
        {"fixed": FixedPolicy(), "heuristic": HeuristicPolicy()}
    )
    fixed_acc = report["policies"]["fixed"]["routing_accuracy"]
    heur_acc = report["policies"]["heuristic"]["routing_accuracy"]
    assert 0.0 < fixed_acc < heur_acc < 1.0
    assert math.isclose(heur_acc, 16 / 26, rel_tol=1e-9)


# --------------------------------------------------------------------------- #
# LLMPolicy
# --------------------------------------------------------------------------- #

def test_llm_policy_constructs_lazily_without_client():
    policy = LLMPolicy()
    assert isinstance(policy.model, str) and policy.model
    assert policy._client is None  # no client built until route() is called


def test_llm_policy_parses_structured_decision():
    client = _client_returning(
        {"decision": "decompose", "reasoning": "multi-step", "confidence": 0.91}
    )
    policy = LLMPolicy(model="claude-sonnet-4-6", client=client)
    task = make_routing_eval_set()[0][0]
    action = policy.route(task)
    assert action.decision == RoutingDecision.DECOMPOSE
    assert action.selected_agent == SubAgentType.PLANNING
    assert math.isclose(action.confidence, 0.91)
    # forced single-tool call on the requested model
    call = client.messages.calls[0]
    assert call["model"] == "claude-sonnet-4-6"
    assert call["tool_choice"] == {"type": "tool", "name": "route_task"}
    assert len(call["tools"]) == 1


def test_llm_policy_direct_tool_uses_retrieval_agent_when_retrieval_task():
    client = _client_returning(
        {"decision": "direct_tool", "reasoning": "lookup", "confidence": 0.8}
    )
    # craft a retrieval task
    task = next(
        t for t, g in make_routing_eval_set()
        if g == RoutingDecision.DIRECT_TOOL and t.requires_retrieval
    )
    action = LLMPolicy(client=client).route(task)
    assert action.decision == RoutingDecision.DIRECT_TOOL
    assert action.selected_agent == SubAgentType.RETRIEVAL


def test_llm_policy_falls_back_on_missing_tool_call():
    client = _FakeClient([])  # no tool_use block at all
    action = LLMPolicy(client=client).route(make_routing_eval_set()[0][0])
    assert action.decision == RoutingDecision.REASON_ONLY
    assert action.confidence == 0.3


def test_llm_policy_falls_back_on_invalid_decision_value():
    client = _client_returning(
        {"decision": "teleport", "reasoning": "nonsense", "confidence": 0.9}
    )
    action = LLMPolicy(client=client).route(make_routing_eval_set()[0][0])
    assert action.decision == RoutingDecision.REASON_ONLY
    assert action.confidence == 0.3


def test_llm_policy_clamps_out_of_range_confidence():
    client = _client_returning(
        {"decision": "reason_only", "reasoning": "x", "confidence": 5.0}
    )
    action = LLMPolicy(client=client).route(make_routing_eval_set()[0][0])
    assert action.confidence == 1.0


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #

def test_run_comparison_report_shape_and_oracle_upper_bound():
    report = run_routing_comparison(
        {
            "fixed": FixedPolicy(),
            "heuristic": HeuristicPolicy(),
            "oracle": _OraclePolicy(),
        }
    )
    assert report["n"] == 26
    assert set(report["policies"]) == {"fixed", "heuristic", "oracle"}
    accs = {k: v["routing_accuracy"] for k, v in report["policies"].items()}
    assert accs["oracle"] == 1.0
    assert accs["fixed"] < accs["heuristic"] < accs["oracle"]
    for metrics in report["policies"].values():
        assert 0.0 <= metrics["macro_f1"] <= 1.0
        assert len(metrics["predictions"]) == 26


def test_format_comparison_report_is_readable():
    report = run_routing_comparison({"heuristic": HeuristicPolicy()})
    text = format_comparison_report(report)
    assert "heuristic" in text
    assert "accuracy" in text
    assert "adversarial" in text
