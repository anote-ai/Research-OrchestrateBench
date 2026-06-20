"""Tests for Exp 2/3 one-command automation helpers."""

from __future__ import annotations

from orchestratebench.automation import (
    build_auto_exp2_measured_records,
    build_auto_exp3_measured_records,
    build_named_policies,
)
from orchestratebench.failures import FailureMode


def test_build_named_policies_supports_default_suite() -> None:
    policies = build_named_policies(("fixed", "heuristic", "retry(heuristic)"))
    assert set(policies) == {"fixed", "heuristic", "retry(heuristic)"}


def test_build_named_policies_rejects_unknown_name() -> None:
    try:
        build_named_policies(("not-a-policy",))
    except ValueError as exc:
        assert "Unsupported policy" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown policy")


def test_build_auto_exp2_measured_records_returns_measured_shape() -> None:
    rows = build_auto_exp2_measured_records(
        policy_names=("fixed", "heuristic"),
        modes=(FailureMode.TOOL_INVOCATION_ERROR,),
        n_runs=2,
        seed=0,
    )
    assert rows
    assert {"policy", "workflow", "scenario_id", "annotator", "notes"} <= set(rows[0])
    assert isinstance(rows[0]["injected_task_success"], bool)
    assert rows[0]["annotator"] == "auto_harness"


def test_build_auto_exp2_measured_records_reuses_scenario_id_across_policies() -> None:
    rows = build_auto_exp2_measured_records(
        policy_names=("fixed", "heuristic"),
        modes=(FailureMode.TOOL_INVOCATION_ERROR,),
        n_runs=1,
        seed=0,
    )
    first_two = rows[:2]
    assert first_two[0]["scenario_id"] == first_two[1]["scenario_id"]
    assert first_two[0]["policy"] != first_two[1]["policy"]


def test_build_auto_exp3_measured_records_returns_measured_shape() -> None:
    rows = build_auto_exp3_measured_records(
        policy_names=("fixed", "heuristic"),
        depths=(3,),
        injection_stages=(0, 1),
        mode=FailureMode.CONTEXT_POLLUTION,
        n_runs=2,
        seed=0,
    )
    assert rows
    assert {"policy", "depth", "scenario_id", "annotator", "notes"} <= set(rows[0])
    assert isinstance(rows[0]["final_task_success"], bool)
    assert rows[0]["failure_mode"] == "context_pollution"
