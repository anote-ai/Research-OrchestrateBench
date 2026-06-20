"""Tests for measured-input skeleton scaffolds."""

from __future__ import annotations

from orchestratebench.failures import FailureMode
from orchestratebench.measured_templates import (
    EXP2_SKELETON_FIELDS,
    EXP3_SKELETON_FIELDS,
    build_exp2_measured_skeleton,
    build_exp3_measured_skeleton,
)


def test_build_exp2_measured_skeleton_shape() -> None:
    rows = build_exp2_measured_skeleton(
        policies=("fixed", "heuristic"),
        workflows=("finance_approval",),
        modes=(FailureMode.TOOL_INVOCATION_ERROR,),
        n_runs=2,
        seed=0,
    )
    assert len(rows) == 4
    assert tuple(rows[0].keys()) == EXP2_SKELETON_FIELDS
    assert rows[0]["failure_mode"] == "tool_invocation_error"
    assert rows[0]["injected_task_success"] == ""


def test_build_exp2_measured_skeleton_reuses_scenario_id_across_policies() -> None:
    rows = build_exp2_measured_skeleton(
        policies=("fixed", "heuristic"),
        workflows=("finance_approval",),
        modes=(FailureMode.TOOL_INVOCATION_ERROR,),
        n_runs=1,
        seed=0,
    )
    assert rows[0]["scenario_id"] == rows[1]["scenario_id"]
    assert rows[0]["policy"] != rows[1]["policy"]


def test_build_exp3_measured_skeleton_shape() -> None:
    rows = build_exp3_measured_skeleton(
        policies=("fixed", "heuristic"),
        depths=(3, 5),
        injection_stages=(0, 1, 2),
        mode=FailureMode.CONTEXT_POLLUTION,
        n_runs=2,
        seed=0,
    )
    # depth 3 -> stages 0,1,2 ; depth 5 -> stages 0,1,2 ; each x 2 runs x 2 policies
    assert len(rows) == 24
    assert tuple(rows[0].keys()) == EXP3_SKELETON_FIELDS
    assert rows[0]["failure_mode"] == "context_pollution"
    assert rows[0]["final_task_success"] == ""


def test_build_exp3_measured_skeleton_skips_invalid_stages() -> None:
    rows = build_exp3_measured_skeleton(
        policies=("fixed",),
        depths=(2,),
        injection_stages=(0, 1, 2, 3),
        mode=FailureMode.CONTEXT_POLLUTION,
        n_runs=1,
        seed=0,
    )
    assert [row["injection_stage"] for row in rows] == [0, 1]
