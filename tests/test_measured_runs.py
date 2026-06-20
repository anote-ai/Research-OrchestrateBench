"""Tests for measured-run schemas and loaders."""

from __future__ import annotations

import json

import pytest

from orchestratebench.measured_runs import (
    analyze_pairwise_compatibility,
    load_exp2_measured_records,
    load_exp3_measured_records,
    prefer_scenario_id,
)


def test_load_exp2_measured_records_from_csv(tmp_path) -> None:
    path = tmp_path / "exp2.csv"
    path.write_text(
        "\n".join(
            [
                "policy,workflow,failure_mode,run,injection_stage,injected_task_success,final_task_success,cascade_radius,recovery_completeness,time_to_detection_ms,escalated,escalation_latency_ms,scenario_id,annotator",
                "retry(heuristic),finance_approval,tool_invocation_error,0,1,true,true,0,1.0,250.0,false,0.0,finance-tool-0,yg",
            ]
        ),
        encoding="utf-8",
    )
    records = load_exp2_measured_records(path)
    assert len(records) == 1
    assert records[0]["failure_mode"] == "tool_invocation_error"
    assert records[0]["scenario_id"] == "finance-tool-0"
    assert records[0]["annotator"] == "yg"


def test_load_exp3_measured_records_from_jsonl(tmp_path) -> None:
    path = tmp_path / "exp3.jsonl"
    payload = {
        "policy": "heuristic",
        "depth": 5,
        "failure_mode": "context_pollution",
        "run": 0,
        "injection_stage": 2,
        "injected_task_success": False,
        "final_task_success": False,
        "cascade_radius": 2,
        "recovery_completeness": 0.0,
        "time_to_detection_ms": 812.4,
        "escalated": True,
        "escalation_latency_ms": 812.4,
        "scenario_id": "depth5-stage3-run0",
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    records = load_exp3_measured_records(path)
    assert len(records) == 1
    assert records[0]["depth"] == 5
    assert records[0]["failure_mode"] == "context_pollution"


def test_load_exp2_measured_records_from_json_object_records_key(tmp_path) -> None:
    path = tmp_path / "exp2.json"
    payload = {
        "records": [
            {
                "policy": "fixed",
                "workflow": "hr_onboarding",
                "failure_mode": "premature_action",
                "run": 1,
                "injection_stage": 0,
                "injected_task_success": False,
                "final_task_success": False,
                "cascade_radius": 4,
                "recovery_completeness": 0.0,
                "time_to_detection_ms": 500.0,
                "escalated": True,
                "escalation_latency_ms": 500.0,
            }
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    records = load_exp2_measured_records(path)
    assert records[0]["workflow"] == "hr_onboarding"


def test_invalid_exp3_record_raises_on_injection_stage_ge_depth(tmp_path) -> None:
    path = tmp_path / "bad_exp3.jsonl"
    payload = {
        "policy": "fixed",
        "depth": 3,
        "failure_mode": "context_pollution",
        "run": 0,
        "injection_stage": 3,
        "injected_task_success": False,
        "final_task_success": False,
        "cascade_radius": 0,
        "recovery_completeness": 1.0,
        "time_to_detection_ms": 100.0,
        "escalated": False,
        "escalation_latency_ms": 0.0,
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="injection_stage must be < depth"):
        load_exp3_measured_records(path)


def test_invalid_failure_mode_raises(tmp_path) -> None:
    path = tmp_path / "bad_exp2.csv"
    path.write_text(
        "\n".join(
            [
                "policy,workflow,failure_mode,run,injection_stage,injected_task_success,final_task_success,cascade_radius,recovery_completeness,time_to_detection_ms,escalated,escalation_latency_ms",
                "fixed,finance_approval,not_a_mode,0,0,false,false,1,0.0,100.0,true,100.0",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="failed validation"):
        load_exp2_measured_records(path)


def test_empty_measured_file_raises(tmp_path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="contained no records"):
        load_exp2_measured_records(path)


def test_prefer_scenario_id_uses_it_when_available() -> None:
    records = [
        {"scenario_id": "s1", "run": 0},
        {"scenario_id": "s2", "run": 1},
    ]
    assert prefer_scenario_id(records, fallback=("run",)) == ("scenario_id",)


def test_prefer_scenario_id_falls_back_when_missing() -> None:
    records = [
        {"run": 0},
        {"run": 1},
    ]
    assert prefer_scenario_id(records, fallback=("run", "injection_stage")) == (
        "run",
        "injection_stage",
    )


def test_analyze_pairwise_compatibility_reports_duplicates_and_missing_policies() -> None:
    records = [
        {
            "policy": "fixed",
            "failure_mode": "tool_invocation_error",
            "workflow": "finance_approval",
            "run": 0,
            "injection_stage": 1,
            "scenario_id": "shared-0",
        },
        {
            "policy": "fixed",
            "failure_mode": "tool_invocation_error",
            "workflow": "finance_approval",
            "run": 0,
            "injection_stage": 1,
            "scenario_id": "shared-0",
        },
        {
            "policy": "heuristic",
            "failure_mode": "tool_invocation_error",
            "workflow": "finance_approval",
            "run": 0,
            "injection_stage": 1,
            "scenario_id": "shared-0",
        },
    ]
    analysis = analyze_pairwise_compatibility(
        records,
        group_keys=("failure_mode",),
        scenario_keys=("scenario_id",),
    )
    assert analysis["group_count"] == 1
    assert analysis["min_shared_pairs"] == 1
    assert len(analysis["duplicate_scenarios"]) == 1
    assert analysis["duplicate_scenarios"][0]["policy"] == "fixed"
    assert len(analysis["missing_policy_groups"]) == 0
    assert len(analysis["no_overlap_pairs"]) == 0


def test_analyze_pairwise_compatibility_reports_no_overlap_pairs() -> None:
    records = [
        {
            "policy": "fixed",
            "depth": 5,
            "injection_stage": 2,
            "run": 0,
        },
        {
            "policy": "heuristic",
            "depth": 5,
            "injection_stage": 2,
            "run": 1,
        },
    ]
    analysis = analyze_pairwise_compatibility(
        records,
        group_keys=("depth", "injection_stage"),
        scenario_keys=("run",),
    )
    assert analysis["group_count"] == 1
    assert analysis["min_shared_pairs"] == 0
    assert analysis["max_shared_pairs"] == 0
    assert len(analysis["duplicate_scenarios"]) == 0
    assert len(analysis["missing_policy_groups"]) == 0
    assert len(analysis["no_overlap_pairs"]) == 1
