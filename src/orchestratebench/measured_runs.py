"""Schemas and loaders for collaborative measured Exp 2/3 runs.

The simulated harness in ``experiments.py`` is useful for mechanism testing,
but the paper ultimately needs *measured* records from the collaborative run.
This module defines the accepted schema for those records and loads them from
CSV / JSON / JSONL into the same long-form structure used by the offline
artifact pipeline.
"""

from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Type

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .failures import FailureMode


class MeasuredRunBase(BaseModel):
    """Common fields shared by measured Exp 2 and Exp 3 records."""

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    policy: str = Field(min_length=1)
    failure_mode: FailureMode
    run: int = Field(ge=0)
    injected_task_success: bool
    final_task_success: bool
    cascade_radius: float = Field(ge=0.0)
    recovery_completeness: float = Field(ge=0.0, le=1.0)
    time_to_detection_ms: float = Field(ge=0.0)
    escalated: bool
    escalation_latency_ms: float = Field(ge=0.0)
    seed: int | None = None
    scenario_id: str | None = None


class MeasuredExp2Record(MeasuredRunBase):
    """Schema for collaborative measured Exp 2 inputs."""

    workflow: str = Field(min_length=1)
    injection_stage: int = Field(ge=0)


class MeasuredExp3Record(MeasuredRunBase):
    """Schema for collaborative measured Exp 3 inputs."""

    depth: int = Field(ge=1)
    injection_stage: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_injection_stage(self) -> "MeasuredExp3Record":
        if self.injection_stage >= self.depth:
            raise ValueError(
                f"injection_stage must be < depth; got stage={self.injection_stage}, depth={self.depth}"
            )
        return self


def _load_raw_records(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    if suffix == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                text = line.strip()
                if not text:
                    continue
                payload = json.loads(text)
                if not isinstance(payload, dict):
                    raise ValueError(f"{path}:{line_no} must contain a JSON object per line")
                rows.append(payload)
        return rows
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            if not all(isinstance(item, dict) for item in payload):
                raise ValueError(f"{path} JSON list must contain only objects")
            return payload
        if isinstance(payload, dict):
            records = payload.get("records")
            if isinstance(records, list) and all(isinstance(item, dict) for item in records):
                return records
        raise ValueError(f"{path} JSON must be a list of objects or {{\"records\": [...]}}")
    raise ValueError(
        f"Unsupported measured-run file format for {path}. Use .csv, .jsonl, or .json."
    )


def _validate_records(
    records: Iterable[Dict[str, Any]],
    model_cls: Type[MeasuredRunBase],
    *,
    source: str,
) -> List[Dict[str, Any]]:
    validated: List[Dict[str, Any]] = []
    for idx, row in enumerate(records, start=1):
        try:
            model = model_cls.model_validate(row)
        except ValidationError as exc:
            raise ValueError(f"{source} record #{idx} failed validation: {exc}") from exc
        validated.append(model.model_dump(mode="json"))
    if not validated:
        raise ValueError(f"{source} contained no records")
    return validated


def load_exp2_measured_records(path: str | Path) -> List[Dict[str, Any]]:
    """Load and validate collaborative measured Exp 2 records."""
    rows = _load_raw_records(path)
    return _validate_records(rows, MeasuredExp2Record, source=str(path))


def load_exp3_measured_records(path: str | Path) -> List[Dict[str, Any]]:
    """Load and validate collaborative measured Exp 3 records."""
    rows = _load_raw_records(path)
    return _validate_records(rows, MeasuredExp3Record, source=str(path))


def prefer_scenario_id(
    records: Sequence[Dict[str, Any]],
    fallback: Sequence[str],
) -> tuple[str, ...]:
    """Use ``scenario_id`` for pairing when every record provides one."""
    if records and all(str(record.get("scenario_id") or "").strip() for record in records):
        return ("scenario_id",)
    return tuple(fallback)


def _group_rows(
    records: Sequence[Dict[str, Any]],
    keys: Sequence[str],
) -> Dict[tuple[object, ...], List[Dict[str, Any]]]:
    grouped: Dict[tuple[object, ...], List[Dict[str, Any]]] = {}
    for record in records:
        group = tuple(record[key] for key in keys)
        grouped.setdefault(group, []).append(record)
    return grouped


def analyze_pairwise_compatibility(
    records: Sequence[Dict[str, Any]],
    *,
    group_keys: Sequence[str],
    scenario_keys: Sequence[str],
) -> Dict[str, Any]:
    """Inspect whether a measured dataset is safe for paired policy comparisons."""
    grouped = _group_rows(records, group_keys)
    all_policies = sorted({str(record["policy"]) for record in records})
    duplicate_scenarios: List[Dict[str, Any]] = []
    missing_policy_groups: List[Dict[str, Any]] = []
    no_overlap_pairs: List[Dict[str, Any]] = []
    overlap_counts: List[int] = []

    for group in sorted(grouped):
        rows = grouped[group]
        group_payload = {
            group_key: value for group_key, value in zip(group_keys, group)
        }
        by_policy: Dict[str, Dict[tuple[object, ...], Dict[str, Any]]] = {}
        scenario_counts: Dict[str, Dict[tuple[object, ...], int]] = {}

        for row in rows:
            policy = str(row["policy"])
            scenario = tuple(row[key] for key in scenario_keys)
            scenario_counts.setdefault(policy, {})
            scenario_counts[policy][scenario] = scenario_counts[policy].get(scenario, 0) + 1
            by_policy.setdefault(policy, {})
            by_policy[policy].setdefault(scenario, row)

        for policy, counts in scenario_counts.items():
            for scenario, count in counts.items():
                if count <= 1:
                    continue
                duplicate_scenarios.append(
                    {
                        **group_payload,
                        "policy": policy,
                        "scenario": {
                            key: value for key, value in zip(scenario_keys, scenario)
                        },
                        "count": count,
                    }
                )

        missing = [policy for policy in all_policies if policy not in by_policy]
        if missing:
            missing_policy_groups.append(
                {
                    **group_payload,
                    "missing_policies": missing,
                }
            )

        for policy_a, policy_b in combinations(sorted(by_policy), 2):
            common = sorted(set(by_policy[policy_a]).intersection(by_policy[policy_b]))
            if common:
                overlap_counts.append(len(common))
                continue
            no_overlap_pairs.append(
                {
                    **group_payload,
                    "policy_a": policy_a,
                    "policy_b": policy_b,
                }
            )

    return {
        "group_keys": list(group_keys),
        "scenario_keys": list(scenario_keys),
        "policy_count": len(all_policies),
        "group_count": len(grouped),
        "duplicate_scenarios": duplicate_scenarios,
        "missing_policy_groups": missing_policy_groups,
        "no_overlap_pairs": no_overlap_pairs,
        "min_shared_pairs": min(overlap_counts) if overlap_counts else 0,
        "max_shared_pairs": max(overlap_counts) if overlap_counts else 0,
    }
