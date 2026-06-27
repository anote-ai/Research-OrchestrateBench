"""Artifact writers shared across experiment runners and reporting tools."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def write_records_csv(records: list[dict[str, object]], path: str | Path) -> None:
    """Write long-form experiment records to CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for record in records for key in record})
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_json_file(payload: object, path: str | Path) -> None:
    """Write experiment artifacts as pretty JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
