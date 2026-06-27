"""Smoke tests for packaged CLI entrypoints."""

from __future__ import annotations

import json

from orchestratebench.cli.run_exp2 import main as run_exp2_main
from orchestratebench.cli.run_exp3 import main as run_exp3_main
from orchestratebench.cli.run_exp23_pipeline import main as run_exp23_pipeline_main


def test_run_exp2_cli_writes_artifacts(tmp_path) -> None:
    output_dir = tmp_path / "exp2"
    exit_code = run_exp2_main(
        ["--n-runs", "1", "--output-dir", str(output_dir), "--n-resamples", "20"]
    )

    assert exit_code == 0
    assert (output_dir / "raw_runs.csv").exists()
    assert (output_dir / "summary.json").exists()
    payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["config"]["source_mode"] == "simulated"


def test_run_exp3_cli_writes_artifacts(tmp_path) -> None:
    output_dir = tmp_path / "exp3"
    exit_code = run_exp3_main(
        [
            "--n-runs",
            "1",
            "--depths",
            "3",
            "--injection-stages",
            "0",
            "--output-dir",
            str(output_dir),
            "--n-resamples",
            "20",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "raw_runs.csv").exists()
    assert (output_dir / "summary.json").exists()
    payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["config"]["source_mode"] == "simulated"


def test_run_exp23_pipeline_cli_writes_manifest(tmp_path) -> None:
    output_root = tmp_path / "pipeline"
    exit_code = run_exp23_pipeline_main(
        [
            "--output-root",
            str(output_root),
            "--exp2-runs",
            "1",
            "--exp3-runs",
            "1",
            "--depths",
            "3",
            "--injection-stages",
            "0",
            "--exp3-modes",
            "context_pollution",
            "--n-resamples",
            "20",
        ]
    )

    assert exit_code == 0
    manifest_path = output_root / "pipeline_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["pipeline"] == "run_exp23_pipeline"
    assert (output_root / "analysis" / "exp2" / "summary.json").exists()
    assert (
        output_root / "analysis" / "exp3" / "context_pollution" / "summary.json"
    ).exists()
