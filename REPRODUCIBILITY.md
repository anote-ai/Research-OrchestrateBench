# Reproducibility Guide

This document explains what is currently reproducible in-repo, what remains a
collaborative measured-data step, and which artifacts should or should not be
treated as paper-ready evidence.

## Result Types

OrchestrateBench currently contains three different evidence types:

| Evidence type | Status | Where it lives | Can it be cited as final paper evidence? |
|---|---|---|---|
| Experiment 1 measured routing diagnostic | Available now | `scripts/reproduce_exp1.py` | Yes, with the usual model/version caveats |
| Experiment 2/3 auto-harness mechanism results | Available now | `scripts/run_exp23_pipeline.py` | Yes for mechanism validation; no as the final external measured result |
| Experiment 2/3 collaborative external measured results | Planned / optional | `--input-file` path supplied by the user | Yes, once those records are collected and validated |

The most important boundary is this:

- `run_exp23_pipeline.py` generates **measured-style** CSVs from the current
  repo harness.
- Those files are valid analyzer inputs, but they are still **repo-generated
  mechanism results**, not externally collected gold measurements.

## Environment

Recommended environment:

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest -q
```

The CI workflow tests Python 3.10, 3.11, and 3.12 on every push / pull
request to `main`.

## Fastest Reproduction Paths

### 1. Policy demo

```bash
python3 scripts/run_demo.py
```

Use this to sanity-check routing, latency, and cost behavior.

### 2. Experiment 1 measured diagnostic

```bash
python3 scripts/reproduce_exp1.py
```

Notes:

- The offline baseline rows run without network access.
- The LLM router row requires `ANTHROPIC_API_KEY`.

### 3. Experiment 2/3 auto-harness pipeline

```bash
./scripts/run_exp23_pipeline.py --with-ci --exp3-modes context_pollution,tool_invocation_error
```

This is the current one-command path for:

- generating measured-style long-form inputs
- validating those inputs
- running Exp 2 / Exp 3 analyses
- exporting raw runs, summaries, pairwise comparisons, and paper-facing tables

Outputs are written under:

- `artifacts/exp23_pipeline/measured_inputs/`
- `artifacts/exp23_pipeline/analysis/exp2/`
- `artifacts/exp23_pipeline/analysis/exp3/<failure_mode>/`
- `artifacts/exp23_pipeline/pipeline_manifest.json`

### 4. Collaborative measured-data path

If you are measuring a real external system rather than the in-repo harness:

```bash
python3 scripts/scaffold_measured_inputs.py --output-dir data/measured
python3 scripts/validate_measured_input.py --experiment 2 --input-file path/to/exp2_measured.csv --strict
python3 scripts/run_exp2.py --input-file path/to/exp2_measured.csv --with-ci
python3 scripts/validate_measured_input.py --experiment 3 --input-file path/to/exp3_measured.jsonl --strict
python3 scripts/run_exp3.py --input-file path/to/exp3_measured.jsonl --with-ci
```

For this path, see also [data/measured/README.md](data/measured/README.md).

## Artifact Semantics

For Exp 2 and Exp 3, the main output files are:

- `raw_runs.csv`: long-form per-run records
- `summary.json`: machine-readable experiment summary and config
- `summary_by_*.csv`: grouped summary tables
- `pairwise_*.csv`: paired bootstrap policy comparisons
- `report.txt`: human-readable text report
- `paper_summary.md`: Markdown summary suitable for the design doc / paper draft
- `paper_tables.tex`: LaTeX tables for paper integration
- `pipeline_manifest.json`: top-level provenance for the one-command pipeline

## Before Citing Numbers

Use this checklist before moving any number from the repo into a design doc,
slide deck, or paper draft:

1. Identify the provenance: Exp 1 measured diagnostic, Exp 2/3 auto-harness, or
   collaborative external measured input.
2. Check the input source in the corresponding `summary.json` or
   `pipeline_manifest.json`.
3. Confirm whether the result is a mechanism validation result or a final
   measured result.
4. Keep the failure mode explicit for Exp 3; each Exp 3 input file should
   contain exactly one `failure_mode`.
5. Preserve the exact command and seed in notes or appendices.

## Current Paper-Safe Framing

The safest current framing is:

- Experiment 1 is already a measured routing diagnostic in-repo.
- Experiment 2/3 auto-harness results are strong enough to support the paper's
  **mechanism claims**.
- If the paper needs final claims about external collaborative measurements,
  those should come from validated measured inputs supplied through
  `--input-file`.
