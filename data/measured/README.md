# Measured Exp 2/3 Inputs

This directory is for the **real long-form measured files** that feed:

- `scripts/run_exp2.py --input-file ...`
- `scripts/run_exp3.py --input-file ...`

These are **input files**, not output artifacts.

If you want the current repo harness to generate, validate, analyze, and export everything
automatically, do **not** use this directory first. Run `./scripts/run_exp23_pipeline.py --with-ci`
instead. Use this directory only when you are recording metrics from a real external system run.

## What To Put Here

Recommended filenames:

- `exp2_measured.csv`
- `exp3_context_pollution.csv`
- `exp3_tool_invocation_error.csv`

Each file should contain one row per:

- Exp 2: `policy x workflow x failure_mode x run`
- Exp 3: `policy x depth x injection_stage x run`

## Fastest Workflow

1. Generate blank skeleton files:

```bash
python3 scripts/scaffold_measured_inputs.py --output-dir data/measured
```

2. Copy the scaffold to a working measured filename:

```bash
cp data/measured/exp2_measured_skeleton.csv data/measured/exp2_measured.csv
cp data/measured/exp3_context_pollution_skeleton.csv data/measured/exp3_context_pollution.csv
```

3. Fill in the blank metric columns from your real system runs:

- `injected_task_success`
- `final_task_success`
- `cascade_radius`
- `recovery_completeness`
- `time_to_detection_ms`
- `escalated`
- `escalation_latency_ms`

4. Validate before analysis:

```bash
python3 scripts/validate_measured_input.py --experiment 2 --input-file data/measured/exp2_measured.csv --strict
python3 scripts/validate_measured_input.py --experiment 3 --input-file data/measured/exp3_context_pollution.csv --strict
```

5. Run the analyzers:

```bash
python3 scripts/run_exp2.py --input-file data/measured/exp2_measured.csv --with-ci
python3 scripts/run_exp3.py --input-file data/measured/exp3_context_pollution.csv --with-ci
```

## Important Rules

- Keep `scenario_id` identical across policies for the same underlying scenario.
- Exp 3 accepts exactly **one `failure_mode` per file**.
- `injection_stage` is zero-based.
- If a row is still blank, validation should fail; do not analyze partial files.

## Suggested First Pass

Use a small pilot first:

```bash
python3 scripts/scaffold_measured_inputs.py --output-dir data/measured --exp2-runs 5 --exp3-runs 5
```

After the pilot pipeline is clean, regenerate with larger counts.
