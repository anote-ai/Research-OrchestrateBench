# OrchestrateBench

> **When should an LLM orchestrator decompose a task vs. call a tool directly?**

OrchestrateBench provides a compact, extensible framework for evaluating and comparing
orchestration routing policies for multi-agent LLM systems.

## The Orchestration Problem

Modern LLM pipelines must decide — for each incoming task — whether to:

| Decision | When | Sub-agent |
|----------|------|-----------|
| `DECOMPOSE` | High complexity, multi-step | PLANNING |
| `DIRECT_TOOL` | Retrieval or API call needed | RETRIEVAL / TOOL_CALL |
| `CODE_EXECUTION` | Computation required | CODE |
| `REASON_ONLY` | Simple QA | — |

## Routing Taxonomy

```
Task → Orchestrator → RoutingDecision → SubAgent → Result
```

Built-in policies:
- **FixedPolicy** — always uses DIRECT_TOOL (baseline)
- **HeuristicPolicy** — rule-based routing on complexity, code, retrieval flags
- *(Extendable to learned/LLM-based policies)*

## Repo Layout

- `src/orchestratebench/core.py` — task schema, policies, retry logic, dependency-aware execution
- `src/orchestratebench/data.py` — sample tasks and workflow generators
- `src/orchestratebench/evaluate.py` — success, latency, cost, throughput, and dependency metrics
- `src/orchestratebench/failures.py` — failure injection, recovery, detection, and cascade measurement
- `src/orchestratebench/experiments.py` — Experiment 2/3 runners, long-form records, artifact export, and reporting helpers
- `scripts/run_demo.py` — rich terminal demo comparing built-in policies
- `scripts/reproduce_exp1.py` — one-command reproduction of the measured Exp 1 routing diagnostic
- `scripts/run_exp2.py` — offline Exp 2 harness demo with retry-aware recovery diagnostics + CSV/JSON artifacts
- `scripts/run_exp3.py` — offline Exp 3 harness demo with cascade depth/stage sweep + CSV/JSON artifacts
- `scripts/validate_measured_input.py` — preflight validation for collaborative measured Exp 2/3 inputs
- `scripts/export_paper_tables.py` — convert Exp 2/3 summary artifacts into paper-friendly Markdown + LaTeX tables
- `examples/` — minimal measured-input templates for Exp 2 and Exp 3
- `tests/` — unit tests covering routing, dependencies, retry handling, and metrics

## Policy Comparison

Run `python scripts/run_demo.py` to see a comparison table like:

| Policy | Success Rate | Mean Latency | Mean Cost | Efficiency |
|--------|-------------|-------------|----------|------------|
| FixedPolicy | 1.000 | 675 ms | $0.011 | 1.000 |
| HeuristicPolicy | 1.000 | 1520 ms | $0.027 | 0.650 |

## Quickstart

```bash
python3 -m pip install -e ".[dev]"
python3 scripts/run_demo.py
python3 scripts/reproduce_exp1.py
python3 scripts/run_exp2.py
python3 scripts/run_exp3.py
python3 scripts/validate_measured_input.py --experiment 2 --input-file examples/exp2_measured_template.csv
python3 -m pytest -q
```

Useful options for the experiment scripts:

```bash
python3 scripts/run_exp2.py --n-runs 100 --with-ci
python3 scripts/run_exp3.py --n-runs 100 --depths 3,5,7 --injection-stages 0,1,2 --with-ci
python3 scripts/run_exp2.py --input-file path/to/exp2_measured.csv --with-ci
python3 scripts/run_exp3.py --input-file path/to/exp3_measured.jsonl --with-ci
python3 scripts/validate_measured_input.py --experiment 3 --input-file path/to/exp3_measured.jsonl --strict
python3 scripts/export_paper_tables.py --experiment 2 --summary-json artifacts/exp2/summary.json
```

## Research Status

- **Experiment 1**: measured routing diagnostic is in-repo and reproducible via
  `scripts/reproduce_exp1.py`. The offline baselines run without network; the LLM row runs when
  `ANTHROPIC_API_KEY` is set.
- **Experiments 2 and 3**: the harness is scaffolded and now supports retry-wrapped policies plus
  richer diagnostics including injected-stage recovery rate, final-task success rate, cascade
  radius, time-to-detection, and escalation latency.
- **Measured-input pipeline**: Exp 2/3 scripts now accept collaborative long-form measured inputs
  via `--input-file` (`.csv`, `.jsonl`, or `.json`) and can validate them up front with
  `scripts/validate_measured_input.py`.
- **Artifact outputs**: `run_exp2.py` now writes raw runs plus mode/workflow summaries to
  `artifacts/exp2/`; `run_exp3.py` writes raw runs plus depth/stage sweep summaries to
  `artifacts/exp3/`.
- **Policy comparison outputs**: both scripts now also export paired bootstrap policy comparisons
  and a human-readable `report.txt` alongside the CSV/JSON artifacts.
- **Paper-ready exports**: both scripts now also write `paper_summary.md` and `paper_tables.tex`,
  so measured runs can flow straight into the design doc / draft paper with minimal manual cleanup.
- **Important scope note**: Exp 2/3 default script runs still use simulated harness traces for
  mechanism validation. Those default numbers are useful for development and paper plumbing, but
  they are not the final measured numbers to cite in the paper. To analyze real collaborative runs,
  pass a measured long-form file through `--input-file`.

## Measured Input Contract

Exp 2 and Exp 3 now share a strict long-form input schema for collaborative measured runs. Accepted
formats are `.csv`, `.jsonl`, and `.json` (either a top-level list of records or
`{"records": [...]}`).

Required fields for both Exp 2 and Exp 3:
- `policy`
- `failure_mode`
- `run`
- `injected_task_success`
- `final_task_success`
- `cascade_radius`
- `recovery_completeness`
- `time_to_detection_ms`
- `escalated`
- `escalation_latency_ms`

Exp 2 adds:
- `workflow`
- `injection_stage`

Exp 3 adds:
- `depth`
- `injection_stage`

Exp 3 file-level constraint:
- each input file must contain exactly one `failure_mode`, matching the single-mode cascade sweep
  that `scripts/run_exp3.py` analyzes

Recommended fields:
- `scenario_id`: strongly recommended for cross-policy pairing. If every row has it, paired
  bootstrap comparisons use `scenario_id`; otherwise the scripts fall back to run-based keys.
- `seed`: optional provenance for reproducibility.

Additional columns are allowed and preserved in the loaded records, so annotation metadata like
`annotator`, `notes`, or dataset-specific tags can travel with the file.

Minimal templates live at:
- `examples/exp2_measured_template.csv`
- `examples/exp3_measured_template.jsonl`

Recommended workflow for real runs:

```bash
python3 scripts/validate_measured_input.py --experiment 2 --input-file path/to/exp2_measured.csv --strict
python3 scripts/run_exp2.py --input-file path/to/exp2_measured.csv --with-ci

python3 scripts/validate_measured_input.py --experiment 3 --input-file path/to/exp3_measured.jsonl --strict
python3 scripts/run_exp3.py --input-file path/to/exp3_measured.jsonl --with-ci
```

The validator checks schema compatibility plus pairwise-analysis hazards such as duplicate
policy/scenario rows, missing policy coverage inside a comparison group, and zero-overlap pairings.

If you already have a finished `summary.json` artifact and want to regenerate only the paper-facing
exports, run:

```bash
python3 scripts/export_paper_tables.py --experiment 2 --summary-json artifacts/exp2/summary.json
python3 scripts/export_paper_tables.py --experiment 3 --summary-json artifacts/exp3/summary.json
```

If you prefer not to install the package in editable mode first, the test suite
also supports running directly from the repo root.

## What This Repo Models

- Independent benchmark tasks such as retrieval, code execution, and lightweight reasoning
- Multi-step enterprise workflows with explicit dependencies
- Policy behavior under retries, skipped tasks, and dependency failures
- Aggregate evaluation metrics for latency, cost, throughput, and orchestration quality

## Venues

- **DAI 2026** — Distributed AI workshop
- **EMNLP ORACLE Workshop** — Open Reasoning and Agent Coordination for LLMs
- **AAAI 2027** — Main track, Multi-Agent Systems

## Citation

```bibtex
@misc{orchestratebench2026,
  title   = {OrchestrateBench: Evaluating LLM Orchestration Routing Policies},
  author  = {Anote AI},
  year    = {2026},
  url     = {https://github.com/anote-ai/research-orchestratebench}
}
```
