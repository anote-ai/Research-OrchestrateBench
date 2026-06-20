# OrchestrateBench

> **When should an LLM orchestrator decompose a task, call a tool, execute code, or simply reason?**

OrchestrateBench is a research benchmark for evaluating the orchestration
meta-decision layer of multi-agent LLM systems: routing, failure recovery, and
cascade propagation. The repository is designed to support both mechanism-level
benchmarking inside the repo harness and paper-oriented analysis workflows that
export Markdown and LaTeX artifacts directly.

## What This Repository Contributes

OrchestrateBench focuses on four research questions:

- **Routing value**: when does intelligent routing beat a zero-decision baseline?
- **Routing mechanism**: do failures come from task difficulty or from the
  routing mechanism itself?
- **Failure attribution**: can multi-agent failures be injected and attributed
  reproducibly by mode and stage?
- **Cascade containment**: how far does one seeded error propagate, and which
  recovery policies actually contain it?

The current repository includes:

- a routing benchmark with `fixed`, `heuristic`, retry-wrapped, and LLM-based
  policy support
- controlled failure injection derived from the MAST-style taxonomy
- Experiment 2 and 3 analyzers with bootstrap summaries and pairwise policy
  comparisons
- one-command export of paper-facing `paper_summary.md` and `paper_tables.tex`

## Current Status

| Experiment | Current status | Best entrypoint |
|---|---|---|
| Exp 1: routing diagnostic | Measured and reproducible in-repo | `python3 scripts/reproduce_exp1.py` |
| Exp 2: failure recovery | Auto-harness pipeline available; collaborative measured input path supported | `./scripts/run_exp23_pipeline.py --with-ci --exp3-modes context_pollution,tool_invocation_error` |
| Exp 3: cascade propagation | Auto-harness pipeline available; collaborative measured input path supported | `./scripts/run_exp23_pipeline.py --with-ci --exp3-modes context_pollution,tool_invocation_error` |
| Exp 4: decomposition quality | Planned | n/a |

Useful repository documents:

- [DESIGN_DOC.md](DESIGN_DOC.md)
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
- [data/measured/README.md](data/measured/README.md)

## Current In-Repo Findings

These are the current repo-verified findings, not a claim that every paper-ready
external measurement has been finalized:

- **Exp 1 measured routing diagnostic**: `FixedPolicy` reaches 23% overall,
  `HeuristicPolicy` reaches 62% overall but 0% on adversarial cases, and the
  current `LLMPolicy` reproduces 100% on the focused gold set.
- **Exp 2 auto-harness**: `retry(heuristic)` helps on `tool_invocation_error`
  but not on latent semantic failures such as `context_pollution` or
  `ambiguous_delegation`.
- **Exp 3 auto-harness**: earlier-stage injected context corruption produces
  larger cascade radius, and deeper pipelines amplify the effect; retry fully
  contains `tool_invocation_error` but does not generically stop latent
  corruption.

## Result Provenance Matters

This repository intentionally supports more than one kind of result:

| Result type | Meaning |
|---|---|
| **Measured diagnostic** | Human-curated or externally measured benchmark evidence |
| **Auto-harness mechanism result** | Repo-generated benchmark evidence used to validate the mechanism and analysis pipeline |
| **Collaborative measured input** | Externally collected long-form input analyzed by the same scripts |

For Exp 2 and Exp 3, the default scripts and the one-command pipeline generate
**auto-harness mechanism results**. These are useful, publishable as mechanism
evidence, and fully reproducible, but they should not be confused with the
final externally collected measured numbers unless the input came from a real
measured file passed through `--input-file`.

## Quickstart

Install and run the core checks:

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest -q
```

Fastest entrypoints by task:

| Goal | Command |
|---|---|
| Demo the routing policies | `python3 scripts/run_demo.py` |
| Reproduce Exp 1 measured diagnostic | `python3 scripts/reproduce_exp1.py` |
| Run Exp 2/3 auto-harness pipeline | `./scripts/run_exp23_pipeline.py --with-ci --exp3-modes context_pollution,tool_invocation_error` |
| Scaffold real measured input files | `python3 scripts/scaffold_measured_inputs.py --output-dir data/measured` |
| Validate a collaborative measured file | `python3 scripts/validate_measured_input.py --experiment 2 --input-file path/to/exp2_measured.csv --strict` |

## Reproducing Experiments

### Experiment 1

```bash
python3 scripts/reproduce_exp1.py
```

Notes:

- Offline baseline rows run without network.
- The LLM row requires `ANTHROPIC_API_KEY`.

### Experiment 2 / 3 auto-harness path

```bash
./scripts/run_exp23_pipeline.py --with-ci --exp3-modes context_pollution,tool_invocation_error
```

This command:

- generates measured-style long-form inputs
- validates those inputs
- runs Exp 2 and Exp 3 analyses
- exports raw runs, summaries, pairwise comparisons, Markdown summaries, LaTeX
  tables, and a run manifest

Main outputs:

- `artifacts/exp23_pipeline/measured_inputs/`
- `artifacts/exp23_pipeline/analysis/exp2/`
- `artifacts/exp23_pipeline/analysis/exp3/<failure_mode>/`
- `artifacts/exp23_pipeline/pipeline_manifest.json`

### Collaborative measured-data path

If you are analyzing a real external run rather than the built-in harness:

```bash
python3 scripts/scaffold_measured_inputs.py --output-dir data/measured
python3 scripts/validate_measured_input.py --experiment 2 --input-file path/to/exp2_measured.csv --strict
python3 scripts/run_exp2.py --input-file path/to/exp2_measured.csv --with-ci
python3 scripts/validate_measured_input.py --experiment 3 --input-file path/to/exp3_measured.jsonl --strict
python3 scripts/run_exp3.py --input-file path/to/exp3_measured.jsonl --with-ci
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
for the full provenance rules and artifact semantics.

## Repository Layout

- `src/orchestratebench/core.py` — task schema, routing policies, retry logic,
  and dependency-aware execution
- `src/orchestratebench/data.py` — benchmark tasks and workflow generators
- `src/orchestratebench/evaluate.py` — success, latency, cost, throughput, and
  dependency metrics
- `src/orchestratebench/failures.py` — failure injection, recovery, detection,
  and cascade measurement
- `src/orchestratebench/experiments.py` — Experiment 2/3 runners, long-form
  records, artifact export, and summary helpers
- `src/orchestratebench/paper_reports.py` — Markdown and LaTeX export helpers
- `scripts/` — one-command experiment entrypoints and validation utilities
- `examples/` — minimal measured-input templates
- `tests/` — unit tests for routing, failure injection, measured input loading,
  statistics, and reporting

## Development and CI

Continuous integration runs on push and pull request to `main` for Python 3.10,
3.11, and 3.12.

Local developer check:

```bash
python3 -m pytest -q
```

## Research Scope Notes

- This repo currently models enterprise-style workflow orchestration with
  explicit dependencies and seeded failure injection.
- Exp 2 and Exp 3 support both automatic harness runs and externally supplied
  measured long-form records.
- The paper-facing exports are generated from the same analysis artifacts as the
  machine-readable summaries, reducing hand-transcription risk.

## Citation

GitHub citation metadata is provided in
[CITATION.cff](CITATION.cff).
If you need a BibTeX entry, the repository currently recommends:

```bibtex
@misc{orchestratebench2026,
  title   = {OrchestrateBench},
  author  = {Yidian Chen and Yingzi Gu},
  year    = {2026},
  url     = {https://github.com/anote-ai/Research-OrchestrateBench},
  note    = {Research benchmark for LLM orchestration routing, failure recovery, and cascade propagation}
}
```
