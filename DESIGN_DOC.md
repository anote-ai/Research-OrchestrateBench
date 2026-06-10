# OrchestraBench — Research Design Document

## Goal

Build the first benchmark that evaluates multi-agent AI systems on cascade failure rates, decomposition quality, and framework-specific failure patterns — providing both a taxonomy of how multi-agent pipelines fail and actionable metrics for comparing orchestration frameworks.

## Objective

1. Construct 100+ multi-agent scenarios across 4 orchestration topologies (linear chain, branching, hierarchical, fully-connected mesh)
2. Evaluate AutoGen, LangGraph, CrewAI, and OpenAI Assistants on cascade failure rate and task success rate
3. Publish a failure taxonomy covering: communication failures, tool misuse, context loss, coordination deadlock, and cascade amplification

## Background / Motivation

Multi-agent AI frameworks shipped rapidly in 2023–2024. AutoGen, LangGraph, and CrewAI each claim to solve multi-agent coordination, but none have been evaluated on failure modes specifically. Practitioners report that multi-agent pipelines fail in mysterious ways — an error in one agent propagates and amplifies. No public benchmark measures cascade failure rates.

## Experimental Design

### Baseline Experiment

**Evaluate task success rate for GPT-4o-based AutoGen on 20 linear-chain scenarios (A → B → C → D)**

- Metric: end-to-end task success rate; per-agent success rate
- Purpose: confirm that individual agent success doesn't predict pipeline success
- Expected result: individual agent success ~80%; end-to-end pipeline success ~40% (errors multiply: 0.8^4 ≈ 0.41)

### Test Experiment 1: Cascade Failure Rate by Topology

For 4 orchestration topologies, measure: cascade failure rate = (pipelines where single agent failure caused full pipeline failure) / (pipelines where at least one agent failed). Evaluate AutoGen, LangGraph, CrewAI.

**Expected result:** cascade failure rates differ significantly by topology — mesh topologies have 3x higher cascade failure rates than linear chains

### Test Experiment 2: Failure Taxonomy Construction

Collect 500+ failure examples. Manually label each with a failure type. Derive taxonomy from bottom-up clustering. Measure inter-rater agreement.

Hypothesized categories: context window overflow, tool hallucination, coordination deadlock, cascade amplification, evaluation capture.

**Expected result:** cascade amplification and context window overflow together account for >60% of pipeline failures

### Test Experiment 3: Human Oversight Placement

Test where human checkpoints reduce cascade failure rate most efficiently: no checkpoints, every agent, highest-risk stages only, after irreversible actions only. Find Pareto-optimal checkpoint placement.

**Expected result:** strategic checkpoints at 2 highest-risk transition points reduce cascade failure by 60% while requiring only 15% of the reviews needed for full checkpointing

## Expected Results

1. A benchmark of 100+ multi-agent scenarios with full failure logging
2. A cascade failure taxonomy with inter-rater agreement measurement
3. **Key finding:** "A single agent failure cascades into full pipeline failure 40–70% of the time depending on topology"
4. Human checkpoint placement recommendation
5. Framework comparison: AutoGen vs. LangGraph vs. CrewAI on cascade failure rate

## Why This Matters / Why People Would Care

- **Multi-agent framework builders:** first head-to-head comparison on failure rates — will shape how frameworks prioritize error handling
- **Enterprise AI teams:** deploying multi-agent pipelines for consequential tasks; cascade failure rates are critical for production safety planning
- **AI safety researchers:** multi-agent cascade failures are a near-term safety concern; this provides the measurement foundation
- **Orchestration ecosystem:** the failure taxonomy will become the vocabulary practitioners use to debug multi-agent failures

## Timeline

| Month | Milestone |
|---|---|
| 1–2 | Scenario construction (100 scenarios × 4 topologies); framework adapter implementation |
| 3 | Baseline experiments + failure taxonomy construction |
| 4 | Cascade failure rate experiments + human checkpoint placement study |
| 5 | Analysis + framework comparison |
| 6 | Submission to NeurIPS 2026 |

## Related Issues

- Design doc GitHub issue: #24
- RSI synthesis issue: #22
- Target conferences: see issues labeled `conference-prep`
- Reproducibility package: see issues labeled `artifact-release`
