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
- `scripts/run_demo.py` — rich terminal demo comparing built-in policies
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
python3 -m pytest -q
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
