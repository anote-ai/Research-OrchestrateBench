# OrchestateBench

> **When should an LLM orchestrator decompose a task vs. call a tool directly?**

OrchestateBench provides a rigorous framework for evaluating and comparing
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

Policies evaluated:
- **FixedPolicy** — always uses DIRECT_TOOL (baseline)
- **HeuristicPolicy** — rule-based routing on complexity, code, retrieval flags
- *(Extendable to learned/LLM-based policies)*

## Policy Comparison

Run `python scripts/run_demo.py` to see a comparison table like:

| Policy | Success Rate | Mean Latency | Mean Cost |
|--------|-------------|-------------|----------|
| FixedPolicy | 1.000 | 1050 ms | $0.025 |
| HeuristicPolicy | 1.000 | 1100 ms | $0.026 |

## Quickstart

```bash
pip install -e .
python scripts/run_demo.py
pytest tests/ -v
```

## Venues

- **DAI 2026** — Distributed AI workshop
- **EMNLP ORACLE Workshop** — Open Reasoning and Agent Coordination for LLMs
- **AAAI 2027** — Main track, Multi-Agent Systems

## Citation

```bibtex
@misc{orchestratebench2026,
  title   = {OrchestateBench: Evaluating LLM Orchestration Routing Policies},
  author  = {Anote AI},
  year    = {2026},
  url     = {https://github.com/anote-ai/research-orchestratebench}
}
```
