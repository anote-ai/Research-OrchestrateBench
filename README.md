# Orchestrate-Bench

> Learning to Choose Tasks, Tools & Code Execution in Multi-Agent Systems

Orchestrate-Bench evaluates the **meta-decision layer** of agentic systems: the orchestrator that decides *when* to decompose a task, *which* sub-agent or tool to invoke, and *when* to execute code vs. reason in-context.

---

## The Orchestration Problem

Modern agentic pipelines must answer three core questions at every step:

1. **Decompose or direct?** Should this task be broken into sub-tasks routed to specialised agents, or handled by a single direct tool call?
2. **Which tool/agent?** Given a decomposed sub-task, which sub-agent type (retrieval, code, planning, summarisation, tool call) minimises cost and latency while maximising success?
3. **Execute or reason?** Should the agent run code in a sandbox, or reason through the problem using in-context chain-of-thought?

Orchestrate-Bench measures how well an orchestration *policy* answers these questions relative to optimal fixed-policy baselines.

---

## Routing Decision Taxonomy

| Decision | When to Use | Typical Latency | Typical Cost |
|----------|-------------|-----------------|-------------|
| DECOMPOSE | Complex multi-step tasks | High | High |
| DIRECT_TOOL | Well-defined single-step tasks | Low | Low |
| CODE_EXECUTION | Numerical / data transformation tasks | Medium | Medium |
| REASON_ONLY | Simple factual or logical questions | Low | Low |

---

## Baseline Policies

| Policy | Strategy | pass@1 |
|--------|----------|--------|
| FixedPolicy | Always DIRECT_TOOL | 0.51 |
| RandomPolicy | Uniform random over decisions | 0.38 |
| OraclePolicy | Always optimal decision | 1.00 |

---

## Benchmark Metrics

- **success_rate** — Fraction of tasks completed successfully
- **mean_latency** — Average total latency in milliseconds
- **mean_cost** — Average total cost in USD
- **routing_accuracy** — Fraction of routing decisions matching oracle labels
- **policy_comparison** — Side-by-side aggregated metrics across multiple policies

---

## Quickstart

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

```python
from orchestratebench.core import AgentTask, FixedPolicy, OrchestratorBench
from orchestratebench.evaluate import success_rate

tasks = [
    AgentTask(task_id="t1", description="Fetch user data", complexity_score=0.3,
              requires_code=False, requires_retrieval=True),
]
bench = OrchestratorBench()
traces = bench.evaluate_policy(FixedPolicy(), tasks)
print(success_rate(traces))
```

---

## Citation

```bibtex
@misc{anote2024orchestratebench,
  title  = {Orchestrate-Bench: Learning to Choose Tasks, Tools & Code Execution in Multi-Agent Systems},
  author = {Anote AI},
  year   = {2024},
  url    = {https://github.com/anote-ai/research-orchestratebench}
}
```
