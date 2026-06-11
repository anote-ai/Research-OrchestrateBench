# Research Design Document: OrchestrateBench

## Vision Statement

Establish **OrchestrateBench**: the first benchmark for evaluating AI agent orchestration systems on *failure cascades, topology resilience, and human-in-the-loop placement* — providing the community with tools to design multi-agent systems that degrade gracefully rather than catastrophically, and directly informing the architecture of production AI orchestration at Anote and across the industry.

---

## Problem Statement & Novelty

Multi-agent AI systems (tool-use chains, agent pipelines, LLM orchestration frameworks) are being deployed in production, but the community lacks:

1. **Cascade failure characterization**: When one agent fails, how does failure propagate through different topologies (serial, parallel, hierarchical, mesh)?
2. **Failure taxonomy**: What are the distinct failure modes in agent orchestration, and which topologies are vulnerable to which modes?
3. **Human checkpoint optimization**: Where should humans be placed in an agent pipeline to maximize error interception with minimum interruptions?
4. **Recovery protocol evaluation**: How do different recovery strategies (retry, fallback, replan, abort) compare across failure types?

Existing benchmarks (GAIA, AgentBench, WebArena) evaluate task completion but do not study failure propagation or orchestration resilience.

### Novel Contributions

| Contribution | Description |
|---|---|
| **OrchestrateBench dataset** | 400 multi-agent tasks across 4 topologies, with injected fault scenarios |
| **CFR metric** | Cascade Failure Rate: fraction of downstream agents that fail given one upstream failure |
| **HCP algorithm** | Human Checkpoint Placement: optimal human review placement given CFR and task criticality |
| **Recovery taxonomy** | 4-category failure taxonomy with associated recovery protocol recommendations |
| **Topology resilience index** | Composite resilience score for comparing orchestration architectures |

### CFR Definition

```
CFR(topology, failure_point) = |{downstream agents that fail}| / |{total downstream agents}|

Cascade Failure Rate for topology T:
  CFR(T) = E[CFR(T, f)] over all failure points f

Topology Resilience Index:
  TRI(T) = 1 - CFR(T) × severity_weight
```

---

## Research Objectives

1. Measure **CFR across 4 topologies**: serial, parallel, hierarchical, mesh — and show that topology choice has ≥30 pp impact on CFR.
2. Develop a **failure taxonomy** that covers ≥90% of observed failures and has actionable recovery protocol mapping.
3. Validate the **HCP algorithm**: show it reduces task failure rate by ≥20% with ≤50% human interruption overhead vs. naive human oversight.
4. Compare **recovery strategies** across failure types and identify which strategy is optimal for each failure category.
5. Quantify the **orchestration overhead cost**: latency and token cost of adding monitoring, checkpoints, and recovery logic.

---

## Dataset Construction

### Task Categories (400 tasks total)

| Category | Count | Topology | Description |
|---|---|---|---|
| Document processing pipeline | 100 | Serial | Extract → Transform → Validate → Store |
| Parallel research synthesis | 100 | Parallel | Multiple agents gather evidence, merge |
| Hierarchical planning | 100 | Hierarchical | Planner → sub-planners → executors |
| Collaborative coding | 100 | Mesh | Multiple agents code, test, review together |

### Fault Injection Protocol

```yaml
Fault types injected per task:
  - agent_timeout: agent fails to respond within deadline
  - hallucination_propagation: agent produces confident incorrect output
  - context_overflow: agent's context window exceeded
  - tool_failure: external tool call returns error
  - coordination_deadlock: two agents wait on each other
  - specification_drift: agent interprets task differently than intended

Injection points: first agent, middle agent, critical path agent
Injection rate: 1 fault per task (controlled), 2-3 faults per task (stress test)
```

### Human Annotation
For each task + fault combination:
- Human expert labels: fault type, severity, recovery action taken, whether cascade occurred
- Two annotators, adjudication for disagreements (κ target > 0.75)

---

## Systems Under Evaluation

| Framework | Architecture | Recovery Support | Notes |
|---|---|---|---|
| LangGraph | Graph-based | Checkpoint + retry | Open-source |
| AutoGen | Multi-agent conversation | Human feedback | Microsoft |
| CrewAI | Role-based | Task delegation | Popular |
| LlamaIndex agents | Tool-use | Error handling | Our stack |
| OpenAI Assistants | Tool-use | Retry | API-based |
| Custom OrchestrateBench baseline | All topologies | HCP algorithm | Our proposed |

---

## Experimental Design

### Baseline Experiment (Experiment 0)
**Protocol**: Run LangGraph on 100 serial tasks with no fault injection. Measure task success rate, latency, cost.

**Expected result**: 89% success rate, 45s avg latency. Establishes normal operating performance as baseline for degradation measurement.

---

### Experiment 1: Cascade Failure Rate by Topology
**Hypothesis**: Serial topology has CFR > 0.70 for agent_timeout faults (failure at step k cascades to all steps k+1...n); parallel topology has CFR < 0.25 for the same fault.

**Protocol**:
1. Inject each fault type at each injection point across all 4 topologies.
2. Record which downstream agents fail (human labeled).
3. Compute CFR(topology, fault_type) for all combinations.
4. Statistical test: ANOVA across topologies for each fault type.

**Expected results**:

| Topology | agent_timeout CFR | hallucination_prop CFR | deadlock CFR |
|---|---|---|---|
| Serial | 0.74 | 0.82 | N/A |
| Parallel | 0.21 | 0.38 | 0.09 |
| Hierarchical | 0.45 | 0.61 | 0.31 |
| Mesh | 0.33 | 0.55 | 0.48 |

- Key finding: hallucination_propagation has the highest CFR across all topologies — it is the most dangerous failure mode because downstream agents accept the confident incorrect output.
- Serial topology CFR gap vs. parallel: 53 pp for timeout faults.

---

### Experiment 2: Failure Taxonomy
**Hypothesis**: A 4-category taxonomy covers ≥90% of observed failures, and each category maps to a distinct optimal recovery protocol.

**Protocol**:
1. Sample 500 failure events from fault injection experiments.
2. Two annotators independently apply taxonomy.
3. Compute inter-rater agreement (κ), iterate on taxonomy until κ > 0.75.
4. Validate: does taxonomy category predict optimal recovery protocol?

**Taxonomy**:
```
Category 1: State corruption failures (hallucination, wrong output format)
  → Recovery: validate + replan with correct state
Category 2: Resource failures (timeout, context overflow, tool failure)
  → Recovery: retry with backoff; fallback to simpler agent
Category 3: Coordination failures (deadlock, circular dependency)
  → Recovery: break deadlock with forced priority ordering
Category 4: Specification failures (drift, ambiguity, misinterpretation)
  → Recovery: human clarification checkpoint; escalate to orchestrator
```

**Expected results**:
- Taxonomy covers 93% of observed failures
- Inter-rater agreement: κ = 0.79
- Category-recovery mapping validated: correct recovery protocol reduces additional failures by 62% vs. generic retry

---

### Experiment 3: Human Checkpoint Placement
**Hypothesis**: HCP algorithm reduces task failure rate by ≥20 pp vs. no checkpoints while requiring ≤50% of the human interruptions of "review every step" policies.

**Protocol**:
1. Implement HCP: place human checkpoints at agents where (CFR × task_criticality) exceeds threshold.
2. Compare against: (a) no checkpoints, (b) checkpoint every agent, (c) checkpoint first/last only, (d) HCP.
3. Measure: task success rate, human interruptions per task, total latency.

**Expected results**:

| Policy | Success Rate | Human Interruptions/task | Latency |
|---|---|---|---|
| No checkpoints | 0.61 | 0.0 | 45s |
| Every agent | 0.94 | 4.2 | 180s |
| First + last | 0.71 | 0.4 | 60s |
| HCP (ours) | 0.87 | 1.1 | 72s |

- HCP achieves 87% success with 1.1 interruptions/task vs. 0.61/0.0 (no checkpoints) and 0.94/4.2 (every agent)
- HCP efficiency: 26 pp success improvement at 74% fewer interruptions than full review

```python
# HCP algorithm sketch
def place_checkpoints(topology, cfr_matrix, task_criticality, interruption_budget):
    risk_scores = {agent: cfr_matrix[agent] * task_criticality for agent in topology.agents}
    # Greedy: place checkpoints at highest-risk agents within budget
    sorted_agents = sorted(risk_scores, key=risk_scores.get, reverse=True)
    return sorted_agents[:interruption_budget]
```

---

### Experiment 4: Recovery Strategy Comparison
**Hypothesis**: Recovery strategy match to failure taxonomy category reduces secondary failures by ≥40% vs. generic retry-all strategy.

**Protocol**:
1. For each failure event in Experiment 1, apply: (a) generic retry, (b) taxonomy-matched recovery.
2. Measure: secondary failure rate (does recovery itself fail?), recovery time, final task success.

**Expected results**:
- Generic retry secondary failure rate: 34%
- Taxonomy-matched recovery secondary failure rate: 19% (-44%)
- Recovery time: taxonomy-matched is 2.1× faster (avoids futile retries)
- Final task success after recovery: 78% (taxonomy-matched) vs. 61% (generic retry)

---

## Expected Results Summary

| Metric | Best Case | Worst Case | Key Finding |
|---|---|---|---|
| CFR by topology | 0.21 (parallel, timeout) | 0.82 (serial, hallucination) | Topology choice ×4 impact on CFR |
| HCP success rate | 0.87 | 0.61 (no checkpoints) | +26 pp with 1.1 interruptions |
| Recovery match improvement | -44% secondary failures | — | Taxonomy-matched recovery critical |
| Taxonomy coverage | 93% of failures | — | 4 categories sufficient |

**Primary claim**: Topology choice and human checkpoint placement are the two highest-leverage decisions in multi-agent system design; CFR and HCP provide quantitative tools for making these decisions.

---

## Why This Matters

**For researchers**: OrchestrateBench provides the first systematic evaluation of multi-agent failure dynamics — essential as orchestration frameworks proliferate.

**For practitioners**: CFR, HCP, and the failure taxonomy give engineering teams concrete tools for designing resilient agent systems.

**For Anote products**: Any multi-agent product (orchestrated pipelines, multi-step AI workflows) can use OrchestrateBench results to make topology and checkpoint decisions.

**RSI connection**: A resilient orchestration framework is a prerequisite for recursive self-improvement — agents that can recover from their own failures are more capable of iterative self-enhancement.

---

## Implementation Plan

```
research-orchestratebench/
├── data/
│   ├── tasks/           # 400 multi-agent tasks
│   ├── fault_specs/     # Fault injection specifications
│   └── human_labels/   # Failure type + severity labels
├── orchestration/
│   ├── topologies/      # Serial, parallel, hierarchical, mesh
│   ├── fault_injector.py
│   └── recovery/        # Recovery protocol implementations
├── metrics/
│   ├── cfr.py
│   ├── tri.py
│   └── hcp.py
├── experiments/
│   ├── exp0_baseline.py
│   ├── exp1_cfr.py
│   ├── exp2_taxonomy.py
│   ├── exp3_hcp.py
│   └── exp4_recovery.py
```

---

## Timeline

| Phase | Duration | Deliverable |
|---|---|---|
| Task construction | 5 weeks | 400 multi-agent tasks |
| Fault injection framework | 3 weeks | Automated fault injection |
| Human annotation | 4 weeks | Failure taxonomy labels |
| Experiments | 5 weeks | All results |
| Paper writing | 4 weeks | NeurIPS 2026 submission |

**Target venue**: NeurIPS 2026

---

## Open Questions & Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Non-determinism in LLM agents | High | Multiple runs (n=5); report mean + CI |
| Fault injection realism | Medium | Human expert validation of fault scenarios |
| Framework API changes | Medium | Pin versions; test on 2 LangGraph versions |
| HCP optimality proof | Low | Provide empirical validation only |

---

## Related Issues

- RSI connection: resilient orchestration as RSI primitive
- Product integration: multi-agent Anote workflows
- Related work audit: GAIA, AgentBench, WebArena
- Reproducibility: non-determinism handling
