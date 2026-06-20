# OrchestrateBench — Research Design Document

**Paper #11 (T7 Orchestration)**  
**Authors**: Yidian Chen, Yingzi Gu  
**Supervisor**: Natan Vidra  
**Date**: 2026-06-16  
**Target venues**: DAI 2026 Industry Track (8/3) · EMNLP ORACLE workshop (9/18) · AAAI 2027 (7/28)

> This document supersedes the auto-generated PR #25 draft and aligns with the official #11
> contribution: the **orchestration meta-decision layer** of agentic systems — when to decompose a
> task, which tool/sub-agent to route to, when to execute code vs. reason — benchmarked against
> fixed-policy baselines on cost, latency, success, and routing reliability.

---

## 1. Goal

Create the first standardized benchmark for evaluating **how multi-agent orchestration systems route,
fail, recover, and decompose tasks** — not just whether they succeed.

Current benchmarks (AgentBench, OdysseyBench, SWE-Bench) measure final task success but cannot
diagnose *why* a pipeline failed, *where* a cascade started, or *which routing decision* caused the
breakdown. This gap directly blocks production deployment decisions: teams cannot compare
orchestration frameworks (AutoGen, LangGraph, CrewAI, Anthropic Agents SDK) on reliability, only on
accuracy.

OrchestrateBench fills this gap with controlled failure injection, per-failure-mode attribution,
cascade propagation measurement, and routing-policy comparison.

### Research Questions

- **RQ1 (routing value)**: When does intelligent routing (heuristic or LLM) actually beat a
  zero-decision baseline, once cost is accounted for?
- **RQ2 (mechanism)**: Is routing reliability a property of task difficulty, or of the routing
  mechanism (keyword/flag matching vs. reasoning over intent)?
- **RQ3 (attribution)**: Can a multi-agent pipeline failure be attributed to a specific failure mode
  and pipeline stage, reproducibly across runs?
- **RQ4 (cascade)**: How far does a single seeded error propagate downstream, and which routing /
  recovery strategies contain it?

### Related Work & Differentiation

| Prior work | What it does | How OrchestrateBench differs |
|---|---|---|
| **MAST** — *Why Do Multi-Agent LLM Systems Fail?* (arXiv:2503.13657) | Taxonomy of observed MAS failures (14 modes, 1,600+ traces, κ=0.88) | We **inject** that taxonomy under controlled, seed-reproducible conditions and measure recovery + cascade per mode |
| **MAS-FIRE** (arXiv:2602.19843) | Fault injection + reliability evaluation for multi-agent systems | **Closest external work.** We lead with **routing-policy comparison** and **cascade radius across pipeline depths**, not reliability scoring alone |
| **TraceElephant** (arXiv:2604.22708) | Post-hoc failure attribution in multi-agent systems | Observational; we add controlled seeding, recovery measurement, and cascade radius as primary metrics |
| **Agents Failure Attribution** (ICML 2025 Spotlight) | Post-hoc attribution to the responsible agent/step | Attribution is a means here, not the end product |
| **Doctor-RAG** (arXiv:2604.00865) | Diagnose + repair RAG failures | RAG-specific repair; we target orchestration-level routing/decomposition, not retrieval repair |
| **AdaptOrch** (arXiv:2602.16873) | A task-adaptive orchestration framework | A method; OrchestrateBench is the benchmark that would evaluate it |
| **"From Spark to Fire"** (arXiv:2603.04474) | Shows errors cascade exponentially in pipelines | We operationalize this into a measurable cascade-radius benchmark |

**External novelty risk.** MAS-FIRE already performs controlled fault injection + reliability
evaluation, so our differentiation cannot be "we inject failures." The paper must lead with: (a)
**cascade radius across pipeline depths** as a first-class comparable metric, (b) failure handling
measured **per routing policy** (Fixed / Heuristic / LLM / Oracle), and (c) the **routing-mechanism
result** of Experiment 1, which MAS-FIRE does not address.

**Verified (full-text read of MAS-FIRE v1, 2026-06-17):** MAS-FIRE **does not quantify cascade
depth** — it notes only qualitatively that a corrupted output "propagates downstream unchecked,"
with no stages-traversed metric — **and does not vary routing policy**: its experimental axes are
*topology* (linear / iterative / bilateral; MetaGPT vs. Table-Critic vs. CAMEL), fault type (15
categories), and foundation model, not routing strategy. Both of OrchestrateBench's primary
differentiators — cascade radius across depths and per-routing-policy failure handling — are
therefore *uncovered* by MAS-FIRE, not merely framed differently.

**Internal overlap (flagged by Natan): Paper #2 — *Intent Specification as a First-Class Evaluation
Object* (Susana Haing / Spurthi).** Differentiation: #2 evaluates whether the agent's
*understanding of intent* is correct; #11 evaluates which *orchestration action* to take given a
correct understanding — route / decompose / execute code / reason — and how those decisions cascade.
We measure the **decision policy and its failure propagation**, not intent correctness. This should
be coordinated before submission so the two papers cite-and-complement rather than collide.

### Literature Review

We organize prior work into four strands and position OrchestrateBench against each.

**(1) Agent benchmarks measure final-task success, not orchestration.** AgentBench, SWE-Bench, and
OdysseyBench evaluate whether an agent completes a task but report a single success signal — they
cannot say *which* orchestration decision (route, decompose, code-vs-reason) failed or how an error
spread. *Beyond the Strongest* (arXiv:2509.23537) quantifies the cost of this blind spot: on
GPQA-Diamond at least one agent was correct in 95.5% of cases, yet orchestration delivered only
87.4% — ~8 points of individually-recoverable correctness discarded by the orchestration layer.
OrchestrateBench is built to attribute exactly this loss.

**(2) Failure taxonomies and post-hoc attribution observe failures; they do not intervene.** MAST
(arXiv:2503.13657) catalogs 14 multi-agent failure modes over 1,600+ traces (κ=0.88) but is purely
descriptive. TraceElephant (arXiv:2604.22708) and the ICML 2025 attribution work assign a failure to
the responsible agent/step *after the fact*. OrchestrateBench instead **injects** the MAST taxonomy
under seed control and measures recovery per mode — attribution becomes a means, not the product.

**(3) Fault injection and cascade — the closest strand, and our main novelty risk.** MAS-FIRE
(arXiv:2602.19843) is the nearest external work: it injects faults via prompt modification, response
rewriting, and message-routing manipulation, defines a 15-type fault taxonomy, and grades recovery in
four tiers. Crucially, its comparative axis is **architectural topology** — it finds iterative designs
neutralize more faults than linear workflows — **not routing policy**. *From Spark to Fire*
(arXiv:2603.04474) shows errors cascade super-linearly in pipelines but ships no benchmark.
OrchestrateBench's daylight from MAS-FIRE is therefore specific and defensible: we make **cascade radius
across pipeline depths** a first-class metric and condition failure handling on **routing policy
(Fixed / Heuristic / LLM / Oracle)**, a dimension MAS-FIRE does not vary. (Terminology caveat:
MAS-FIRE's "message-routing manipulation" is a fault-injection *mechanism*; our "routing policy" is
the orchestration decision *under study* — the paper must disambiguate these explicitly.)

**(4) Orchestration methods and frameworks are evaluation targets, not competitors.** AdaptOrch
(arXiv:2602.16873) is a task-adaptive orchestration *method*; the orchestration-pattern benchmark
(arXiv:2603.22651) compares sequential/parallel/hierarchical/reflexive *architectures* on a
cost-accuracy Pareto; Doctor-RAG (arXiv:2604.00865) diagnoses and repairs *retrieval* failures.
OrchestrateBench is complementary: it is the reliability benchmark these methods would be scored on, and
it targets the routing/decomposition decision layer rather than retrieval repair or one architecture
family.

**Synthesis — the gap.** No existing work ships a *controlled, reproducible injection harness that
measures recovery and cascade containment as first-class metrics, compared across routing policies.*
That intersection — injection (vs. observation), cascade radius (vs. reliability score), and
routing-policy conditioning (vs. topology) — is OrchestrateBench's contribution.

---

## 2. Objective

Build and validate a benchmark framework with four concrete capabilities:

1. **Routing-policy comparison with statistical rigor** (issue #16, PR #20/#26): compare Fixed,
   Heuristic, LLM, and Oracle routers on routing accuracy, cost, latency, and success, with bootstrap
   confidence intervals and paired significance testing.
2. **Failure-mode taxonomy benchmark** (issue #4, with Y. Gu): inject high-frequency MAST failure
   modes under seed control; measure per-mode recovery rate, escalation latency, and final task
   accuracy.
3. **Cascade propagation measurement** (issue #7, with Y. Gu): instrument agent-to-agent message
   passing to track how a single seeded error propagates downstream. Primary metric: **cascade
   radius**.
4. **Decomposition quality measurement**: for high-complexity `DECOMPOSE` tasks, compare subtasks
   against gold decompositions and measure delegation fidelity, granularity, and wasted calls.

---

## 3. Experiments

**Datasets & annotation.** Workflow suites are synthetically generated from templated task graphs
(stages, inter-task dependencies, per-task `complexity_score`) and screened by the two co-authors for
realism. Initial workflow families: finance approval (4 stages), HR onboarding (5 stages), and DevOps
deployment (5 stages). Gold routing and decomposition labels are produced independently by both
co-authors with adjudication on disagreement (target inter-annotator agreement Cohen's κ > 0.7).
Failure scenarios for Experiments 2/3 are seeded **deterministically** via the `FailureInjector`
shipped in PR #21, so every reported number is reproducible from a seed.

### Experiment 1 — Routing policy comparison on enterprise workflows *(measured diagnostic + planned suite)*

- **Setup**: Run FixedPolicy, HeuristicPolicy, and an LLM-based policy on three enterprise workflow
  suites. Each suite has task dependencies and varying complexity.
- **Metrics**: success_rate, mean_latency, mean_cost, orchestration_efficiency_score,
  task_dependency_score, routing_accuracy, routing_distribution.
- **Planned scale**: 200 traces per policy × 3 workflows × 3 policies = 1,800 traces.
- **Preliminary implementation & results** (`routing_comparison.py`, PR #26): focused
  routing-accuracy diagnostic over **26 expert-labelled gold cases** spanning all four routing
  decisions — 16 *aligned* (task flags match intent) and 10 *adversarial* (flags missing/misleading,
  but description intent is clear).

| Policy | Overall | Aligned | Adversarial |
|---|---:|---:|---:|
| FixedPolicy | 23% | 25% | 20% |
| HeuristicPolicy (keyword/flags) | 62% | **100%** | **0%** |
| **LLM-as-Router (Claude Sonnet 4.6)** | **100%** | **100%** | **100%** |
| OraclePolicy (ceiling) | 100% | 100% | 100% |

**Finding (RQ2).** HeuristicPolicy scores **100% on aligned but 0% on adversarial** cases, cleanly
quantifying where rule/keyword routing breaks. The model-driven `LLMPolicy` scores **100% overall,
including 100% on adversarial cases**, closing the entire gap to the Oracle ceiling. Because a
cost-efficient model saturates this focused set, the heuristic's adversarial blind spot is shown to
be a property of the routing **mechanism** (keyword/flag matching vs. reasoning over intent), not task
difficulty. The larger 1,800-trace workflow suite is where the LLM router is expected to drop below
100% and yield a graded comparison. Results were stable across 3 passes (78/78); the prompt did not
include gold labels.

### Experiment 2 — Failure injection and recovery *(in progress — core, with Y. Gu / #4)*

- **Setup**: Using the MAST failure taxonomy, inject controlled failures at specific pipeline stages.
  Start with five high-frequency modes: ambiguous delegation, tool invocation error, context
  pollution, conflicting sub-agent outputs, and premature action.
- **Metrics**: per-failure-mode recovery rate, escalation latency, and final task accuracy under
  failure.
- **Scale**: 100 traces per failure mode × 5 modes × 3 policies = 1,500 traces.
- **Status**: scaffolded by PR #21 and now aligned in code with retry-aware offline harness metrics
  (`recovery_rate`, `final_task_success_rate`, `mean_time_to_detection_ms`,
  `mean_escalation_latency_ms`, `mean_cascade_radius`). The runner now also exports long-form raw
  runs, grouped CSV/JSON summaries, and paired bootstrap policy-comparison artifacts for
  workflow-level review. It can now also ingest *measured* collaborative records from `.csv`,
  `.jsonl`, or `.json` via `--input-file`, with schema validation in
  `scripts/validate_measured_input.py`, and emit paper-facing markdown / LaTeX tables; numbers
  below remain hypotheses until the collaborative Exp 2 run is completed.

### Experiment 3 — Cascade propagation depth *(in progress — core, with Y. Gu / #7)*

- **Setup**: Inject a single seeded error at stage 1, 2, or 3 of multi-stage workflows and measure how
  many downstream stages are corrupted.
- **Metrics**: cascade radius, time-to-detection, recovery completeness score.
- **Scale**: 100 traces per injection point × 3 depths × 3 injection points = 900 traces.
- **Status**: scaffolded by PR #21 and now exposed in code via depth-wise diagnostics
  (`mean_cascade_radius`, `mean_recovery_completeness`, `final_task_success_rate`,
  `mean_time_to_detection_ms`). The current runner now sweeps injection stages 1/2/3 and exports
  long-form raw runs, grouped CSV/JSON summaries, and paired bootstrap policy-comparison artifacts.
  It can now also ingest *measured* collaborative records from `.csv`, `.jsonl`, or `.json` via
  `--input-file`, with schema validation in `scripts/validate_measured_input.py`, and emit
  paper-facing markdown / LaTeX tables; this remains the primary differentiation against generic
  fault injection work.

### Experiment 4 — Decomposition quality

- **Setup**: For high-complexity tasks (`complexity_score > 0.7`) that trigger `DECOMPOSE`, compare
  the sub-task decomposition against expert-annotated gold decompositions.
- **Metrics**: delegation fidelity, decomposition granularity (too coarse vs. too fine), and wasted
  sub-agent calls.
- **Scale**: 50 complex tasks × 3 policies = 150 traces.

**Total planned trace budget**: ~4,350 traces across all experiments.

---

## 4. Baseline Experiment

**FixedPolicy** — always routes every task to a single default action regardless of task properties.
It has no failure handling, no decomposition, and no retry logic.

This is the simplest possible orchestrator: it makes zero routing decisions. Any framework that
cannot beat FixedPolicy on a given task category has negative orchestration value — the routing
overhead costs more than it helps.

Secondary baseline: **HeuristicPolicy** — rule-based routing on complexity score, requires_code, and
requires_retrieval flags. This represents a hand-tuned production system without learned routing.

---

## 5. Test vs. Baseline

| Policy | Type | What it tests |
|---|---|---|
| FixedPolicy | Baseline | Zero-intelligence lower bound |
| HeuristicPolicy | Baseline | Rule-based production routing |
| **LLM-as-Router** | Test | Model chooses routing by reasoning over task description + metadata |
| **RetryPolicy(Heuristic)** | Test | Heuristic + automatic retry on failure — does retry alone contain cascades? |
| **RetryPolicy(LLM)** | Test | LLM routing + retry — best-case but highest-cost option |
| OraclePolicy | Ceiling | Routes to the gold label |

**Key comparison axes**:

- Success rate gap (test - baseline) per task complexity band.
- Cost efficiency: success_rate / mean_cost ratio.
- Failure recovery: per-failure-mode recovery rate (Experiment 2).
- Cascade containment: cascade radius under failure injection (Experiment 3).

**Hypothesis.** LLM-as-Router will outperform Heuristic on complex tasks but at higher cost. Retry
wrapping either policy should improve recovery rate but may not reduce cascade radius without
detection, because retry can repeat a corrupted state.

---

## 6. Expected Results

1. **Routing accuracy** *(measured diagnostic)*: FixedPolicy at **23%**, validating the ~25% lower
   bound; HeuristicPolicy at **62% overall but 0% on adversarial cases**; **LLM-as-Router (Claude
   Sonnet 4.6) at 100% overall, including 100% adversarial**. On this focused diagnostic the LLM
   matches Oracle; the broader workflow suite is expected to produce a graded comparison
   (projected LLM 85–95% vs. HeuristicPolicy 65–75%).
2. **Failure recovery** *(hypothesis, to measure)*: recovery should be higher for tool invocation
   errors than for context pollution or conflicting sub-agent outputs. Exact rates must come from
   Exp 2, not be asserted before measurement.
3. **Cascade propagation** *(hypothesis, to measure)*: earlier-stage failures should have larger
   cascade radius than later-stage failures; retry should reduce final failure rate but may not fully
   contain propagation without detection.
4. **Cost-quality Pareto** *(planned)*: simple tasks may favor Fixed/Heuristic due to cost, while
   complex or adversarial tasks should justify model routing.
5. **Statistical confidence**: all comparisons report 95% bootstrap CIs and paired bootstrap
   significance tests (PR #20).

---

## 7. Why It Matters

Multi-agent systems are moving from demos to production. AutoGen, LangGraph, CrewAI, and Anthropic
Agents SDK all ship orchestration layers, but there is **no standard way to evaluate orchestration
reliability** — only task accuracy.

This matters because:

- **Production failures are silent**: a misrouted task might still produce a plausible-looking answer.
  Without failure attribution, teams cannot distinguish "working" from "lucky."
- **Cascade failures are the #1 production risk**: prior work shows errors can propagate through
  pipelines; OrchestrateBench makes cascade radius a primary metric.
- **Framework selection is currently guesswork**: teams choose AutoGen vs. LangGraph based on API
  ergonomics, not measured reliability. OrchestrateBench enables apples-to-apples comparison on
  failure handling.
- **Direct Anote product integration (#14)**: the same keyword-routing weakness exists in Panacea
  (`multi_agent_system.py`); a flag-gated model-driven router was added there (PR #222) as the
  opt-in remedy this benchmark motivates.

---

## 8. Why People Would Care

**For ML engineers / AI engineers**: "My multi-agent pipeline works on demos but fails unpredictably
in production. OrchestrateBench tells me which failure modes my orchestrator cannot handle and how far
errors cascade — before I ship."

**For framework maintainers** (AutoGen, LangGraph, CrewAI teams): OrchestrateBench provides a
standardized reliability leaderboard. Being ranked on failure recovery, not just task success,
differentiates frameworks on the axis production teams care about.

**For the research community**: Existing agent benchmarks do not cover orchestration-level decisions.
OrchestrateBench fills the gap between "can the agent solve the task?" and "can the orchestrator route,
recover, and decompose reliably?"

**For Anote**: The benchmark directly informs Panacea's agent architecture. Every failure mode we
discover in OrchestrateBench becomes a hardening target for the production system. Research → product
is a 1:1 pipeline, not an afterthought.

---

## 9. Reproducibility & Ethics

Every reported number should be regenerable from a fixed seed. Artifact-release checklist:

- [x] Harness, four routing policies, statistics module, and failure injector scaffolded in code.
- [x] Exp 2/3 offline harness now exposes retry-aware recovery, detection, escalation, and cascade
  diagnostics from stable workflow seeds / task IDs.
- [x] Exp 2/3 scripts now export machine-readable raw runs and grouped summaries for plots/tables.
- [x] Exp 2/3 scripts now export paired policy-comparison artifacts for statistical write-up.
- [x] Exp 2/3 measured-input schema, templates, and validator are in repo for collaborative runs.
- [x] Exp 2/3 scripts now emit paper-facing markdown / LaTeX summary artifacts from the same run.
- [x] Experiment 1 diagnostic has measured Fixed / Heuristic / LLM / Oracle results.
- [ ] Exp 2/3/4 measured results.
- [ ] Full workflow-suite seeds + held-out split.
- [ ] Cost log by experiment.

Ethics: this is a reliability benchmark intended to make production AI systems safer, more auditable,
and cheaper to operate by surfacing silent orchestration failures before deployment. Risks include
leaderboard overfitting and dual-use failure knowledge; mitigations are held-out seeded scenarios,
cost reporting, and explicit scope limits. All workflows are synthetic; no human-subjects data is
used.

---

## 10. Current Status

- **Measured**: Experiment 1 routing diagnostic (PR #26), with LLM-as-Router at 100% overall and 100%
  adversarial.
- **Merged scaffolding**: statistical rigor (PR #20) and failure injection/cascade framework (PR #21).
- **Codebase status**: Exp 2/3 harness now supports retry-wrapped policies and emits recovery,
  final-task-success, detection, escalation, and cascade diagnostics in the offline scripts. Those
  scripts now also accept measured long-form collaborative inputs and preflight validation.
- **In progress with Y. Gu**: Experiment 2 failure taxonomy (#4) and Experiment 3 cascade propagation
  (#7). These should not be finalized solo.
- **Open differentiation items for 6/16 standup**: MAS-FIRE external overlap and Paper #2 internal
  intent-spec overlap.

---

## 11. Milestones & Timeline

Gated to the primary venue (**DAI 2026 Industry Track, 8/3**), with one week of buffer before
submission. Owners: Yidian Chen (Y.C.) and Yingzi Gu (Y.G.).

| Window | Milestone | Owner |
|---|---|---|
| 6/16 – 6/20 (Fri) | Design doc finalized (RQs, metrics, hypotheses, literature review, milestones) and linked in the shared research drive | Y.C. |
| 6/20 – 6/27 | Run **Exp 2** (failure injection/recovery, #4) and **Exp 3** (cascade depth, #7) on the workflow suites; replace every hypothesis with measured numbers | Y.C. + Y.G. |
| 6/27 – **7/1** | Full 1,800-trace **Exp 1** suite (graded comparison) + **first complete paper draft** — *meets Natan's July 1 initial-draft deadline* | Y.C. + Y.G. |
| 7/1 – 7/14 | **Exp 4** (decomposition) + cross-model sweep + cost log; tighten MAS-FIRE (topology-vs-routing) and Paper #2 (decision-policy-vs-intent) differentiation | Y.C. + Y.G. |
| 7/14 – 7/21 | Internal mock peer review + revisions; finalize bootstrap CIs and significance tests | both |
| 7/21 – 7/27 | Camera-ready formatting + reproducibility package + artifact release (seeds, gold set, one-command repro) | both |
| **7/27** | **Freeze** — one-week buffer before DAI | both |
| **8/3** | **DAI 2026 Industry Track submission** | both |

**Fallbacks**: EMNLP ORACLE workshop (9/18) if Exp 2/3 slip past the DAI window; AAAI 2027 (7/28
abstract) only if the cascade-radius headline lands with strong measured results.
