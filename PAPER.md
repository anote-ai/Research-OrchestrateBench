# OrchestraBench: Evaluating Multi-Agent Orchestration Failure Modes, Recovery, and Decomposition Quality

**Authors**: Yidian Chen, Yingzi Gu · **Supervisor**: Natan Vidra (Anote)
**Target venues**: DAI 2026 Industry Track (8/3) · EMNLP ORACLE workshop (9/18) · AAAI 2027 (7/28)

> **DRAFT v0.6 (2026-06-22, addresses issue #8).** Built from the approved design document (`DESIGN_DOC.md`).
> **Honesty markers**: All four experiments now report *measured* results (Exp 1 routing; Exp 2 = real Claude
> N=30 + domain-grounded loan-approval re-run N=30; Exp 3 = real Claude N=90, cascade-by-depth; Exp 4 = real
> Claude N=20, decomposition). Exp 2/3 (failure taxonomy / cascade) remain a **core contribution with Y. Gu
> (issues #4/#7)**.

---

## Abstract

Multi-agent orchestration frameworks (AutoGen, LangGraph, CrewAI, Anthropic Agents SDK) are moving
from demos to production, yet they can only be compared on *task accuracy* — not on *reliability*.
Existing agent benchmarks measure whether a pipeline succeeds, but cannot diagnose *why* it failed,
*where* a cascade began, or *which* routing decision caused the breakdown. We present **OrchestraBench**,
the first benchmark that evaluates how multi-agent systems **fail, recover, and decompose** as
first-class, comparable metrics. OrchestraBench contributes (i) a controlled, seed-reproducible
**failure-injection harness** over templated enterprise workflows, (ii) **cascade-radius** and
per-failure-mode recovery as primary metrics, and (iii) a **routing-policy comparison** with full
statistical rigor. Our first experiment isolates the routing-mechanism question: on a 26-case
gold-labelled diagnostic, a keyword/flag heuristic — representative of production rule-based routers —
scores **0% on adversarial cases** (misleading or missing surface flags), while a model-driven
router that reasons over intent scores **100%**, closing the entire gap to an oracle ceiling. This
quantifies, reproducibly, that the heuristic's blind spot is a property of the routing *mechanism*,
not task difficulty. Two further experiments, run with a real Claude agent over a **verifiable arithmetic
dependency chain** as a controlled mechanism probe, quantify the reliability axis no existing benchmark
reports: across five MAST failure modes, the four latent/semantic modes fail completely (final-task success
**0.0**) while a tool fault is fully recovered (**1.0**), and **retry does not repair the latent modes** — it
extends a corrupted trace rather than containing it; and cascade radius **scales monotonically with pipeline
depth** (mean 1.0 / 2.9 / 5.0 at depths 3 / 5 / 7) — the stages-traversed signature MAS-FIRE leaves
unquantified. We frame these as mechanism probes on a controlled chain, not domain-workload claims (§7).

---

## 1. Introduction

Multi-agent systems increasingly orchestrate several specialized agents — routers, retrievers,
tool-callers, planners — behind a single task. When such a pipeline produces a wrong or low-quality
answer in production, the failure is often *silent*: a misrouted task still yields a plausible-looking
response. Current benchmarks (AgentBench, OdysseyBench, SWE-Bench) report final task success and so
cannot tell teams **which** orchestration decision failed, **how far** an error propagated, or
**whether** intelligent routing was worth its cost. This blocks the most basic production decision:
choosing and hardening an orchestration framework on *reliability* rather than ergonomics.

**Contributions.**
1. **OrchestraBench**, a benchmark that treats orchestration *failure handling* as the unit of
   evaluation: controlled failure injection, per-failure-mode recovery, and cascade propagation.
2. A **seed-reproducible** workflow + failure-injection harness (every reported number regenerable
   from a seed) over three enterprise workflow families.
3. A **routing-policy comparison** (Fixed / Heuristic / LLM / Oracle) with bootstrap confidence
   intervals and paired significance testing, isolating *when* intelligent routing beats a
   zero-decision baseline once cost is accounted for.

**Research questions.**
- **RQ1 (routing value)**: When does intelligent routing (heuristic or LLM) actually beat a
  zero-decision baseline, once cost is accounted for?
- **RQ2 (mechanism)**: Is routing reliability a property of task *difficulty*, or of the routing
  *mechanism* (keyword/flag matching vs. reasoning over intent)? *(Answered by Exp 1.)*
- **RQ3 (attribution)**: Can a pipeline failure be attributed to a specific failure mode *and* stage,
  reproducibly across runs?
- **RQ4 (cascade)**: How far does a single seeded error propagate downstream, and which
  routing/recovery strategies contain it?

---

## 2. Related Work

| Prior work | What it does | How OrchestraBench differs |
|---|---|---|
| **MAST** — *Why Do Multi-Agent LLM Systems Fail?* (Cemri, Pan et al., **arXiv:2503.13657**; 14 failure modes in 3 categories, MAST-Data 1,600+ traces over 7 frameworks, κ=0.88) | Empirically-grounded *taxonomy* of **observed** MAS failures | We **inject** that taxonomy under seed-controlled conditions and measure recovery + cascade *per mode* — MAST observes, we intervene |
| **MAS-FIRE** (**arXiv:2602.19843**) | Fault **injection** + reliability evaluation; intra-/inter-agent fault taxonomy via prompt / message-routing manipulation | **Closest prior work** (see §2.1). We differ on emphasis: cascade **radius across pipeline depths** + **routing-policy comparison** as the unit of analysis, not reliability scoring alone |
| **TraceElephant** — *Seeing the Whole Elephant* (**arXiv:2604.22708**; 220 annotated traces) | Benchmark for *post-hoc failure attribution* in MAS | Attribution is observational; we add controlled seeding + cascade radius as **primary** metrics |
| **Agents Failure Attribution** (ICML 2025 Spotlight; 184 tasks) | Post-hoc attribution to the responsible agent/step | Same: attribution is a means here, not the product |
| **Orchestration-pattern benchmark** (**arXiv:2603.22651**; 10k SEC filings) | Compares sequential / parallel / hierarchical / reflexive architectures on a cost-accuracy Pareto (reflexive F1 0.943 @2.3×; hierarchical 0.921 @1.4×) | Architecture comparison on *accuracy/cost*; we evaluate *failure handling* and routing **mechanism** — complementary |
| **AdaptOrch** (arXiv:2602.16873) | Task-adaptive orchestration *framework* (+12–23%) | A method; OrchestraBench is the benchmark that would evaluate it |
| **"From Spark to Fire"** (arXiv:2603.04474) | Errors cascade exponentially in pipelines | We operationalize this into a measurable cascade-radius benchmark |

**Motivating data point.** *Beyond the Strongest* (arXiv:2509.23537) reports that on GPQA-Diamond at
least one agent was correct in **95.5%** of cases, yet orchestration reached only **87.4%** — i.e.
orchestration *discards* ~8 points of individually-recoverable correctness. OrchestraBench is built to
attribute exactly this loss.

**The gap.** Existing work either *observes* failures (MAST), *attributes* them post-hoc
(TraceElephant, ICML 2025), or *proposes* better orchestration (AdaptOrch). The one work that also
*injects* (MAS-FIRE) scores reliability but does not make cascade radius or routing-policy mechanism
its unit of analysis. OrchestraBench's contribution is a **controlled, reproducible injection harness
that measures recovery and cascade containment as first-class metrics, compared across routing
policies.**

### 2.1 Threats to novelty (anticipating reviewers) — addresses #18

An honest audit (the space moved fast in early 2026) surfaces one direct overlap and our response:

- **MAS-FIRE (arXiv:2602.19843) already performs controlled fault injection + reliability evaluation
  for MAS**, overlapping OrchestraBench's Exp 2/3 (issues #4/#7) substantially. Our differentiation
  is **not** "we inject failures" (they do too); it must be (a) **cascade radius across pipeline
  depths** as a first-class comparable metric, (b) failure handling measured **per routing policy**
  (Fixed/Heuristic/LLM/Oracle), and (c) the **routing-mechanism result** of Exp 1, which MAS-FIRE does
  not address. A reviewer *will* ask "why not just extend MAS-FIRE?" — the paper must lead with the
  routing + cascade-radius framing, not generic injection. **Verified (full-text read, 2026-06-17): MAS-FIRE does *not* quantify cascade depth (no stages-traversed metric — only the qualitative "propagates downstream unchecked") and does *not* vary routing policy; its axes are topology (MetaGPT / Table-Critic / CAMEL), fault-type (15), and model. Both of our primary differentiators are uncovered, not merely reframed.**
- **Attribution benchmarks (TraceElephant 2604.22708; ICML 2025) are observational/post-hoc** — a
  different instrument from seeded, reproducible injection + recovery. Lower overlap risk.
- **Internal overlap (flagged by N. Vidra in the #11 assignment): Paper #2, *Intent Specification as a
  First-Class Evaluation Object* (S. Haing / Spurthi).** Differentiation: #2 evaluates whether the
  agent's *understanding of intent* is correct (intent-spec vs. syntactic tool-call correctness); #11
  evaluates which *orchestration action* to take given a correct understanding (route / decompose /
  code-vs-reason) and how those decisions cascade. We measure the **decision policy and its failure
  propagation**, not intent correctness — coordinate scope with the #2 team so the papers complement. **Verified (read of #2 design doc, 6/17): the overlap is surface-level — #2 (IntentBench) scores spec quality (SQS) + intent-violation (IVR) on single spec→code pairs in the coding domain; #11 measures routing / cascade / recovery across a multi-agent pipeline. Different unit, metrics, and failure mode → complementary ends of the pipeline, no measurement overlap to resolve.**
- **Status (2026-06-18): both novelty risks resolved.** The external overlap (MAS-FIRE) is closed by
  full-text verification — neither *cascade radius across depths* nor *per-routing-policy* failure
  handling is covered by it (above). The internal overlap (Paper #2 / IntentBench) is closed by reading
  its design doc — it scores intent/spec quality (IVR/SQS) at the spec→code layer, while OrchestraBench
  measures cascade/recovery/routing at the multi-agent orchestration layer, with no measurement overlap.
  Remaining work is a courtesy cross-cite with the #2 team, not a differentiation gap.

---

## 3. The OrchestraBench Benchmark

**Workflow suites.** Synthetically generated from templated task graphs (stages, inter-task
dependencies, per-task `complexity_score`), screened by both co-authors for realism. Three enterprise
families: finance approval (4 stages), HR onboarding (5 stages), DevOps deployment (5 stages).

**Routing decision space.** Each task is routed to one of four actions: `DIRECT_TOOL`,
`CODE_EXECUTION`, `DECOMPOSE`, `REASON_ONLY`.

**Gold labels.** Routing and decomposition labels are produced independently by both co-authors with
adjudication on disagreement (target inter-annotator agreement Cohen's κ > 0.7).

**Failure injection.** Failure scenarios are seeded **deterministically** on top of the suites via the
`FailureInjector` (shipped, PR #21), so every number is reproducible from a seed.

**Metrics.** routing_accuracy, routing_macro_F1, per-class routing metrics, success_rate, mean_latency,
mean_cost, orchestration_efficiency, per-failure-mode recovery rate, escalation latency, **cascade
radius**, time-to-detection, recovery completeness, decomposition delegation fidelity.

---

## 4. Experimental Setup

| Policy | Type | What it tests |
|--------|------|---------------|
| FixedPolicy | Baseline | Zero-intelligence lower bound (always one route) |
| HeuristicPolicy | Baseline | Rule/keyword routing on complexity + flags (production rule-based system) |
| **LLM-as-Router** | Test | Model decides routing per-task by reasoning over description + metadata |
| RetryPolicy(Heuristic/LLM) | Test | Adds automatic retry — does retry alone contain cascades? |
| OraclePolicy | Ceiling | Routes to the gold label (upper bound) |

All comparisons report 95% bootstrap CIs and paired bootstrap significance (α = 0.05) via the
statistics module (PR #20).

---

## 5. Results

### 5.1 Experiment 1 — Routing policy comparison *(measured)*

We isolate the routing-mechanism question on a focused diagnostic of **26 expert-labelled gold cases**
covering all four routing decisions: 16 *aligned* (surface flags match intent) and 10 *adversarial*
(flags missing or misleading, but the description's intent is unambiguous). The model-driven router is
Claude Sonnet 4.6 via forced structured (tool-use) output; results are stable across 3 independent
passes (78/78), and the prompt never sees the gold label.

| Policy | Overall | Aligned | Adversarial |
|---|---|---|---|
| FixedPolicy | 23% | 25% | 20% |
| HeuristicPolicy (keyword/flags) | 62% | **100%** | **0%** |
| **LLM-as-Router (Sonnet 4.6)** | **100%** | 100% | **100%** |
| OraclePolicy (ceiling) | 100% | 100% | 100% |

**Finding.** The heuristic is perfect on aligned cases but collapses to **0%** on adversarial ones;
the model-driven router recovers **all** of them (0% → 100%), closing the entire gap to the oracle.
Because a *cost-efficient* model saturates this set, the heuristic's adversarial failure is shown to
be a property of the routing **mechanism** (keyword/flag matching vs. reasoning over intent), **not**
task difficulty. The larger 1,800-trace workflow suite is where the LLM router is expected to drop
below 100% and yield a graded comparison (projected 85–95% vs. heuristic 65–75%).

> **Product tie-in (#14).** The same keyword-routing weakness exists in Anote's Panacea
> (`multi_agent_system.py`); a flag-gated model-driven router was added there (PR #222) as the
> opt-in remedy this experiment motivates.

### 5.2 Experiment 2 — Failure injection and recovery *(real Claude measured run + offline harness; core, with Y. Gu / #4)*

**Design.** Five MAST failure modes — *ambiguous delegation*, *tool-invocation error*, *context
pollution*, *conflicting sub-agent outputs*, *premature action* — injected at the **prompt level** into a
verifiable arithmetic dependency chain executed by a **real Claude agent** (Sonnet 4.6), under routing
policies `fixed` / `heuristic` / `retry(heuristic)`. Real recovery / cascade are measured by exact-match
against ground truth; the same modes also run through the offline auto-harness (PR #33/#34) for large-N
projection, and real records feed the measured-input pipeline (`measured_runs.py`). Code: `real_run.py`.

**Real measured findings (Claude Sonnet 4.6, N=30, 2026-06-20).** `tool_invocation_error` is **fully
recovered** (recovery 1.0, cascade radius 0) — the agent computes by hand when the tool fails. The four
**latent/semantic modes** (ambiguous delegation, context pollution, conflicting outputs, premature
action) all **fail** (final-task success **0.0**) and **cascade to every downstream stage** (cascade
radius **2 of 2**). Critically, **`retry(heuristic)` does not recover them**: retrying while the failure
is still present reproduces the error and only **lengthens time-to-detection** — it extends a corrupted
trace instead of containing it. This real-agent result **confirms** the offline-harness mechanism: retry
repairs retryable tool faults but not failures needing attribution, state repair, or semantic validation
(RQ3/RQ4).

**Statistics & limitations (honest).** Bootstrap 95% CIs are degenerate here — latent modes 0.0 [0.0, 0.0]
and the tool fault 1.0 [1.0, 1.0] (n=6 each) — reflecting the controlled construct, not statistical strength
(see §7, construct validity). Small-N MVP; agents are LLM-simulated (no real external tools); the workload is
a controlled arithmetic chain (clean ground truth for cascade, not a domain task); success is
exact-match. Next: the full 1,500-trace suite, harder/domain workloads, and a cross-model sweep. Data:
`data/measured/exp2_real.csv`.

**Domain-grounded validation (loan-approval workflow, real Claude N=30, 2026-06-22).** To test whether
these signatures are an artifact of the abstract arithmetic chain, we re-ran the *identical* failure
injection on a domain-grounded variant: the same verifiable computation reframed as a multi-role
loan-approval pipeline (Intake Officer → Risk Analyst → Compliance Officer → Approval Manager), each stage
prompted in business terms (`run_exp2 --domain`). **The core mechanism holds**: the latent/semantic modes
(conflicting outputs, context pollution, premature action) still fail completely (final-task success
**0.0**), and the tool fault stays more recoverable than the latent modes — the failure-mode *ordering* is
robust across framings (**Figure 2**). Crucially, the **absolute rates shift with framing**: `ambiguous_delegation` rises to
**0.50** (the business-role context lets the agent infer the intended operation about half the time) and
`tool_invocation_error` recovery falls to **0.67** (by-hand recomputation is noisier inside the richer
business prompt). This framing-sensitivity is the signature of *real agent behavior*, not a structural
tautology — the mechanism is robust while the magnitudes respond to context. Directly addresses the
construct-validity concern (§7). Data: `data/measured/exp2_domain_real.csv`.

### 5.3 Experiment 3 — Cascade propagation depth *(real Claude measured run, N=90 + offline harness; core, with Y. Gu / #7)*

**Design.** Inject a single seeded error at stage 1/2/3 of variable-depth (3-, 5-, 7-stage) pipelines
and measure how far it propagates. **Metrics**: **cascade radius** (downstream stages corrupted — the
primary differentiator vs. MAS-FIRE, which has no stages-traversed metric), time-to-detection, recovery
completeness, per routing policy. **Scale**: planned = 100 × 3 depths × 3 injection points = **900
traces**; the one-command harness (`scripts/run_exp23_pipeline.py`) currently reproduces **135
measured-style rows per mode** and ingests real records via `--input-file`.

**Offline auto-harness findings (2026-06-20, `context_pollution`).** Earlier-stage failures produce
strictly larger cascades, and deeper pipelines amplify it: mean cascade radius = **2 / 4 / 6** for
inject-stage-1 at depths 3 / 5 / 7, **1 / 3 / 5** for stage-2, **0 / 2 / 4** for stage-3 — exactly the
depth×stage signature no existing benchmark measures. `retry(heuristic)` does **not** improve cascade
containment or final-task success on latent failures.

**Real measured findings (Claude Sonnet 4.6, N=90, 2026-06-22).** `run_exp3` extends the Experiment 2
real-agent design to variable depth — the same prompt-level injection and exact-match grading, looped over
3-/5-/7-stage chains with the error injected at stage 1. Across the four latent/semantic failure modes, **mean
cascade radius scales monotonically with depth: 1.0 [1.0, 1.0] (depth 3) → 2.9 [2.6, 3.0] (depth 5) → 5.0 [5.0, 5.0] (depth 7)** (bootstrap 95% CIs, latent modes pooled, n=24/depth; **Figure 1**) — the failure
propagates to essentially every downstream stage (depth − 2). `tool_invocation_error` stays at **cascade radius
0 at every depth** (the agent recomputes by hand). The lone deviation is `ambiguous_delegation` at depth 5
(mean 2.5 vs 3.0 for the other latent modes), where the real agent occasionally guesses the intended operation
correctly — a realistic-noise signature absent from the deterministic offline model. All 90 records pass the
`MeasuredExp3Record` schema. Data: `data/measured/exp3_real.csv`. **This is the paper's headline cascade-depth
result and the primary differentiator vs. MAS-FIRE, which reports no stages-traversed metric.** (The
offline-harness figures above are a large-N trend projection under a slightly different cascade convention;
these real measured values supersede them as the primary result.)

### 5.4 Experiment 4 — Decomposition quality *(real Claude measured run, N=20; secondary)*

For composite tasks `(a op b) op (c op d)` with a canonical 3-step gold decomposition, we score the
produced decomposition against gold — **delegation fidelity** (gold sub-results recovered), **granularity
error**, **wasted sub-tasks** — comparing a `monolithic` (single-step) vs a `decompose` (planner) policy.

**Real measured findings (Claude Sonnet 4.6, N=20, 2026-06-22).** The `decompose` policy produces a
faithful three-step decomposition (**delegation fidelity 1.00, granularity error 0**), while `monolithic`
reaches the correct final answer but exposes no delegable sub-structure (**fidelity 0.37, granularity error
2**). Notably, `final_correct` is **1.0 for both** — the arithmetic is easy enough that both get the answer;
the discriminator is the **decomposition structure**, not final correctness. For multi-agent orchestration
this is exactly the point: a monolithic agent can be *right* yet produce nothing a planner can delegate,
audit, or recover from — decomposition quality is about delegable structure, which a routing policy can
measure and reward. Data: `data/measured/exp4_real.csv`. (Scale: 10 tasks × 2 policies = 20 traces; a larger
suite with a 3-policy comparison and gold κ is future work.)

---

## 6. Discussion

Experiment 1 establishes the paper's first reproducible claim: routing reliability is a function of
*mechanism*. Experiments 2 and 3 deliver the reliability headline on a controlled verifiable chain: the
**failure-mode gap** is measured (four latent modes at 0.0 final-task success vs. a fully-recovered tool
fault), and **cascade radius scales with pipeline depth** (1.0 → 2.9 → 5.0 at depths 3 / 5 / 7) — the
stages-traversed signature no existing benchmark quantifies. Critically, **retry does not eliminate latent
cascades**: it reproduces the failure and lengthens time-to-detection, so *detection/attribution* — not
blind retry — is the necessary containment mechanism. We are deliberate that these are mechanism probes on
a controlled chain (§7); domain-workflow validation is the next step.

---

## 7. Limitations

- The Exp 1 diagnostic is small (26 cases) and *saturates* for a capable model — it is a mechanism
  probe, not a difficulty benchmark; the workflow-suite results are needed for a graded comparison.
- Workflows are synthetically generated; real-world pipeline validation is future work (#12).
- Single model family reported for the LLM router; a cross-model sweep is future work.
- **Construct validity of Exp 2/3 (important).** The failure-recovery and cascade experiments run on a
  *verifiable arithmetic dependency chain*, chosen for clean ground truth, not a domain workflow. In this
  construct cascade radius is **partly by design** — a downstream stage consumes the upstream value, so a
  latent corruption propagates to ~(depth − 2) stages — so we present Exp 2/3 as **controlled mechanism
  probes**, not domain-workload claims. The evidence that this measures *real agent behavior* rather than a
  tautology is the non-determinism we observe: `ambiguous_delegation` at depth 5 yields mean cascade
  2.875 [2.625, 3.0], below the structural maximum, because the real agent sometimes recovers the intended
  operation. **We have taken a first step (§5.2): a domain-grounded loan-approval re-run shows the
  failure-mode *ordering* is robust across framings while absolute rates shift with context (real agent
  behavior, not a tautology).** Full finance / HR / DevOps suite validation remains future work.
- **Single LLM, not a literal multi-agent system.** Exp 2/3 execute one Claude agent over a staged chain
  with prompt-level fault injection — simulating orchestration failure modes rather than wiring N
  independent agents; a real multi-agent harness is future work.

---

## 8. Ethics & Broader Impact *(addresses #17)*

OrchestraBench evaluates the *reliability* of automated orchestration. Improving failure attribution
and cascade containment is intended to make production AI systems **safer, more auditable, and cheaper
to operate** — surfacing silent failures before deployment rather than after.

**Risks and mitigations.**
- *Leaderboard overfitting*: a public reliability benchmark can incentivize tuning to the injected
  failure modes rather than real robustness. Mitigation: seed-controlled, **held-out** failure
  scenarios, and reporting cost alongside accuracy so "reliability via brute-force expensive routing"
  stays visible rather than rewarded.
- *Dual use*: cataloguing how orchestrators fail could inform adversarial prompting. Mitigation: the
  injected modes are already public (MAST); we add measurement, not new attack surface.
- *Over-trust*: a high score must not be read as production safety. We state scope limits (synthetic
  workflows; mechanism probes) explicitly (§7).

No human-subjects data is used; all workflows are synthetic. The benchmark is released for research and
engineering reliability evaluation.

## 9. Reproducibility *(addresses #15)*

Every reported number is regenerable from a fixed seed. Artifact-release checklist:

- [x] **Code released** — harness, four routing policies, 26-case gold set, `FailureInjector` (PR #21),
  statistics module (PR #20), offline Exp 2/3 harness (PR #30) + **real-LLM measured run `real_run.py`
  (PR #37, Exp 2 N=30 + Exp 3 N=90)**; Apache-2.0.
- [x] **Deterministic data** — workflow suites + failure scenarios generated from seeds; the gold set
  is committed, not regenerated.
- [x] **Tests** — routing comparison ships 12 unit tests; **full suite 107 passing** (incl. 8 for the
  Exp 2/3 harness, PR #30); non-LLM policies run offline (no key/network).
- [x] **Pinned environment** — Python ≥3.10; deps pinned in `pyproject.toml`; CI on 3.10/3.11/3.12.
- [ ] **LLM-router reproducibility** — model id (`claude-sonnet-4-6`), exact prompt, and `tool_choice`
  config documented; result stable across 3 passes (78/78). State the non-determinism caveat and
  report multi-trial means + bootstrap CIs (PR #20) in the final table.
- [x] **Cost log** — Sonnet 4.6; Exp 1 ≈ $0.07/run, Exp 2 (N=30) ≈ $0.4, Exp 3 (N=90) ≈ $0.4; full
  real-LLM total across the project ≈ **$1** (token-estimated; short arithmetic prompts). Reproduction cost
  is negligible — a strength for an artifact-release benchmark.
- [x] **One-command repro** — `python scripts/reproduce_exp1.py` regenerates the §5.1 table from the
  committed gold set (offline baselines need no key; LLM row runs when `ANTHROPIC_API_KEY` is set).

## 10. Conclusion

OrchestraBench reframes multi-agent evaluation from "did the task succeed?" to "did the orchestrator
route, recover, and decompose reliably?" Experiment 1 delivers a clean, reproducible result —
model-driven routing closes a 0%→100% adversarial gap that keyword routing cannot — while the
failure-injection and cascade experiments measure, on a controlled chain, the reliability axis production
teams most need: latent failures that retry cannot repair, and a cascade radius that grows with pipeline
depth (1.0 → 2.9 → 5.0 across depths 3 / 5 / 7).

---

## Figures

- **Figure 1** (`figures/exp3_cascade_by_depth.png`): Exp 3 — mean cascade radius vs pipeline depth; latent/semantic modes scale 1.0 → 2.9 → 5.0 across depths 3/5/7 while `tool_invocation_error` stays at 0 (real Claude, N=90).
- **Figure 2** (`figures/exp2_arith_vs_domain.png`): Exp 2 — failure-mode final-task success, arithmetic chain vs loan-approval workflow; the failure-mode *ordering* is robust across framings while absolute rates shift (real Claude, N=30 each).

## References *(working list; to be formatted to venue style)*

1. Cemri, Pan, et al. **MAST: Why Do Multi-Agent LLM Systems Fail?** arXiv:2503.13657, 2025. (14 failure modes; 1,600+ traces; κ=0.88.)
2. **MAS-FIRE: Fault Injection and Reliability Evaluation for Multi-Agent Systems.** arXiv:2602.19843, 2026. (Closest external work; axes = topology × fault-type × model.)
3. **TraceElephant: Seeing the Whole Elephant — Post-hoc Failure Attribution.** arXiv:2604.22708, 2026. (220 annotated traces.)
4. **Which Agent Causes Task Failures and When?** ICML 2025 Spotlight. (Post-hoc attribution; 184 tasks.)
5. **Orchestration-Pattern Benchmark** (sequential/parallel/hierarchical/reflexive). arXiv:2603.22651, 2026. (10k SEC filings; cost-accuracy Pareto.)
6. **AdaptOrch: Task-Adaptive Orchestration.** arXiv:2602.16873, 2026. (Orchestration method; +12–23%.)
7. **From Spark to Fire: Error Cascades in Agent Pipelines.** arXiv:2603.04474, 2026.
8. **Beyond the Strongest: Orchestration vs. Best-Agent Ceiling.** arXiv:2509.23537, 2025. (GPQA-Diamond: 95.5% any-agent vs. 87.4% orchestrated.)
9. **Doctor-RAG: Diagnosing and Repairing RAG Failures.** arXiv:2604.00865, 2026.
10. **IntentBench: Intent Specification as a First-Class Evaluation Object** (Anote Paper #2, S. Haing / Spurthi). Complementary spec-layer benchmark — cross-cite.
11. AgentBench; SWE-Bench; OdysseyBench — final-task-success agent benchmarks.

---
### Authoring notes (not part of submission)
- **Done (solo, no Exp 2/3 numbers needed)**: §1–§4, §5.1 (measured Exp 1), §2 + §2.1 novelty audit
  (#18 — **both external/internal risks resolved 6/17**), §5.2–§5.4 experiment designs + metrics + scale,
  §8 ethics (#17), §9 repro checklist (#15), §6/§7/§10.
- **Done (real measured, 6/20–22)**: §5.2 (Exp 2, real Claude N=30) and §5.3 (Exp 3, real Claude N=90) —
  run via `real_run.py` (PR #37), bootstrap CIs computed, paper at v0.4. Coordinate with Yingzi on her
  offline-harness pipeline (`measured_runs.py`) + gold-label κ for the larger workflow-suite run.
- **Still open**: Exp 4 (decomposition); add LLM/Oracle routing-policy columns to Exp 2/3 cascade (needs a
  real run); domain-workflow validation for construct validity (§7); full 1,800-trace Exp 1 suite; format
  references to venue style.
- **Source**: in-repo `DESIGN_DOC.md` (1:1 mapping; its milestone table gates to DAI 8/3 with a 7/27 freeze).

### #9 — Venue strategy
| Venue | Deadline | Fit | Note |
|---|---|---|---|
| **DAI 2026 Industry Track** | 8/3 | High — production-reliability angle, shorter format | Primary near-term target; Exp 1 + partial Exp 2 may suffice |
| **AAAI 2027** | 7/28 (abstract) | Medium-high — top-tier, needs full Exp 2/3 | Stretch; only if cascade results land |
| **EMNLP ORACLE** (workshop) | 9/18 | High — reliability/operational-eval theme fits exactly | De-risked fallback (official venue per #11 sheet) |
- **Strategy**: target **DAI 2026** as primary (industry track rewards the reliability + cost-Pareto
  framing); **EMNLP ORACLE workshop (9/18)** as the de-risked fallback if Exp 2/3 slip past the DAI
  window; **AAAI 2027** only if Exp 2/3 deliver the cascade headline with strong measured results.
