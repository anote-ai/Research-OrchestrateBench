# OrchestraBench — One-Page Summary

**What it is.** A benchmark and reproducible harness for measuring how multi-agent LLM
orchestration systems route tasks, fail, recover, and decompose work — not just whether
they reach the right final answer.

**Who it's for.**
- ML/AI engineers shipping multi-agent pipelines who need to know which failure modes their
  orchestrator cannot handle before production.
- Framework maintainers (AutoGen, LangGraph, CrewAI, Anthropic Agents SDK) who currently have
  no standardized reliability axis to compete on.
- Researchers studying agentic-system robustness who need a controlled, seed-reproducible
  injection harness rather than observational failure logs.

**The four core measured findings (Claude Sonnet 4.6, real-agent runs — not simulated).**

| # | Finding | Headline number | Status |
|---|---|---|---|
| 1 | Routing mechanism, not task difficulty, explains failure | Heuristic 0% -> LLM 100% on adversarial cases | Measured (26-case diagnostic) |
| 2 | Retry recovers tool faults but not semantic/latent failures | Tool 1.0 recovery vs. 0.0 for 4 latent modes | Measured, N=30 + domain N=30 |
| 3 | Cascade radius scales with pipeline depth | 1.0 -> 2.9 -> 5.0 across depths 3/5/7 | Measured, N=90 |
| 4 | Decomposition quality is invisible to final-answer grading | Fidelity 1.00 (decompose) vs. 0.37 (monolithic) | Measured, N=20 |

A fifth, policy-conditioned result shows an LLM router can cut cascade radius ~5.5x at depth 7
— but an ablation shows most of that gain is the trusted-upstream-value hint, not autonomous
fault detection. We report the ablation alongside the headline number rather than omit it
(see PAPER.md §5.5 and §7 for the full honesty framing).

**What's still open** (see `DESIGN_DOC.md` §9 for the full checklist):
- The full 1,800-trace Experiment 1 workflow suite (current diagnostic is 26 hand-labelled cases).
- Collaborative external measured records for Exp 2/3 beyond the in-repo real-agent runs.
- Exp 4 at full 50-task x 3-policy scale with inter-annotator κ on gold decompositions.
- Cross-model sweep (currently single model family, Claude Sonnet 4.6).
- Validation on the full finance/HR/DevOps enterprise workflow suite (current Exp 2/3 results
  are controlled mechanism probes on a verifiable arithmetic chain, by design — see PAPER.md §7).

**Where to look.**
- `DESIGN_DOC.md` — research questions, related work, experiment specs, milestones.
- `PAPER.md` — full academic draft (v0.6) with all measured results, limitations, and ethics.
- `BLOG.md` — this benchmark explained for a general engineering audience.
- `paper/main.tex` / `paper/main.pdf` — LaTeX/PDF build of the paper draft.
- `REPRODUCIBILITY.md` — exact provenance rules and commands for every cited number.
- `data/measured/*_real.csv` — the raw measured data behind every number in this document.
