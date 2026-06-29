# Your AI agents aren't failing where you think they are

*A plain-language summary of OrchestraBench, a research project measuring how multi-agent AI systems route, fail, and recover.*

## The problem in one sentence

Multi-agent AI systems (think: a "router" agent that hands a task off to a coder agent, a retrieval agent, a planner) can produce a confident-looking wrong answer, and today there is basically no standard way to tell which internal decision caused that -- was it the routing, a tool failure, a misunderstanding that snowballed, or a bad task breakdown?

Most existing agent benchmarks (AgentBench, SWE-Bench, and similar) only tell you "the agent got the right final answer" or "it didn't." That's like grading a factory only on whether the final product shipped, with no idea which machine on the line broke.

OrchestraBench is a benchmark built specifically to open up that black box for the orchestration layer -- the part of a multi-agent system that decides who does what, what happens when something goes wrong, and how far that wrongness spreads.

## What we actually measured (not projected -- measured)

We ran a series of experiments with a real Claude agent (Sonnet 4.6), not simulations, and the results are genuinely interesting.

**1. Keyword-based routing has a structural blind spot, and it isn't subtle.**
We tested a common pattern: route tasks based on keywords/flags in the request (e.g. "if it mentions 'calculate', send it to the code tool"). On a 26-case test set split into "aligned" cases (the keywords match the real intent) and "adversarial" cases (the keywords are missing or misleading, but a human would still understand the intent), the keyword router scored 100% on aligned cases and **0% on adversarial cases**. A model-driven router -- Claude reasoning over the task description instead of pattern-matching keywords -- scored 100% on both, fully closing the gap. That is a clean, reproducible demonstration that the failure is not about how hard the task is, it is about how the router decides.

**2. Not all failures are created equal, and retry is not a magic fix.**
We deliberately injected five common multi-agent failure types (based on a published taxonomy of real-world agent failures, MAST) into a controlled task chain. The result: a tool-calling error was fully recoverable (the agent worked around it), but the four "semantic" failure types -- like an agent misunderstanding what it was asked to do, or carrying forward corrupted context -- were never recovered by simple retries in this setup (final-task success stayed at 0.0). Retrying just repeated the same mistake, and sometimes made it slower to notice anything was wrong. A follow-up run on a domain-flavored loan-approval workflow (rather than abstract arithmetic) showed the same *ordering* of which failures are recoverable, even though the exact numbers shifted with the framing -- a sign this is a real, generalizing pattern rather than an artifact of one toy setup.

**3. Errors compound with pipeline depth, and you can put a number on it.**
We measured how far a single injected error spreads through pipelines of different lengths (3, 5, and 7 stages). The error's "blast radius" grew roughly in step with pipeline depth -- mean cascade radius went **1.0 -> 2.9 -> 5.0** stages as depth went 3 -> 5 -> 7 -- while the recoverable tool error stayed contained at zero spread regardless of depth. This cascade-radius-across-depth measurement is something prior failure-injection research (including the closest related benchmark, MAS-FIRE) has not reported as a first-class number.

**4. Smart routing can dramatically shrink that blast radius -- but check what's actually doing the work.**
Adding an LLM-based router (instead of a fixed or keyword policy) to the cascade experiment cut the latent-failure cascade radius roughly 5.5x at the deepest pipeline tested. That sounds like a strong "use a smarter router" result, and it partly is -- but when we re-ran the same test *without* handing the router the trusted upstream value (an ablation), most of that gain disappeared. In other words, a lot of the benefit was the router being handed clean information to check against, not the model autonomously detecting that something was wrong. We report both numbers, because the honest version of this finding is more useful than the impressive-sounding one.

**5. Smart task breakdown matters even when the final answer is right.**
We compared a "plan and delegate" approach against a single agent doing everything in one shot, on composite tasks with a known gold decomposition. Both approaches reached the correct final answer essentially every time, but the single-shot agent produced little to no usable internal structure for a human or downstream system to audit or hand off pieces of (delegation fidelity 0.37), while the plan-and-delegate approach matched the gold decomposition almost exactly (delegation fidelity 1.00). Looking only at "did it get the right answer" would have completely missed this difference.

## Why this matters if you're building with AI agents

If your team is choosing between AutoGen, LangGraph, CrewAI, or a custom orchestration layer, you are currently choosing based on developer experience and vibes, not measured reliability. OrchestraBench's goal is to make "how does this framework fail, and how far does that failure spread" into something you can benchmark, the same way you would benchmark latency or accuracy.

The practical takeaway, even before the full benchmark suite ships: if your production routing layer uses keyword or flag matching, you likely have an adversarial blind spot similar to the one we measured -- it will look fine in normal testing and fail silently exactly when a request does not use the "expected" words. Routing on reasoning over intent, rather than surface pattern matching, closes that gap. And if you're tempted to "just add retry" as your failure-handling strategy, our data says that works for one specific failure family (tool errors) and does nothing -- or actively hurts time-to-detection -- for the semantic failures that are arguably more common in production.

## What's measured vs. what's still ahead

To be transparent about scope: every specific number above (routing accuracy, recovery rates, cascade radius by depth, the 5.5x routing-policy effect and its ablation, decomposition fidelity) is a real measured result from a working harness in this repository, not a projection. They were run on a controlled, verifiable task chain (chosen specifically so we have unambiguous ground truth) plus one domain-flavored validation run, not yet on the full enterprise-style workflow suite (finance / HR / DevOps) with real external tools at the planned 1,800-trace scale. Validating these patterns at larger scale, across multiple model families, and on messier real-world workflows is the next phase of this work -- see `DESIGN_DOC.md` and `PAPER.md` in this repository for the full technical writeup, all source numbers, and exact reproduction commands (most of which run offline, for well under $2 of total API cost across every real-agent experiment in the project so far).

## Try it yourself

```bash
python3 -m pip install -e ".[dev]"
python3 scripts/reproduce_exp1.py        # routing mechanism result, no API key needed
python3 scripts/analyze_measured.py      # recomputes every measured number in this post from committed data
```

See `DEMO.md` for a full guided walkthrough of all four experiments, and `REPRODUCIBILITY.md` for the provenance rules behind every number cited here.
