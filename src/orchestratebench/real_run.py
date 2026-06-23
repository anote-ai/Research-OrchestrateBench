"""Real-LLM measured run for OrchestraBench Exp 2/3 (issues #4 / #7).

The offline harness (``experiments.py`` / ``failures.py``) simulates execution
deterministically — great for mechanism testing, but the paper needs *measured*
records from a real agent. This module runs a REAL Claude agent over a
**verifiable arithmetic dependency chain** (each stage computes a value from the
previous stage's result), injects each MAST failure mode into the agent's ACTUAL
prompt, and measures REAL recovery + cascade by checking each stage's output
against the known-correct value. Output rows match the ``MeasuredExp2Record``
schema in ``measured_runs.py`` so they feed the validate + paper-table pipeline.

Cost-conscious: short arithmetic tasks, Sonnet, ~120 output tokens, one call per
stage. ``ANTHROPIC_API_KEY`` is read from the environment (never hard-coded).
Set ``ORCHESTRATEBENCH_MOCK=1`` to exercise the orchestration logic
deterministically WITHOUT calling Claude (no key, no cost) — used to validate the
pipeline before spending any budget.

Honesty / limitations (state these in the paper):
- agents are LLM-simulated (no real external tools);
- the task is a controlled arithmetic chain (clean ground truth for cascade, not
  a domain workload);
- success is exact-match on the numeric result;
- this is a small-N mechanism run, not the full 1,500-trace suite.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .failures import FailureMode

DEFAULT_MODEL = os.getenv("ORCHESTRATEBENCH_LLM_MODEL", "claude-sonnet-4-6")
MAX_TOKENS = int(os.getenv("ORCHESTRATEBENCH_MAX_TOKENS", "120"))
MOCK = os.getenv("ORCHESTRATEBENCH_MOCK") == "1"
POLICIES: Tuple[str, ...] = ("fixed", "heuristic", "retry(heuristic)")

CSV_FIELDS = [
    "policy", "workflow", "failure_mode", "run", "injection_stage",
    "injected_task_success", "final_task_success", "cascade_radius",
    "recovery_completeness", "time_to_detection_ms", "escalated",
    "escalation_latency_ms", "scenario_id", "annotator",
]

# Exp 3 column order matches measured_templates.EXP3_SKELETON_FIELDS so the
# output feeds the same validate + paper-table pipeline as the skeletons.
EXP3_CSV_FIELDS = [
    "policy", "depth", "failure_mode", "run", "seed", "injection_stage",
    "scenario_id", "injected_task_success", "final_task_success",
    "cascade_radius", "recovery_completeness", "time_to_detection_ms",
    "escalated", "escalation_latency_ms", "annotator", "notes",
]


@dataclass
class Stage:
    idx: int
    op: str          # "=", "+", "*", "-"
    operand: int
    gold: int        # correct cumulative result through this stage


def build_chain(n_stages: int, seed: int) -> List[Stage]:
    """Deterministic arithmetic chain; stage i depends on stage i-1's result."""
    rng = random.Random(seed)
    val = rng.randint(2, 9)
    stages = [Stage(0, "=", val, val)]
    for i in range(1, n_stages):
        op = rng.choice(["+", "*", "-"])
        operand = rng.randint(2, 9)
        val = {"+": val + operand, "*": val * operand, "-": val - operand}[op]
        stages.append(Stage(i, op, operand, val))
    return stages


def _prompt(
    stage: Stage, upstream_claimed: Optional[int], mode: Optional[FailureMode]
) -> Tuple[str, bool]:
    """Build the agent prompt; failure modes corrupt the REAL input. Returns (prompt, injected)."""
    if stage.idx == 0:
        return f"Start a calculation chain. Output exactly one line: RESULT={stage.operand}", False
    clean = (
        f"Previous stage result = {upstream_claimed}. "
        f"Compute {upstream_claimed} {stage.op} {stage.operand}. "
        f"Output exactly one line: RESULT=<integer>."
    )
    if mode is None:
        return clean, False
    rng = random.Random(stage.idx * 31 + 7)
    if mode == FailureMode.CONTEXT_POLLUTION:
        # corrupt the upstream value the agent sees, with NO hint that it is wrong
        bogus = (upstream_claimed or 0) + rng.choice([-7, -5, -3, 3, 5, 7])
        return (
            f"Previous stage result = {bogus}. "
            f"Compute {bogus} {stage.op} {stage.operand}. Output RESULT=<integer>."
        ), True
    if mode == FailureMode.TOOL_INVOCATION_ERROR:
        # tool down, but a capable agent can do the arithmetic itself (retryable)
        return (
            f"[CALCULATOR TOOL ERROR — unavailable] Previous result = {upstream_claimed}. "
            f"Compute {upstream_claimed} {stage.op} {stage.operand} by hand. Output RESULT=<integer>."
        ), True
    if mode == FailureMode.AMBIGUOUS_DELEGATION:
        # the operation is left unspecified — the agent must guess which op to apply
        return (
            f"Previous result = {upstream_claimed}. Apply the appropriate arithmetic operation "
            f"with {stage.operand}. Output RESULT=<integer>."
        ), True
    if mode == FailureMode.CONFLICTING_OUTPUTS:
        # two contradictory upstream values, NO indication which is correct
        wrong = (upstream_claimed or 0) + rng.choice([-6, -4, 4, 6])
        return (
            f"Two upstream agents disagree on the previous result: one reports {upstream_claimed}, "
            f"the other reports {wrong}. Compute {stage.op} {stage.operand}. Output RESULT=<integer>."
        ), True
    if mode == FailureMode.PREMATURE_ACTION:
        # force action before the upstream result is available
        return (
            f"The previous stage has NOT reported its result yet. Still, compute {stage.op} "
            f"{stage.operand} applied to it now. Output RESULT=<integer>."
        ), True
    return clean, False


ROLES = ("Intake Officer", "Risk Analyst", "Compliance Officer", "Approval Manager",
         "Audit Reviewer", "Settlement Clerk", "Portfolio Lead")
_ACTION = {
    "+": "add the processing fee of {n} (i.e., compute {u} + {n})",
    "*": "apply the risk multiplier of {n} (i.e., compute {u} * {n})",
    "-": "deduct the {n} adjustment (i.e., compute {u} - {n})",
}


def _domain_prompt(
    stage: Stage, upstream_claimed: Optional[int], mode: Optional[FailureMode]
) -> Tuple[str, bool]:
    """Domain-grounded (loan-approval) framing of the SAME verifiable computation, each stage
    handled by a role agent. Identical numeric ground truth as `_prompt`; only the framing is
    domain / multi-role. Failure modes are injected exactly as in `_prompt` (addresses the
    construct-validity question: do the cascade/recovery signatures survive a realistic,
    role-played workflow, not just abstract arithmetic?)."""
    role = ROLES[stage.idx % len(ROLES)]
    if stage.idx == 0:
        return (f"You are the {role} opening a new loan-approval case. Record the principal "
                f"amount. Output exactly one line: RESULT={stage.operand}"), False
    act = _ACTION[stage.op].format(n=stage.operand, u=upstream_claimed)
    cap = act[0].upper() + act[1:]
    clean = (f"You are the {role} in a loan-approval pipeline. The prior stage reported a working "
             f"value of {upstream_claimed}. {cap}. Output exactly one line: RESULT=<integer>.")
    if mode is None:
        return clean, False
    rng = random.Random(stage.idx * 31 + 7)
    if mode == FailureMode.CONTEXT_POLLUTION:
        bogus = (upstream_claimed or 0) + rng.choice([-7, -5, -3, 3, 5, 7])
        act_b = _ACTION[stage.op].format(n=stage.operand, u=bogus)
        return (f"You are the {role}. The prior stage reported a working value of {bogus}. "
                f"{act_b[0].upper()}{act_b[1:]}. Output RESULT=<integer>."), True
    if mode == FailureMode.TOOL_INVOCATION_ERROR:
        return (f"You are the {role}. [The risk-scoring system is unavailable.] Prior value "
                f"{upstream_claimed}. {cap} by hand. Output RESULT=<integer>."), True
    if mode == FailureMode.AMBIGUOUS_DELEGATION:
        return (f"You are the {role}. Prior value {upstream_claimed}. Apply the standard adjustment "
                f"with factor {stage.operand}. Output RESULT=<integer>."), True
    if mode == FailureMode.CONFLICTING_OUTPUTS:
        wrong = (upstream_claimed or 0) + rng.choice([-6, -4, 4, 6])
        return (f"You are the {role}. Two analysts disagree on the prior value: one reports "
                f"{upstream_claimed}, the other {wrong}. Apply the {stage.op} {stage.operand} step. "
                f"Output RESULT=<integer>."), True
    if mode == FailureMode.PREMATURE_ACTION:
        return (f"You are the {role}. The prior stage has NOT submitted its value yet. Still, perform "
                f"the {stage.op} {stage.operand} step now. Output RESULT=<integer>."), True
    return clean, False


def _parse(text: str) -> Optional[int]:
    m = re.search(r"RESULT\s*=\s*(-?\d+)", text)
    return int(m.group(1)) if m else None


HARD_MODES = (
    FailureMode.AMBIGUOUS_DELEGATION,
    FailureMode.PREMATURE_ACTION,
    FailureMode.CONTEXT_POLLUTION,
    FailureMode.CONFLICTING_OUTPUTS,
)


def _call_claude(client, prompt: str) -> Tuple[Optional[int], float]:
    t0 = time.time()
    resp = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    return _parse(text), (time.time() - t0) * 1000.0


def _mock_value(
    stage: Stage, prev_gold: Optional[int], upstream: Optional[int],
    mode: Optional[FailureMode], injected: bool,
) -> int:
    """Deterministic mock for logic validation (no Claude). An injected hard mode
    corrupts the stage; a wrong upstream cascades (downstream computes on the bad
    value). Retryable tool faults at the injected stage stay correct."""
    upstream_wrong = stage.idx > 0 and upstream != prev_gold
    if (injected and mode in HARD_MODES) or upstream_wrong:
        return stage.gold + 1
    return stage.gold


def measure_chain(
    client, n_stages: int, injection_stage: int, mode: FailureMode, policy: str, seed: int,
    prompt_fn=_prompt,
) -> dict:
    """Run one chain with `mode` injected at `injection_stage`; measure real cascade/recovery."""
    stages = build_chain(n_stages, seed)
    latencies: List[float] = []
    succeeded: List[bool] = []
    upstream: Optional[int] = None
    use_mock = MOCK or client is None
    for s in stages:
        m = mode if s.idx == injection_stage else None
        prompt, injected = prompt_fn(s, upstream, m)
        prev_gold = stages[s.idx - 1].gold if s.idx > 0 else None
        if use_mock:
            val, lat = _mock_value(s, prev_gold, upstream, m, injected), 0.0
        else:
            val, lat = _call_claude(client, prompt)
        # retry(heuristic): one retry on the injected stage if it failed. Realistic —
        # the failure is STILL present on retry, so transient/tool faults may clear
        # while latent/semantic ones persist.
        if policy == "retry(heuristic)" and s.idx == injection_stage and val != s.gold:
            if use_mock:
                val = s.gold if m == FailureMode.TOOL_INVOCATION_ERROR else val
                lat2 = 0.0
            else:
                prompt_r, _ = prompt_fn(s, upstream, m)  # same failure persists on retry
                val, lat2 = _call_claude(client, prompt_r)
            lat += lat2
        succeeded.append(val == s.gold)
        latencies.append(lat)
        upstream = val  # downstream builds on the (possibly wrong) CLAIMED value -> real cascade
    inj = min(injection_stage, len(stages) - 1)
    post = succeeded[inj + 1:]
    cascade = sum(1 for ok in post if not ok)
    recovery = round((len(post) - cascade) / len(post), 3) if post else 1.0
    final_ok = succeeded[-1]
    ttd = round(sum(latencies[inj:]) if not final_ok else latencies[inj], 1)
    escalated = not final_ok
    return {
        "injection_stage": inj,
        "injected_task_success": succeeded[inj],
        "final_task_success": final_ok,
        "cascade_radius": cascade,
        "recovery_completeness": recovery,
        "time_to_detection_ms": ttd,
        "escalated": escalated,
        "escalation_latency_ms": ttd if escalated else 0.0,
    }


def _write_csv(rows: List[dict], fields: List[str], out_csv: str) -> None:
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def run_exp2(
    client, out_csv: str, n_stages: int = 4, n_runs: int = 3,
    modes: Optional[List[FailureMode]] = None, policies: Tuple[str, ...] = POLICIES,
    domain: bool = False,
) -> List[dict]:
    modes = modes or list(FailureMode)
    pf = _domain_prompt if domain else _prompt
    wf = "loan_approval" if domain else "arith_chain"
    rows: List[dict] = []
    for mode in modes:
        for policy in policies:
            for run in range(n_runs):
                # same chain (seed=run) across policies -> paired comparison; inject at stage 1
                r = measure_chain(client, n_stages, 1, mode, policy, seed=run, prompt_fn=pf)
                rows.append({
                    "policy": policy, "workflow": wf, "failure_mode": mode.value,
                    "run": run, **r,
                    "scenario_id": f"{wf}_{mode.value}_{policy}_r{run}", "annotator": "yc-real",
                })
    _write_csv(rows, CSV_FIELDS, out_csv)
    return rows


def run_exp3(
    client, out_csv: str, depths: Tuple[int, ...] = (3, 5, 7), n_runs: int = 3,
    injection_stage: int = 1, modes: Optional[List[FailureMode]] = None,
    policies: Tuple[str, ...] = POLICIES, domain: bool = False,
) -> List[dict]:
    """Cascade-by-depth (#7): fix an early injection, vary chain depth, measure how
    far the failure propagates downstream. Same seed across depths shares the chain
    prefix, so deeper runs differ only by how many downstream stages exist."""
    modes = modes or list(FailureMode)
    pf = _domain_prompt if domain else _prompt
    rows: List[dict] = []
    for depth in depths:
        if injection_stage >= depth - 1:  # need >=1 downstream stage to observe cascade
            continue
        for mode in modes:
            for policy in policies:
                for run in range(n_runs):
                    r = measure_chain(client, depth, injection_stage, mode, policy, seed=run, prompt_fn=pf)
                    inj = r["injection_stage"]
                    rows.append({
                        "policy": policy, "depth": depth, "failure_mode": mode.value,
                        "run": run, "seed": run, **r,
                        "scenario_id": f"depth{depth}__{mode.value}__inj{inj}__run{run:04d}",
                        "annotator": "yc-real", "notes": "loan_approval" if domain else "",
                    })
    _write_csv(rows, EXP3_CSV_FIELDS, out_csv)
    return rows


def _client():
    if MOCK:
        return None
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("ANTHROPIC_API_KEY not set (or run with ORCHESTRATEBENCH_MOCK=1)")
    from anthropic import Anthropic  # lazy import; not needed in mock mode
    return Anthropic(api_key=key)


def main() -> None:
    ap = argparse.ArgumentParser(description="Real-LLM Exp 2/3 measured run (arithmetic chain).")
    ap.add_argument("--exp", type=int, choices=(2, 3), default=2)
    ap.add_argument("--out", default=None, help="default: data/measured/exp{N}_real.csv")
    ap.add_argument("--stages", type=int, default=4, help="Exp 2 chain length")
    ap.add_argument("--depths", default="3,5,7", help="Exp 3 chain depths (comma-separated)")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--domain", action="store_true", help="domain-grounded loan-approval workflow (role agents) instead of bare arithmetic")
    args = ap.parse_args()
    client = _client()
    tag = "MOCK" if MOCK else f"real Claude ({DEFAULT_MODEL})"
    if args.exp == 2:
        out = args.out or ("data/measured/exp2_domain_real.csv" if args.domain else "data/measured/exp2_real.csv")
        rows = run_exp2(client, out, n_stages=args.stages, n_runs=args.runs, domain=args.domain)
        print(f"[{tag}] Exp2: wrote {len(rows)} rows -> {out}")
        agg = defaultdict(list)
        for r in rows:
            agg[r["failure_mode"]].append(r["final_task_success"])
        for mode, fs in sorted(agg.items()):
            print(f"  {mode:24s} final_success_rate={sum(fs) / len(fs):.2f}  (n={len(fs)})")
    else:
        out = args.out or ("data/measured/exp3_domain_real.csv" if args.domain else "data/measured/exp3_real.csv")
        depths = tuple(int(d) for d in args.depths.split(",") if d.strip())
        rows = run_exp3(client, out, depths=depths, n_runs=args.runs, domain=args.domain)
        print(f"[{tag}] Exp3: wrote {len(rows)} rows -> {out}")
        agg = defaultdict(list)
        for r in rows:
            agg[(r["failure_mode"], r["depth"])].append(r["cascade_radius"])
        for (mode, depth), cr in sorted(agg.items()):
            print(f"  {mode:24s} depth={depth} cascade_radius_mean={sum(cr) / len(cr):.2f}  (n={len(cr)})")


if __name__ == "__main__":
    main()
