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

# Exp 4 (decomposition quality, #19) uses its own metric columns — delegation fidelity etc.,
# not the cascade/recovery schema of Exp 2/3.
EXP4_CSV_FIELDS = [
    "policy", "task_id", "run", "scenario_id", "final_correct", "delegation_fidelity",
    "granularity_error", "wasted_subtasks", "produced_steps", "gold_steps", "annotator", "notes",
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
    client, n_stages: int, injection_stage: int, mode: FailureMode, policy: str, seed: int
) -> dict:
    """Run one chain with `mode` injected at `injection_stage`; measure real cascade/recovery."""
    stages = build_chain(n_stages, seed)
    latencies: List[float] = []
    succeeded: List[bool] = []
    upstream: Optional[int] = None
    use_mock = MOCK or client is None
    for s in stages:
        m = mode if s.idx == injection_stage else None
        prompt, injected = _prompt(s, upstream, m)
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
                prompt_r, _ = _prompt(s, upstream, m)  # same failure persists on retry
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
) -> List[dict]:
    modes = modes or list(FailureMode)
    rows: List[dict] = []
    for mode in modes:
        for policy in policies:
            for run in range(n_runs):
                # same chain (seed=run) across policies -> paired comparison; inject at stage 1
                r = measure_chain(client, n_stages, 1, mode, policy, seed=run)
                rows.append({
                    "policy": policy, "workflow": "arith_chain", "failure_mode": mode.value,
                    "run": run, **r,
                    "scenario_id": f"{mode.value}_{policy}_r{run}", "annotator": "yc-real",
                })
    _write_csv(rows, CSV_FIELDS, out_csv)
    return rows


def run_exp3(
    client, out_csv: str, depths: Tuple[int, ...] = (3, 5, 7), n_runs: int = 3,
    injection_stage: int = 1, modes: Optional[List[FailureMode]] = None,
    policies: Tuple[str, ...] = POLICIES,
) -> List[dict]:
    """Cascade-by-depth (#7): fix an early injection, vary chain depth, measure how
    far the failure propagates downstream. Same seed across depths shares the chain
    prefix, so deeper runs differ only by how many downstream stages exist."""
    modes = modes or list(FailureMode)
    rows: List[dict] = []
    for depth in depths:
        if injection_stage >= depth - 1:  # need >=1 downstream stage to observe cascade
            continue
        for mode in modes:
            for policy in policies:
                for run in range(n_runs):
                    r = measure_chain(client, depth, injection_stage, mode, policy, seed=run)
                    inj = r["injection_stage"]
                    rows.append({
                        "policy": policy, "depth": depth, "failure_mode": mode.value,
                        "run": run, "seed": run, **r,
                        "scenario_id": f"depth{depth}__{mode.value}__inj{inj}__run{run:04d}",
                        "annotator": "yc-real", "notes": "",
                    })
    _write_csv(rows, EXP3_CSV_FIELDS, out_csv)
    return rows


def build_composite_task(seed: int) -> dict:
    """A verifiable composite task `(a op1 b) op2 (c op3 d)` with a canonical 3-step gold
    decomposition. Clean ground truth lets us score decomposition quality (Exp 4 / #19)."""
    rng = random.Random(seed)
    a, b, c, d = (rng.randint(2, 9) for _ in range(4))
    op1, op2, op3 = (rng.choice(["+", "*", "-"]) for _ in range(3))
    f = {"+": lambda x, y: x + y, "*": lambda x, y: x * y, "-": lambda x, y: x - y}
    r1, r2 = f[op1](a, b), f[op3](c, d)
    final = f[op2](r1, r2)
    return {"expr": f"({a} {op1} {b}) {op2} ({c} {op3} {d})",
            "gold_results": [r1, r2, final], "gold_final": final, "gold_steps": 3}


def _decompose_prompt(task: dict, policy: str) -> str:
    if policy == "monolithic":
        return (f"Compute {task['expr']} in a single step. "
                f"Output exactly one line: STEP: {task['expr']} = <integer>.")
    return (f"You are a planner agent. Decompose this calculation into sequential sub-steps, one per "
            f"line, in the exact format 'STEP: <expression> = <integer>'. Resolve inner parentheses "
            f"first, then combine. Do not skip steps. Task: compute {task['expr']}.")


def _parse_steps(text: str) -> List[int]:
    """Integer results of each 'STEP: ... = N' line."""
    return [int(m) for m in re.findall(r"STEP:.*?=\s*(-?\d+)", text)]


def _mock_decompose(task: dict, policy: str) -> List[int]:
    if policy == "monolithic":
        return [task["gold_final"]]            # correct final, but one step (poor delegation)
    return list(task["gold_results"])          # full, correct decomposition


def score_decomposition(produced: List[int], task: dict) -> dict:
    gold = task["gold_results"]
    matched = sum(1 for g in gold if g in produced)
    return {
        "final_correct": bool(produced) and produced[-1] == task["gold_final"],
        "delegation_fidelity": round(matched / len(gold), 3),
        "granularity_error": abs(len(produced) - task["gold_steps"]),
        "wasted_subtasks": max(0, len(produced) - matched),
        "produced_steps": len(produced),
        "gold_steps": task["gold_steps"],
    }


def measure_decomposition(client, seed: int, policy: str) -> dict:
    task = build_composite_task(seed)
    if MOCK or client is None:
        produced = _mock_decompose(task, policy)
    else:
        resp = client.messages.create(
            model=DEFAULT_MODEL, max_tokens=256,
            messages=[{"role": "user", "content": _decompose_prompt(task, policy)}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        produced = _parse_steps(text)
    return score_decomposition(produced, task)


def run_exp4(
    client, out_csv: str, n_tasks: int = 10,
    policies: Tuple[str, ...] = ("monolithic", "decompose"),
) -> List[dict]:
    """Decomposition quality (#19): score the produced sub-task decomposition against a
    canonical gold for composite tasks, comparing a monolithic vs a decomposing policy."""
    rows: List[dict] = []
    for policy in policies:
        for run in range(n_tasks):
            r = measure_decomposition(client, seed=run, policy=policy)
            rows.append({
                "policy": policy, "task_id": run, "run": run, **r,
                "scenario_id": f"decomp_{policy}_t{run:04d}", "annotator": "yc-real", "notes": "",
            })
    _write_csv(rows, EXP4_CSV_FIELDS, out_csv)
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
    ap.add_argument("--exp", type=int, choices=(2, 3, 4), default=2)
    ap.add_argument("--out", default=None, help="default: data/measured/exp{N}_real.csv")
    ap.add_argument("--stages", type=int, default=4, help="Exp 2 chain length")
    ap.add_argument("--depths", default="3,5,7", help="Exp 3 chain depths (comma-separated)")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--tasks", type=int, default=10, help="Exp 4 number of composite tasks")
    args = ap.parse_args()
    client = _client()
    tag = "MOCK" if MOCK else f"real Claude ({DEFAULT_MODEL})"
    if args.exp == 2:
        out = args.out or "data/measured/exp2_real.csv"
        rows = run_exp2(client, out, n_stages=args.stages, n_runs=args.runs)
        print(f"[{tag}] Exp2: wrote {len(rows)} rows -> {out}")
        agg = defaultdict(list)
        for r in rows:
            agg[r["failure_mode"]].append(r["final_task_success"])
        for mode, fs in sorted(agg.items()):
            print(f"  {mode:24s} final_success_rate={sum(fs) / len(fs):.2f}  (n={len(fs)})")
    elif args.exp == 3:
        out = args.out or "data/measured/exp3_real.csv"
        depths = tuple(int(d) for d in args.depths.split(",") if d.strip())
        rows = run_exp3(client, out, depths=depths, n_runs=args.runs)
        print(f"[{tag}] Exp3: wrote {len(rows)} rows -> {out}")
        agg = defaultdict(list)
        for r in rows:
            agg[(r["failure_mode"], r["depth"])].append(r["cascade_radius"])
        for (mode, depth), cr in sorted(agg.items()):
            print(f"  {mode:24s} depth={depth} cascade_radius_mean={sum(cr) / len(cr):.2f}  (n={len(cr)})")
    else:  # exp 4 — decomposition quality
        out = args.out or "data/measured/exp4_real.csv"
        rows = run_exp4(client, out, n_tasks=args.tasks)
        print(f"[{tag}] Exp4: wrote {len(rows)} rows -> {out}")
        agg = defaultdict(lambda: defaultdict(list))
        for r in rows:
            agg[r["policy"]]["fid"].append(r["delegation_fidelity"])
            agg[r["policy"]]["fin"].append(1 if r["final_correct"] else 0)
            agg[r["policy"]]["gran"].append(r["granularity_error"])
        for policy, m in sorted(agg.items()):
            print(f"  {policy:12s} delegation_fidelity={sum(m['fid']) / len(m['fid']):.2f} "
                  f"final_correct={sum(m['fin']) / len(m['fin']):.2f} "
                  f"granularity_err={sum(m['gran']) / len(m['gran']):.2f}")


if __name__ == "__main__":
    main()
