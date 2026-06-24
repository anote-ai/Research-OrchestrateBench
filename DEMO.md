# 7/1 Midpoint Demo Runbook — OrchestraBench

A ~8-minute live walkthrough of all four experiments. **Every headline result runs
offline from committed data — no API key required.** Only the optional live agent
run (Step 4) needs `ANTHROPIC_API_KEY`, and even that has a keyless fallback.

## 0. One-time setup

```bash
# from a fresh clone of the research repo:
python3 -m pip install -e ".[dev]"
python3 -m pytest -q          # expect: 151 passed
```

## Demo flow

### 1. Experiment 1 — routing mechanism (offline)

```bash
python3 scripts/reproduce_exp1.py
```

- **Show**: the §5.1 table. Heuristic scores **100% aligned / 0% adversarial**; the
  model-driven router recovers all adversarial cases (**0% → 100%**, oracle ceiling).
- **Say**: "routing reliability is a property of the *mechanism*, not task difficulty —
  keyword matching has a structural blind spot that reasoning over intent closes."
- *(Offline baselines need no key; the LLM row runs live only if `ANTHROPIC_API_KEY` is set.)*

### 2. Experiments 2/3/4 — measured statistics (offline)

```bash
python3 scripts/analyze_measured.py
```

- **Show**: every §5.2–§5.4 number regenerated from the committed CSVs, with bootstrap CIs:
  - **Exp 2**: tool fault fully recovered (**1.0**); four latent modes fail (**0.0**), cascade 2/2.
  - **Exp 3**: cascade radius scales with depth — **1.0 → 2.9 → 5.0** at depths 3 / 5 / 7.
  - **Exp 4**: decompose vs monolithic delegation fidelity **+0.63, p<0.0001** (paired bootstrap).
- **Say**: "retry can't repair latent failures — it extends a corrupted trace; cascade
  radius grows with pipeline depth, the stages-traversed signature no existing benchmark quantifies."

### 3. Figures

Open the two committed figures:

- `figures/exp3_cascade_by_depth.png` — cascade radius vs depth (latent modes vs tool fault).
- `figures/exp2_arith_vs_domain.png` — failure-mode ordering robust across arithmetic vs loan-approval framings.

### 4. (Optional — needs key) Live real-agent run

```bash
# real run — needs ANTHROPIC_API_KEY (~$0.4, N=30); writes to /tmp to keep committed data intact
python3 -m orchestratebench.real_run --exp 2 --out /tmp/exp2_live.csv

# no key? show the same mechanism live with the mock harness (no key, no cost):
ORCHESTRATEBENCH_MOCK=1 python3 -m orchestratebench.real_run --exp 2 --out /tmp/exp2_mock.csv
```

- **Say**: "this is the harness that produced the committed CSVs — a real Claude agent over a
  verifiable dependency chain with prompt-level fault injection and exact-match grading."

## One-line summary

> "OrchestraBench measures how multi-agent systems **fail, recover, and decompose**. All four
> experiments now run on real Claude data; everything reproduces offline from a clone for under
> $1.50 of total API spend."

## If something fails

- **No API key** — skip Step 4's live run (use the `ORCHESTRATEBENCH_MOCK=1` line). Steps 1–3 are
  fully offline and cover every headline result.
- **`rich` not installed** — `scripts/run_demo.py` (a bonus FixedPolicy-vs-Heuristic comparison)
  falls back to plain print; Steps 1–3 above don't depend on it.
- **Never overwrite committed data** — any live/mock `real_run.py` call in this demo writes to
  `/tmp`, never to `data/measured/`.
