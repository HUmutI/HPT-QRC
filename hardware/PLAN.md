# Hardware Execution Plan — HPT-QRC on Quandela Ascella

Status: **planning** (no HW code built yet). Strategy: run a cheap cost-probe first,
read the real shot/latency numbers off Ascella, then decide final scope.

---

## 1. Why this is not a drop-in

The simulated pipeline (`src/multi_qrc.py`) uses `merlin.QuantumLayer` with
`ComputationSpace.UNBUNCHED`, which returns the **exact Fock probability vector**
computed by SLOS. Real hardware returns **samples (shots)**, not probabilities,
plus loss, partial photon distinguishability, and dark counts.

So the hardware path replaces *only* the feature-extraction step:

```
simulated:  data --phase encode--> QuantumLayer (exact p-vector) --LexGrouping--> features
hardware:   data --phase encode--> RemoteProcessor sample(N shots) --histogram--> p_hat --LexGrouping--> features
```

The Ridge readout, scaler, HAR context, dataset loaders — all unchanged.

---

## 2. Chip fit (already OK)

| | Model needs | Ascella has |
|---|---|---|
| Modes | 8 (1 input + 7 memory) | 12 |
| Photons | ≤ 4 (`photon_list=[2,3,4]`) | up to 6 usable |

Architecture fits. No mode/photon reduction required for the chip itself.

**But circuit depth is a risk:** `window=10` + virtual depth ≤3 stacks up to ~13
8-mode rectangle interferometers (~28 beamsplitters each). Deep circuit → compounding
photon loss → low coincidence rate → more shots per usable sample. The cost probe
must measure the real coincidence/loss rate before committing.

---

## 3. Shot-budget math (why full reproduction is off the table)

Full benchmark, simulated, per dataset ≈ 1000 timesteps. Each timestep is a distinct
data encoding = a distinct circuit job. Reservoir count = `len(photon_list) × n_virtual_nodes`
= 3 × 3 = **9** circuits.

| Scope | Circuit configs | Shots/config | Total shots | Verdict |
|---|---|---|---|---|
| Full: 3 datasets × 5 seeds × 9 reservoirs × ~1000 steps | ~1.35e5 | ~1e4 | **~1.3e9** | infeasible on any tier |
| One dataset, 1 seed, 9 reservoirs, ~1000 steps | ~9e3 | ~1e4 | **~9e7** | likely exceeds Academic quota |
| **Validation subset:** 1 dataset, champion 1 reservoir, ~200 test steps | ~200 | ~1e3 | **~2e5** | affordable, defensible |

The subset is the target. It demonstrates *the feature map survives real shot noise* —
which is the actual scientific claim a referee cares about — without reproducing every cell.

---

## 4. Cost probe (do this FIRST)

Goal: measure real per-job latency, shot consumption, coincidence rate, and loss on
Ascella with **one tiny run**, then extrapolate the true budget.

Probe spec:
- 1 reservoir (single photon count, e.g. `[3]`, virtual depth 1 → shallow)
- 10 timesteps from NARMA10
- 1000 shots/step
- → 10 jobs, ~1e4 shots total — trivial quota cost

Record into `hardware/run_log.csv`:
- wall-clock per job (includes cloud queue time)
- shots requested vs valid coincidences returned (→ loss factor)
- empirical Fock dist vs simulated Fock dist (→ fidelity / correlation)
- total credits/shots consumed (read off Quandela dashboard before/after)

From those numbers: extrapolate cost + time for the full validation subset, then green-light it.

---

## 5. Account recommendation

**Use Academic, not Explorer.** Reasoning:
- Explorer (free) quota is sized for tutorials, not 1e5+ shot research runs.
- Academic gives larger quota + is the correct provenance to cite in a paper.

⚠️ **Verify current 2026 quotas yourself** — Quandela revises tiers; do not trust old
numbers. Check: monthly shot/credit cap, max shots/job, Ascella vs Belenos availability,
and queue priority. Confirm the cost-probe fits the free tier so you can test the
pipeline before spending Academic credits.

Apply for Academic here: cloud.quandela.com (academic program / contact form).

---

## 6. Code-change checklist (build after probe)

Planned `hardware/` contents:

- [ ] `hw_backend.py` — wrap `pcvl.RemoteProcessor("qpu:ascella", token)`; token from
      `QUANDELA_TOKEN` env var (never hardcode/commit).
- [ ] `phase_export.py` — read the fixed random reservoir phases (`t_*` params) out of a
      built `QuantumLayer` so the exact same circuit is programmed on HW. The `t` phases are
      seeded-random and never trained (`layer.eval()`), so they must be extracted, not re-sampled.
- [ ] `hw_features.py` — per timestep: set `input*` phases from encoded data, set `t_*` phases
      from `phase_export`, run `pcvl.algorithm.Sampler` for N shots, histogram into the
      UNBUNCHED Fock subspace, then apply the **same `LexGrouping(output_size, lex_out)`**
      mapping the sim uses so feature dims match.
- [ ] `run_probe.py` — the §4 cost probe; writes `run_log.csv`.
- [ ] `run_hw_subset.py` — §3 validation subset driver; **caches every job result to disk**
      (JSON per (reservoir, step)) so a re-run never re-bills shots.
- [ ] `compare_sim_hw.py` — sim vs HW Fock-dist correlation plot + forecast MSE on HW
      features vs sim features on the same test window. This is the paper figure.

Reuse existing assets:
- `src/noise_models.py` already simulates shot noise + indistinguishability — use it to
  predict the HW result *before* spending shots, and to validate the probe matches expectation.
- `src/data_loader.py`, the Ridge readout, and scaler are untouched.

---

## 7. Open decisions (resolve after probe)

- Final dataset for the subset: NARMA10 (clean synthetic, easiest to interpret) vs
  S&P 500 RV (the headline financial result — stronger for the paper if budget allows).
- Champion config on HW: single `[3]` reservoir (cheapest) vs the full `[2,3,4]` ensemble
  (matches the headline number). Depends on probe loss/cost.
- Whether to reduce `window`/virtual-depth to cut circuit depth and recover coincidence rate.
