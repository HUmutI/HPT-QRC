# Hardware execution — plan and outcome

**Status: complete.** 126 timesteps collected on `qpu:belenos` across eight submitted jobs.
Results in `README.md` §7 and `hardware/results/qpu_combined.json`. This file records the
plan, what it cost, and what the execution taught us — the operational findings are the part
worth reading before repeating any of it.

## 1. What the first attempt taught us

A three-photon, eight-mode configuration run on Belenos in July 2026:

| Observation | Value |
|---|---|
| Requested shots per timestep | 1 000 |
| Raw counts returned | 39–63 |
| Counts surviving unbunched post-selection | 21–36, over 56 Fock bins |
| Correlation of hardware features vs simulation | **0.18–0.48** |

Sampling noise, not calibration: the `transmittance^n` scaling of n-fold coincidences. The fix
is fewer photons, not more shots.

## 2. Design consequences, and how they turned out

| Choice | Reason | Measured outcome |
|---|---|---|
| **2 photons**, not 3 | rate rises by `1/transmittance` | 390× more counts; correlation 0.33 → 0.805–0.844 |
| 10 modes | enough Fock bins, good per-bin statistics | 45 bins, drop rate 0.14–0.15 |
| depth 1 | loss compounds with depth | — |
| `max_shots_per_call` ~2e6 | old default `10 × shots` starved post-selection | avoids the credit auto-adjustment entirely |
| Sliced submission | 5-minute execution cap | ~20 timesteps per job |

## 3. The constraint we did not anticipate

Per-timestep wall time is **~14 s regardless of shot count** — 13.7 s at 2×10⁴ shots, 14.4 s at
5×10³, 15.2 s at 2×10⁴. The chip is limited by thermo-optic phase-shifter settling between
circuit configurations, not by collecting photons.

This inverts the tradeoff that holds in simulation. There, shots are the expensive axis and you
buy timesteps by lowering them. On hardware, shots are nearly free in wall-clock terms and
**timesteps are the scarce resource**: any five-minute job yields ~20 of them whatever you
request. A reservoir needing one configuration per timestep is latency-bound by reconfiguration.

## 4. Budget reality

Credits, not time, cap the shot budget. Requests were auto-reduced repeatedly (1.2e9 → 1.18e8,
1.8e8 → 1.01e8, 4.8e8 → 8.6e7). At two photons this still delivered 4–20×10³ coincidences per
timestep — six to eight times below the ~3×10⁴ that simulation says is needed to beat the
classical control. **A hardware accuracy result in the winning regime is not affordable on the
free tier**; the ask would be roughly 30× the shot budget for one 126-step run.

## 5. Operational findings

- **Platform status is not an availability signal.** Both QPUs reported `maintenance` while the
  API accepted and queued jobs. Only attempting a submission is informative.
- **HTTP 400 is ambiguous.** Quandela returns it both when creating a job with an auto-reduced
  shot budget and when rejecting one. Requesting a budget *below* the granted ceiling avoids
  the adjustment and returns a clean job id. Submitting blind on a 400 orphaned several jobs.
- **Cancelled jobs still return data.** A job killed by the five-minute cap returns the
  iterations it finished; discarding them throws away real measurements.
- **Slices must share one phase trajectory.** `total_steps` sets `n_train`, which sets the input
  scaler, which sets the encoding phases. `combine_qpu_slices.py` refuses to stitch across
  trajectories.
- **One queued job at a time** on the free tier; a second submission returns HTTP 400.

## 6. Run order

```bash
python -m pytest tests/ -q                                   # core must match MerLin
python hardware/compare_sim_local.py                         # zero-cost gate
python hardware/submit_qpu_run.py --platform qpu:belenos --steps 144 \
       --slice-start 0 --slice-len 20 --shots 4000 --max-shots-per-call 2000000
python hardware/fetch_qpu_run.py                             # harvest, any time, any machine
python hardware/combine_qpu_slices.py                        # stitch and fit the readouts
```

Submit and harvest are separate because the queue runs to hours and a laptop will not stay
awake for it. Size from the *measured* coincidence rate, not the published transmittance.
