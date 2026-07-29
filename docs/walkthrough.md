# Research log

Decisions, findings and open items. Results tables live in `README.md` and
`results/REPORT.md`; this file records *why* things are the way they are.

Superseded v1/v2 content has been removed rather than annotated — those results are
retracted (see `results/CHANGELOG.md`), and leaving them beside current numbers invites
mixing the two.

---

## 2026-07-29 — audit of the windowed model

Ran a diagnostic sweep on the then-current model (`src/multi_qrc.py`) before extending it.
Four findings, in order of how much they mattered.

**The quantum features were not doing anything.** On NARMA-10 under the repository's own
protocol, ridge on the raw window scored MSE `5.2405e-3`; the model scored `5.38e-3`. A
sweep over encoding gain (0.58–8π rad), circuit depth (1–13 interferometers), photon number
(2–3), mode count (8–24) and feature standardisation produced a best of `5.2314e-3` — a
0.17 % difference from the control, i.e. nothing.

Mechanism: NARMA-10 outputs span [0.20, 0.78], and the encoding applied them directly as
phases in radians, so the input modulated only 0.58 rad of the available 2π. Measured
consequence: quantum-feature standard deviation was 16× smaller than the raw-window block
concatenated beside it, so a single ridge penalty across both blocks effectively discarded
the quantum half. `_fit_scaler`/`_apply_scaler` existed in the class but were never called
from `fit`/`predict`.

**The protocol was not literature-comparable.** `fit(y_train)` predicted `y[t+1]` from a
window of `y`. The standard NARMA benchmark drives with the exogenous input `u` and never
shows the model the target's history. Under the standard protocol the same code scored
NRMSE 0.433 against a linear control's 0.435 — while published ESN results are ≈0.185.

**No recurrence.** `results/mc_scores.csv` recorded linear memory capacity of exactly 5.0 at
window 5 — the signature of a windowed map with no state.

**The ESN baseline was misconfigured.** It had no input-scaling parameter. Adding one and
sweeping gave NRMSE 0.176, matching the literature. The v1/v2 comparisons were therefore
against an opponent roughly 3× weaker than a correct ESN.

Conclusion: the architecture needed a state, the protocol needed replacing, and the
baselines needed fixing before any claim could be made.

## Rebuild decisions

**Feedback into the encoding, not an optical memory.** State is fed back through
`W_fb s_{t-1}` added to the encoding phases, with the leaky integration in classical
post-processing. Chosen specifically so the recurrent model costs the *same shot budget* as
the windowed one on hardware: each timestep is still one circuit configuration and one batch
of samples. Nothing here needs fast feed-forward or an optical delay line.

**An input window inside the encoding (`encode_window`).** Initially the encoding saw only
`u_t` and performance plateaued around NRMSE 0.32. NARMA-10's target contains the product
`u_t · u_{t-9}`, which the optics cannot form unless both lags appear in the same encoding.
Setting `encode_window` to the task's own order moved NRMSE to 0.279 in one step. That is a
real, interpretable dependence and deserves an ablation figure.

**A fast exact core (`src/photonic_core.py`).** MerLin's `QuantumLayer` costs ~7 ms per
forward pass *regardless of batch size* — the cost is fixed overhead, not the permanent. A
recurrent reservoir steps one timestep at a time, so that overhead would have set the budget
for the whole programme (~7 s per 1000-step sequence per reservoir, times seeds, times
Optuna trials). These circuits are layered, so the fixed blocks compose once and each step
costs a few small matrix products plus a batch of permanents: 0.022 ms, 310× faster, and
agreeing with MerLin to 1e-16.

One bug worth recording: the first version indexed the submatrix as `U[input_modes, pattern]`
instead of `U[pattern, input_modes]`. Those are *not* transposes of each other unless `U` is
symmetric, so the permanent differed. Caught by cross-checking against Perceval's SLOS
backend; `tests/test_photonic_core.py` now pins it.

**Ridge penalty selected per model.** A fixed penalty shared across models whose feature
counts differ by an order of magnitude is not a comparison. Adding validation-based selection
changed the ordering materially: before it the photonic model lost to the control; after it,
at α ≈ 30, it won.

## What the noise study says

Device imperfections barely register. Sweeping indistinguishability from V = 0.5 to 1.0
moves NRMSE between 0.2468 and 0.2486; sweeping g²(0) from 0 to 0.30 moves it 0.2479 →
0.2498. Running the full Ascella and Belenos models — every source at once, with threshold
detectors — gives 0.2483 and 0.2465 against a noiseless 0.2479.

Shot count is the whole story: 0.411 at 1 000 coincidences per timestep, 0.380 at 10 000,
0.343 at 30 000, 0.275 in the infinite-shot limit.

The interpretation is mechanical rather than surprising. A reservoir is a random feature map
and the readout is refitted on whatever the device produces, so a slightly perturbed map is
still a perfectly good map. Sampling noise is different in kind: it corrupts every feature
independently at every timestep and no readout can absorb it.

That yields a design rule that inverts the usual instinct — optimise for coincidence rate,
not photon quality — and a concrete threshold, since rate falls as `transmittance^n`.

## Open items

- **Hardware.** Both QPUs have been in `maintenance` throughout.
  `hardware/run_reservoir_hw.py` is written and rehearsed locally and against the cloud
  simulator. Protocol caveat: closed-loop feedback cannot be batched into one cloud job, so
  the `replay` protocol simulates the feedback path and replays the resulting phase
  trajectory on the chip. The `openloop` variant is fully on-device but is the weaker model.
  Both must be reported.
- **`R²_OS` and the Patton log-RV bias correction** are specified in `docs/PROTOCOL.md` §5
  and still unimplemented. Only relevant to S&P 500, where no model is distinguishable.
- **Walk-forward CV** has not been re-run under the new protocol. The v2 result (photonic 7th
  of 8) is retracted along with the rest of v2, so current S&P 500 evidence is the
  fixed-split result only.
- **Santa Fe laser** was planned as an additional standard photonic-RC benchmark and has not
  been added; the dataset is not vendored here.
- **MCS has little power at these sample sizes.** With 200–600 test points it retains almost
  every model on every task. DM-HAC is the informative test, and the paper should say so
  rather than present the MCS as if it were discriminating.

## Positioning

Defensible claims: a recurrent linear-optical reservoir beats tuned classical feature maps on
NARMA-10 and NARMA-20 at matched feature dimension, with DM-HAC p < 0.001; recurrence
accounts for roughly half the error; and accuracy is essentially unaffected by device
imperfections at measured Ascella/Belenos levels while being strongly limited by coincidence
count, which gives both a design rule and a feasibility threshold.

Claims to avoid: anything about quantum advantage, anything about S&P 500, and anything about
hardware until a chip is available.
