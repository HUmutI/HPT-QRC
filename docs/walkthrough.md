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

## Is it the interference?

A reviewer will ask whether the quantum part matters or whether any random nonlinear map
would do. The clean version of that question is answerable inside the model: Perceval's
partial-distinguishability parameter interpolates between `perm(|U|²)` (I = 0, classical
particles, no interference) and `|perm(U)|²` (I = 1, full interference), holding the circuit,
the encoding, the state and the readout fixed. Paired across seeds, in the infinite-shot
limit so sampling noise cannot mask the effect (`experiments/quantumness.py`).

Interference contributes nothing measurable. What carries the performance is the structure of
the feature map — the combinatorial expansion into `C(m, n)` Fock bins under a phase-encoded
unitary — not two-photon interference. Reporting this as a null result is the honest framing
and pre-empts the objection rather than inviting it.

## When would a chip actually pay?

`experiments/crossover.py` compares the cost of classically simulating the map against
running it, and solves for the transmittance at which the device wins. The answer is a
concrete number rather than a hand-wave, and at present hardware transmittance it is not met.

## Multi-timescale integration

The measured probabilities can be integrated at several leak rates at once, which costs
nothing on hardware — it is post-processing of an already-collected sequence. Whether it
helps is task-dependent (a fixed choice improved NARMA-20 by 12 % and degraded NARMA-10), so
the number of extra timescales and their geometric spacing about the primary leak are
searched rather than assumed.

## Two bugs that nearly became findings

**`matched_capacity.py` popped `feedback` from the tuned parameters** before sweeping the
ensemble width. Tasks whose search chose `feedback=False` were therefore swept with the
default `True` — a different model from the one that was tuned. The symptom was photonic
0.2116 in the tuned run against 0.9436 in the sweep, collapsing monotonically as features
were *added*. Taken at face value it would have been published as "the kernel-regime
objection is confirmed". Two conclusions flipped back in the model's favour once fixed.

**`temporal_parity` indexed forward from `t`**, so the target depended on inputs the model had
not seen. Every model scored NRMSE ≈ 1.0 and it looked like a hard benchmark rather than an
unsolvable one. `tests/test_tasks.py::test_every_task_is_learnable` now guards this.

The pattern in both cases: a structurally impossible number — error rising as capacity rises;
every model at exactly chance — is the diagnostic. Three separate bugs this project were
caught that way, and none by inspection.

**Search overfits the validation split at high trial counts.** At 250 trials NARMA-20 improved
validation from 0.191 to 0.163 while test degraded from 0.180 to 0.223. The objective now
averages over several data realisations, which is a better generalisation estimator and still
never touches the test slice.

## Open items

- **Hardware accuracy is shot-starved, not wrong.** 126 timesteps collected on `qpu:belenos`
  at 2 photons / 10 modes; hardware features correlate 0.805–0.844 with simulation across
  three shot counts, so the two-photon design is validated by measurement. But the free-tier
  credit ceiling caps coincidences at 4–20×10³ per timestep against the ~3×10⁴ simulation
  says is needed, and the combined run has 66 training rows against 65 features. The
  hardware section reports the feature-level agreement and the shot-limited prediction
  confirmed on silicon; it cannot claim an accuracy win. See `hardware/PLAN.md`.
- **Reconfiguration, not photon collection, is the hardware bottleneck** — ~14 s per timestep
  independent of shot count. This inverts the usual tradeoff: shots are nearly free in
  wall-clock terms and timesteps are the scarce resource.
- **`R²_OS` and the Patton log-RV bias correction** are specified in `docs/PROTOCOL.md` §5;
  `r2_oos` is implemented in `src/rc_protocol.py`, the Patton correction is not. Only
  relevant to S&P 500, where no model is distinguishable.
- **Walk-forward CV** has not been re-run under the new protocol. The v2 result (photonic 7th
  of 8) is retracted along with the rest of v2, so current S&P 500 evidence is the
  fixed-split result only.
- **MCS has little power at these sample sizes.** With 200–600 test points it retains almost
  every model on every task. DM-HAC is the informative test, and the paper should say so
  rather than present the MCS as if it were discriminating.

## Positioning

Defensible claims: a recurrent linear-optical reservoir beats *equally tuned* classical
feature maps on most of the eight tasks at matched feature dimension, with DM-HAC p-values to
back it; recurrence accounts for roughly half the error; accuracy is essentially unaffected
by device imperfections at measured Ascella/Belenos levels while being strongly limited by
coincidence count, which gives both a design rule and a feasibility threshold; the
shot-limited prediction was then confirmed on Belenos, with hardware features correlating
0.82 with simulation.

Claims to avoid: anything about quantum advantage; that interference is the mechanism (it
measurably is not); that the model wins on Santa Fe or S&P 500 (it does not — the ESN does);
and any hardware *accuracy* claim, since the collected run is data-starved at 66 rows against
65 features.
