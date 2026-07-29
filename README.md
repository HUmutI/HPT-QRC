<div align="center">
  <h1>Team Qedi 🐈‍⬛</h1>
  <p><strong>EPFL Quantum Hackathon 2026 • Quandela Challenge → Academic Paper Extension</strong></p>
  <h3>A Recurrent Linear-Optical Reservoir for Time-Series Forecasting</h3>
  <p><i>Standard reservoir-computing benchmarks, matched-capacity baselines, and robustness at measured Quandela hardware noise levels.</i></p>
</div>

> **Scope.** Accuracy results are produced by exact classical simulation of linear-optical
> Fock probabilities. Noise results are produced by Perceval's `NoiseModel` at the operating
> points measured on Quandela's Ascella and Belenos processors. Neither is a hardware
> measurement. A hardware run is implemented and rehearsed (`hardware/run_reservoir_hw.py`)
> but both QPUs have been in `maintenance` throughout this work; the section below states
> exactly what is and is not measured on a chip.

> **Concurrent and independent work.** A closely related architecture for swaption-surface
> reconstruction was independently proposed by Amanov & Azamov (arXiv:2603.10707);
> a transverse-field Ising QRC for realised-volatility forecasting is in Li, Mukhopadhyay,
> Bayat & Habibnia (arXiv:2505.13933). This work differs by (i) a recurrent formulation with
> state feedback rather than a static or windowed map, (ii) benchmarking under the standard
> input-driven protocol so numbers are comparable to the reservoir-computing literature,
> (iii) matched-capacity baselines, and (iv) a noise study at measured device parameters.

---

## 1. What changed, and why

An earlier version of this model (`src/multi_qrc.py`, kept for reproducibility) encoded a
sliding window of the series into interferometer phases and read out Fock probabilities. It
had no state. Re-examining it produced four findings that invalidated the previous results:

1. **The quantum features were not contributing.** On NARMA-10 under that code's own
   protocol, a plain ridge on the raw window scored MSE `5.24e-3` against the model's
   `5.38e-3`. Sweeps over encoding gain, circuit depth, photon number, mode count and
   feature standardisation found no configuration beating that control by more than noise.
   The input spanned only 0.58 rad of the available 2π phase range, leaving the quantum
   block with 16× less variance than the classical block appended beside it, so the ridge
   effectively ignored it.
2. **The benchmark protocol was not comparable to the literature.** The model was fed the
   target's own history; the standard NARMA benchmark drives with an exogenous input and
   requires the model to build the target's dynamics internally.
3. **There was no recurrence.** Measured linear memory capacity was exactly the window
   length — the signature of a windowed feature map, not a reservoir.
4. **The ESN baseline had no input-scaling parameter.** With one, it reaches NRMSE 0.183 on
   NARMA-10, matching the literature's ≈0.185. Previously it was far too easy to beat.

The current model adds a state that persists across timesteps:

```
e_t = g_in · W_in u_t  +  g_fb · W_fb s_{t-1}      (phases, radians)
p_t = photonic_layer(e_t)                          (unbunched Fock probabilities)
s_t = (1 − leak) · s_{t-1} + leak · p_t            (leaky integration)
```

`W_in`, `W_fb` and the interferometers are fixed random draws; only the ridge readout is
trained. **The feedback is classical and happens between shots**, so a recurrent run costs
exactly the same shot budget as the windowed model — no optical memory or fast feed-forward
is required.

---

## 2. Results

Standard input-driven protocol, 5 seeds, NRMSE (RMSE ÷ std of the target; lower is better).
The ridge penalty is selected on a validation slice **per model**, and every model passes
through the identical readout and metric (`src/rc_protocol.py`).

**Median** over 5 seeds. Medians rather than means because on the chaotic tasks the seed
distribution is heavy-tailed — a single unlucky Lorenz-63 trajectory sends one model's mean
to 3.5× its median, which would decide the ranking on one draw. Means and per-seed values are
in `results/benchmarks/all_raw.csv`.

| Model | NARMA-5 | NARMA-10 | NARMA-20 | MG (h=17) | Lorenz-63 (h=20) | S&P 500 RV |
|---|---|---|---|---|---|---|
| **Photonic (recurrent)** | 0.0152 | **0.0951** | **0.2409** | **0.0005** | 0.2810 | 0.6993 |
| Photonic (no feedback) | 0.0152 | 0.2156 | 0.4355 | 0.0005 | 0.2821 | 0.6993 |
| Echo state network | **0.0109** | 0.2794 | 0.4132 | 0.0005 | 0.3396 | **0.6823** |
| Random Fourier features | 0.0164 | 0.1759 | 0.4347 | 0.0014 | **0.2110** | 0.7274 |
| Polynomial window | 0.0158 | 0.1763 | 0.4741 | 0.0204 | 3.3614 | 0.7045 |
| Linear window (control) | 0.3801 | 0.4446 | 0.4495 | 0.6176 | 0.9187 | 0.6957 |

The photonic reservoir wins **NARMA-10 and NARMA-20**, ties on Mackey-Glass, and loses on
**NARMA-5, Lorenz-63 and S&P 500**. It wins exactly where the task demands nonlinear memory
of a stochastic drive, and not elsewhere. Diebold–Mariano with Newey–West HAC:

- **NARMA-10 and NARMA-20** — p < 0.001 against every baseline. This is the result.
- **Mackey-Glass** — p = 0.41 vs the ESN. A genuine tie; the task saturates at 5e-4.
- **NARMA-5** — the ESN wins. The task needs only 5 steps of memory, exactly the regime where
  a large tanh reservoir's long linear memory suffices and the photonic map's nonlinearity
  buys nothing.
- **Lorenz-63** — random Fourier features win on the median. On the *mean* the photonic model
  appears to win, but that inverts under a single seed and we do not claim it. Seed spread
  covers 0.06 to 11.0 across models; five seeds cannot separate them here.
- **S&P 500 RV** — p = 0.45–0.88. Nothing is distinguishable; the Hansen MCS retains all six.

**Recurrence is what carries the result.** Removing the feedback and changing nothing else
doubles the error on both NARMA tasks (p < 0.001).

### Is it the optics, or just more features?

The tuned photonic configuration uses more features than the baselines, so every family was
swept across a common dimension range (`experiments/matched_capacity.py`, 3 seeds):

| Model | dim | NRMSE |
|---|---|---|
| **Photonic** | **1381** | **0.109** |
| Photonic | 701 | 0.164 |
| Random Fourier features | 1200 | 0.188 |
| Echo state network | 1001 | 0.268 |
| Polynomial window | 679 | 0.457 |

At matched dimension the photonic map is ahead, and it stays ahead: the ESN saturates past
~2000 units while the photonic model continues to improve. A single reservoir at dim 701
already matches the ESN's best.

![capacity](results/figures/capacity_narma10.png)

### The encoding window tracks the task's own order

NARMA-N's target contains the cross-lag product `u_t · u_{t-N+1}`, which a linear-optical map
can only form if both lags sit in the *same* encoding. Sweeping `encode_window` with
everything else fixed (`experiments/ablation_encoding.py`, 3 seeds) shows the optimum
tracking N:

| Task | Task order N | Best `encode_window` | NRMSE |
|---|---|---|---|
| NARMA-5 | 5 | 8 | 0.128 |
| NARMA-10 | 10 | 12 | 0.265 |
| NARMA-20 | 20 | 25 | 0.470 |

This is a prediction the architecture makes and the data confirms, rather than a
hyperparameter found by search.

**Caveat on the feedback ablation.** In this small hardware-viable configuration the feedback
contributes only 5–11 % and slightly *hurts* on NARMA-5. The 2× effect in the benchmark table
is measured at the accuracy-tuned configuration (16 modes, photons {2,3}, depth 3, 8
reservoirs). Feedback's contribution grows with model capacity; both numbers are real and
both are reported.

### Echo state property

A reservoir's state must be a function of the input history alone, which requires
perturbations to the initial state to decay. Sweeping the feedback gain
(`experiments/esp_check.py`, 3 seeds; memory = steps for a perturbation to fall below 1 % of
its peak):

| `g_fb` | Memory (steps) | Regime | NRMSE |
|---|---|---|---|
| 0.0 | 44 | contracting | 0.281 |
| 0.3 | 72 | contracting | **0.276** |
| 0.6 | 109 | contracting | 0.277 |
| 1.0 | 352 | contracting | 0.390 |
| 1.5 | never | **ESP-violating** | 0.449 |
| ≥ 2.0 | never | **ESP-violating** | 0.65–0.73 |

Memory lengthens with feedback gain until the echo state property breaks, and the best
accuracy sits just below that transition — the standard edge-of-stability picture. The
benchmarks use `g_fb = 0.3`, inside the contracting regime.

### What it is winning with

`experiments/memory_ipc.py` measures linear memory capacity (Jaeger) and information
processing capacity (Dambre et al.):

| System | dim | Linear MC | IPC total | IPC per feature |
|---|---|---|---|---|
| Photonic (no feedback) | 132 | 10.5 | 111.9 | **0.85** |
| ESN (200) | 201 | 27.0 | 123.9 | 0.62 |
| Photonic (recurrent) | 132 | 11.1 | 69.3 | 0.53 |
| ESN (500) | 501 | 29.6 | 213.9 | 0.43 |
| Linear window (20) | 20 | 19.1 | 13.0 | 0.65 |

The photonic map packs more nonlinear capacity per feature than an ESN but carries less
linear memory. It trades memory depth for nonlinearity — which is why it wins on NARMA,
where the target is a low-order polynomial of a bounded stretch of history, and does not win
on tasks that reward long linear memory.

---

## 3. Robustness to hardware noise

This is the question a hardware collaborator actually asks, and it follows the approach
suggested by Quandela: see
[MerLin's noisy-simulation guide](https://merlinquantum.ai/0.4/user_guide/noisy_simulations.html).
Every noisy evaluation runs through Perceval's `NoiseModel` with **threshold detectors**, so
the numbers come from Quandela's own device model rather than an approximation of ours. The
configuration used is the hardware-viable one (2 photons, 12 modes, depth 1) — not the
accuracy-tuned one, which no current device could run.

Measured operating points, read live from the Quandela cloud API:

| | Ascella | Belenos |
|---|---|---|
| Clock | 80 MHz | 4.94 MHz |
| HOM visibility | 86.36 % | 82.7 % |
| g² (0) | 1.95 % | 18.2 % |
| Transmittance | 2.44 % | ~4.8 % |
| Detectors | threshold | threshold, 24 modes |

**Device imperfections cost essentially nothing:**

| Sweep | Range | NRMSE |
|---|---|---|
| Indistinguishability | V = 0.50 → 1.00 | 0.2468 → 0.2486 (flat) |
| g²(0) | 0.00 → 0.30 | 0.2479 → 0.2498 (+0.8 %) |
| Full Ascella model | all sources at once | 0.2483 vs 0.2479 noiseless |
| Full Belenos model | all sources at once | 0.2465 vs 0.2479 noiseless |

**Finite sampling is the entire story.** Pushing the sweep to 10⁶ coincidences per timestep
(`results/noise/shot_convergence.csv`, 3 seeds) locates both thresholds:

| Coincidences per timestep | NRMSE |
|---|---|
| 1 000 | 0.418 ± 0.061 |
| 10 000 | 0.399 ± 0.026 |
| 30 000 | 0.321 ± 0.012 |
| 100 000 | 0.305 ± 0.028 |
| 300 000 | 0.285 ± 0.024 |
| 1 000 000 | 0.283 ± 0.024 |
| ∞ | 0.275 ± 0.028 |

The classical control sits at 0.348. So the reservoir needs **≳3×10⁴ coincidences per
timestep to be worth using at all**, and **≳3×10⁵ to come within 4 % of its own noiseless
limit**, beyond which it is converged.

**The finding generalises.** Repeating the whole sweep on NARMA-20
(`results/noise/noise_narma20_all.csv`) gives the same ordering — device imperfection costs a
few percent, sampling costs tens of percent:

| Sweep | Range | NRMSE |
|---|---|---|
| Indistinguishability | 0.50 → 1.00 | 0.5450 → 0.5276 (−3.2 %) |
| g²(0) | 0.00 → 0.30 | 0.5276 → 0.5366 (+1.7 %) |
| Coincidences per step | 10³ → ∞ | 0.6353 → 0.4528 (−29 %) |

The dependence on indistinguishability is now visible rather than flat, but it remains an
order of magnitude smaller than the sampling effect.

A reservoir is a random feature map, and the readout is refitted on whatever the device
actually produces — so a slightly different map is still a perfectly good map. Sampling
noise is different in kind: it corrupts each feature independently on every timestep and
cannot be absorbed by the readout.

The practical consequence is a design rule that inverts the usual instinct: **optimise for
coincidence rate, not photon quality.** Rate falls as `transmittance^n`, so photon number is
the dominant lever:

| Platform | n=2 | n=3 | n=4 |
|---|---|---|---|
| Ascella | 4.8e4 /s | 1.2e3 /s | 28 /s |
| Belenos | 1.2e4 /s | 5.6e2 /s | 27 /s |

Time per timestep to reach each threshold:

| | 3×10⁴ (beats control) | 3×10⁵ (near-converged) |
|---|---|---|
| Ascella, 2 photons | 0.63 s | 6.3 s → 63 min for 600 steps |
| Belenos, 2 photons | 2.6 s | 26 s → 4.3 h for 600 steps |
| Ascella, 3 photons | 26 s | 4.3 min → 43 h for 600 steps |

Two photons on Ascella is the only combination that fits a realistic session. This is why the
hardware path uses two.
The earlier three-photon probe in this repository returned ~48 counts from 1000 requested
shots spread over 56 Fock bins, and its features correlated only 0.18–0.48 with simulation —
a direct consequence of the same scaling.

![coincidence rate](results/figures/coincidence_rate.png)

---

## 4. Is it the interference, or a random nonlinear map? (It is the map.)

The sharpest objection to any photonic reservoir is that a classical random feature map does
the same job. Random Fourier features come close to this model on several tasks, so the
question is not whether a random nonlinear map helps — it plainly does — but whether the
*quantum interference* contributes anything.

Partial distinguishability answers it cleanly, because it interpolates exactly between the two
hypotheses at otherwise identical circuit, phases, shots and readout:

- `indistinguishability = 0` — photons behave as classical distinguishable particles; the
  distribution is the permanent of `|U|²`, a positive matrix. A classical stochastic map.
- `indistinguishability = 1` — fully indistinguishable bosons; the distribution is
  `|perm(U)|²`, with interference between the `n!` photon assignments.

Run in the infinite-shot limit so sampling cannot mask the effect, paired across seeds, with an
effect required to clear both p < 0.05 and half a seed standard deviation
(`experiments/quantumness.py`):

| Task | n | V=0 | V=1 | gain | p | verdict |
|---|---|---|---|---|---|---|
| channel_eq | 2 | 0.1726 | 0.1728 | 0.999 | 0.571 | no detectable effect |
| channel_eq | 3 | 0.1725 | 0.1737 | 0.993 | 0.541 | no detectable effect |
| narma10 | 2 | 0.2465 | 0.2373 | 1.039 | 0.886 | no detectable effect |
| narma10 | 3 | 0.2225 | 0.2198 | 1.012 | 0.280 | no detectable effect |
| parity_d3 | 2 | 0.7506 | 0.7901 | 0.950 | 0.324 | no detectable effect |
| parity_d3 | 3 | 0.3140 | 0.3835 | 0.819 | 0.166 | no detectable effect |
| henon | 2 | 0.8811 | 0.8830 | 0.998 | 0.571 | no detectable effect |
| henon | 3 | 0.8811 | 0.8849 | 0.996 | 0.034 | interference *hurts* |

**Seven of eight configurations show no detectable effect**, gains scatter in both directions
(0.819–1.039), and the single significant case has interference making things marginally
worse. We therefore state plainly: **this architecture is a classical random nonlinear feature
map realised in optics.** The interferometer and Fock measurement supply a useful
high-dimensional map; the indistinguishability of the photons is not what makes it work.

One distinction matters and is visible in the table. On the parity task, **photon number**
helps a great deal — n=3 reaches 0.314 against n=2's 0.751 — while indistinguishability does
nothing. The resource being exploited is **Fock-space dimension**, `C(m,n)`, not interference.

Two consequences, one scientific and one practical:

- It explains Section 3. Insensitivity to Hong–Ou–Mandel visibility there and absence of an
  interference effect here are the same fact seen twice.
- **It lowers the hardware bar substantially.** If indistinguishability is irrelevant, the
  source does not need to be indistinguishable, and HOM visibility — usually the headline
  figure of merit for a single-photon source — is not the specification to optimise for this
  application. Transmittance is (see Section 5).

## 5. Would photonics ever beat simulating it?

The obvious objection to any photonic reservoir at these scales is that a laptop simulates it
faster. `experiments/crossover.py` answers that quantitatively rather than dodging it.
Classical simulation cost grows combinatorially in photon number; the device's cost grows as
`1/transmittance^n`. Which grows faster decides the question.

Using the measured throughput of this implementation (7.3×10⁷ complex MACs/s, single core)
against the measured platform specs, at the 3×10⁴-coincidence threshold from Section 3:

| Platform | Closest the device gets | Transmittance needed for a crossover |
|---|---|---|
| Ascella (t = 2.44 %) | 1.6×10³ slower, at n=2 | ≥ 0.105 at n=12 in 24 modes (**4× better**) |
| Belenos (t = 4.84 %) | 6.6×10³ slower, at n=2 | ≥ 0.132 at n=12 (**3× better**) |

**There is no crossover at any photon number on current hardware**, and we say so plainly.
But the requirement is now a number rather than a hope. Note the required transmittance
*falls* with photon number — 0.978 at n=2 down to 0.105 at n=12 — because classical cost
outgrows the rate penalty. High photon number is where optics could win, if loss drops far
enough to reach it. At n=2 and n=3 on Belenos the required transmittance exceeds 1, meaning
simulation is so cheap in that regime that no device could beat it.

This is why we make no computational-advantage claim anywhere, and why the useful framing for
this class of model is loss reduction rather than photon count for its own sake.

## 6. Cloud execution

The full cloud path — chunked submission, threshold-detector sampling, coincidence
post-selection, cached job results — has been exercised end to end on Quandela's cloud
platforms. 600 timesteps, 8000 requested coincidences per step, 2 photons in 12 modes,
2 reservoirs, NARMA-10:

| Platform | Protocol | Device NRMSE | Exact sim | Classical control | Lift | Feature corr. |
|---|---|---|---|---|---|---|
| `sim:ascella` | **open-loop** | **0.3474** | 0.2817 | 0.4230 | **1.217** | 0.999 |
| `sim:ascella` | replay | 0.3931 | 0.2898 | 0.4230 | 1.076 | 0.999 |
| `sim:slos` | replay | 0.4211 | 0.2898 | 0.4230 | 1.004 | 1.000 |

**The open-loop protocol is both the stronger claim and the better result.** It disables
feedback, so every timestep is independent and the run is genuinely end-to-end on the
platform with no simulation anywhere in the loop — and it still beats the classical control
by 1.22×. This is consistent with the encoding-window ablation: in the small hardware-viable
configuration the feedback contributes little and can even hurt, so giving it up costs
nothing here while removing the one caveat the replay protocol carries.

Two things this does and does not show.

**It does** validate the two-photon design decision. Device-vs-simulation feature
correlation is 0.999 (worst case 0.994) across 600 timesteps. The earlier three-photon probe
on the Belenos QPU managed only 0.18–0.48, because at `transmittance³` it collected ~48
counts per step across 56 Fock bins.

**It does** independently corroborate the noise result. Sending 30 identical phase settings
to each platform at 20 000 shots and measuring total variation against an exact local
calculation (`scripts/compare_platforms.py`) gives:

| Platform | TVD vs exact | vs perfect sampler |
|---|---|---|
| perfect sampler (multinomial, 20 000 shots) | 0.0193 ± 0.0004 | — |
| `sim:slos` | 0.0211 | +5σ |
| `sim:ascella` | 0.0288 | +26σ |

Both platforms sit slightly above the ideal-sampler null, which is expected: post-selection
discards events, so the effective sample size is below the requested shot count. That offset
is common to both, so the device-model question is settled by the **excess over `sim:slos`**:
`sim:ascella` is 0.0077 further from exact, a 21σ separation.

So `sim:ascella` does apply device noise, and the amount is small — 0.008 of displacement on
top of 0.019 of sampling error. That is the same conclusion Section 3 reaches from Perceval's
`NoiseModel` — device imperfection is a minor perturbation next to finite sampling — arrived
at independently through Quandela's own emulator.

Note that the NRMSE ordering in the table above is *not* evidence either way: `sim:ascella`
scored better than the noiseless `sim:slos`, which is seed-to-seed scatter in a downstream
regression, not a statement about the distributions. The distribution comparison is the
measurement that settles it.

Note also that the lift at 8000 shots is only 1.00–1.08, consistent with the shot-convergence
result that ~3×10⁴ coincidences per timestep are needed before the reservoir is clearly worth
using.

## 7. QPU status

**Both `qpu:ascella` and `qpu:belenos` have reported `status: maintenance` throughout this
work, so no result in this repository is a QPU measurement.** What exists:

- A validated smoke test and a 10-step, 1000-shot probe on `qpu:belenos` from an earlier
  three-photon configuration (`hardware/run_log.csv`), which is what established the
  coincidence-rate problem.
- `hardware/run_reservoir_hw.py`, which runs the current model on a QPU and reports
  hardware, simulation and classical-only readouts on identical timesteps. Rehearsed locally
  and against the cloud simulator.
- Job `96baa2b6-…` was cancelled by the platform at 307 s having completed 4 % of its
  iterations — the free tier's 5-minute cap. Submission is now chunked and each chunk is
  cached independently, so an interrupted run resumes without re-billing shots.

Because closed-loop feedback cannot be batched into a single cloud job, two protocols are
provided and both will be reported:

- `replay` — the recurrence is run in simulation to produce the phase trajectory, which is
  then replayed on hardware as one batched job. The chip evaluates the same circuit settings
  the closed-loop system would have visited, and the readout is trained on hardware-measured
  features. **The feedback path itself is simulated**, and any write-up must say so.
- `openloop` — feedback disabled, every timestep independent. Genuinely end-to-end on
  hardware with no simulation in the loop; a weaker model but a stronger claim.

Local calibration puts the viable operating point at 600 timesteps, 12 modes, 2 reservoirs,
where the reservoir gives a 1.46× improvement over the classical control.

---

## 8. Layout

```
src/
  photonic_core.py    exact boson-sampling probabilities; matches MerLin to 1e-16, 310x faster
  temporal_qrc.py     the recurrent reservoir and its readout
  rc_protocol.py      shared split, ridge-penalty selection, metrics
  tasks.py            benchmark registry with literature-standard protocols
  baselines_rc.py     ESN (with input scaling), RFF, polynomial, linear control
  noise.py            hardware specs, Perceval noise backend, coincidence-rate model
  dm_mcs.py           Diebold-Mariano (Newey-West HAC) and Hansen Model Confidence Set
  multi_qrc.py        the previous windowed model, kept so old numbers reproduce
experiments/
  tune_temporal.py    Optuna search, same budget for every model
  run_benchmarks.py   headline table with DM and MCS
  matched_capacity.py NRMSE against feature dimension
  memory_ipc.py       memory and information processing capacity
  noise_study.py      the robustness sweeps
  make_figures.py     figures
hardware/             QPU execution path (see section 4)
tests/                36 tests; the core-vs-MerLin check gates any hardware submission
```

`src/photonic_core.py` exists because MerLin's `QuantumLayer` costs ~7 ms per forward pass
regardless of batch size, and a recurrent reservoir must step one timestep at a time. These
circuits are layered, `U = A_d · D(φ_{d-1}) ··· A_1 · D(φ_0) · A_0`, so the fixed blocks can
be built once and each timestep costs a few small matrix products plus a batch of
permanents. It computes the same distribution — the test suite checks it against both MerLin
and Perceval's SLOS backend to 1e-10.

## 9. Reproducing

```bash
conda create -n quandela python=3.11 && conda activate quandela
pip install -r requirements.txt
python -m pytest tests/ -q                                    # 36 tests
python experiments/tune_temporal.py --dataset narma10 --model all --trials 150
python experiments/run_benchmarks.py --seeds 5
python experiments/matched_capacity.py --dataset narma10 --seeds 3
python experiments/noise_study.py --dataset narma10 --sweep all --seeds 3
python experiments/make_figures.py
```

Hardware runs need a Quandela token in `PCVL_CLOUD_TOKEN` (see `hardware/README.md`) and
should be rehearsed with `--local` first; every cloud job is cached, so a re-run never
re-spends shots.

## 10. What we do not claim

- **No quantum advantage, and no claim that interference contributes.** Section 4 tests that
  directly and finds no detectable effect on seven of eight configurations. The architecture is
  a classical random nonlinear feature map realised in optics.
- No computational advantage. Section 5 shows there is no crossover with classical simulation
  at any photon number on current hardware.
- No claim on S&P 500 realised volatility, where it does not win and no model is
  statistically distinguishable from any other.
- No hardware claim until a QPU leaves maintenance and the run in section 4 completes.
- The matched-capacity comparison is the honest version of the accuracy table; the headline
  configuration uses more features than the baselines and that is stated wherever it appears.
