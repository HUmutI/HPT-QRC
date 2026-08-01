# HPT-QRC Experimental Protocol (Pre-Registered)

**Status:** Locked for the Phase-1/2 benchmark runs.
**Version:** 1.0 — 2026-05-12.
**Scope:** This document fixes the experimental protocol *before* the upgraded benchmark runs are executed. Any deviation must be documented as a `DEVIATION` block at the bottom of this file with rationale and date.

The purpose is to remove post-hoc degrees of freedom (cherry-picking, dataset-leakage shortcuts, baseline under-tuning) that would invalidate reviewer scrutiny.

---

## 1. Models under test

### Primary model
- `HPT_QRC_Multi` (`multi_qrc.py`) at the current best config:
  - `window = 10` (override per-dataset only if pre-registered window sweep favours otherwise; for S&P 500 RV we keep `window ∈ {3, 5, 10}` as a *sensitivity* axis, not a model selector).
  - `photon_list = [2, 3, 4]` (heterogeneous photon-number ensemble).
  - `n_virtual_nodes = 3`.
  - `lex_out = 10` per circuit.
  - `ridge_alpha = 1e-4` unless walk-forward CV picks otherwise per fold (the CV-selected α is logged).

### Baselines
| Model | Tuning | Notes |
|---|---|---|
| AR(1), AR(3) | Default | Closed-form |
| ARMAX(1,0,0) | Default | Exogenous lag included for X variants |
| HAR | Default (1/5/22 lag) | Corsi 2009 |
| HARX | Default | HAR + exogenous lag-1 (VIX) |
| ESN | **Optuna 100 trials** over `res_size ∈ {50, 100, 200, 500, 1000, 2000}`, `leak ∈ {0.1, 0.3, 0.5, 0.9}`, `spectral_radius ∈ {0.6, 0.9, 1.1}`, `ridge_alpha ∈ {1e-4 … 1e0}` log-spaced | Best per dataset cached in `results/esn_best_configs.json` |
| LSTM | **Optuna 100 trials** over `layers ∈ {1, 2, 3}`, `hidden ∈ {32, 64, 128}`, `lr ∈ {1e-4, 1e-3, 1e-2}` log, `dropout ∈ {0, 0.2, 0.4}`. Adam, batch 32, max 200 epochs, early stopping patience 10 on val loss | Best per dataset cached in `results/lstm_best_configs.json` |
| RFF + Ridge | **Optuna 100 trials** over `D_RFF ∈ {matched-to-QRC, 0.5×, 2×}`, `gamma ∈ {1e-3 … 1e2}` log, `ridge_alpha` log-spaced | Random Fourier Features for the Gaussian kernel; matched-dim variant is the headline comparator |
| Classical-Ridge ablation | `window=5`, `ridge_alpha=10.0` | Same HAR-style context, no quantum features (existing in `classical_baselines.py`) |

All "X" variants append exogenous regressors (Fama–French markers / VIX) with appropriate lag (see §4).

### Equal compute budget rule
Every tuned baseline gets **100 Optuna trials** per dataset; HPT-QRC hyperparameters are *not* tuned per dataset (we report at the v2 default), so the comparison is *tuned baselines vs. fixed-config HPT-QRC* — biased *against* HPT-QRC, which is the conservative reporting direction.

---

## 2. Datasets

| Dataset | Source | Size | Target | Notes |
|---|---|---|---|---|
| NARMA-10 | `data_loader.load_narma10` | 1000 (800 train / 200 test) | univariate next-step | Seeded RNG; 5 seeds |
| Mackey-Glass | `data_loader.load_mackey_glass(τ=17)` | 1000 (800 / 200) | univariate 17-step-ahead | 5 seeds |
| S&P 500 Realised Volatility | Oxford-Man / `data_loader.load_sp500`, 5-min RV | 2010-01-04 → 2024-12-31 | log-RV next-day | Walk-forward (§3); exogenous = lagged Fama-French + lagged VIX |
| VIX | yfinance | 6,288 daily samples | next-day VIX | Walk-forward (§3); exogenous = none |

Versions of all datasets are checked into `data/` and SHA-256 hashes are recorded in `results/dataset_hashes.txt`.

---

## 3. Splits

### Synthetic (NARMA, Mackey-Glass)
- 5 seeds, each: fixed 80/20 train/test split, 100-step burn-in discarded for the readout fit.

### Financial (RV, VIX) — walk-forward
- Window: train 5 years, validation 6 months, test 6 months.
- Step: slide window forward by 6 months.
- This yields ≥4 test folds per dataset across 2010–2024.
- Per fold: scaler fit on train only; Optuna tuning on train+val (val used for early stopping / α selection); final metric reported on test.
- Reported result: median + IQR across folds, plus per-fold table.
- Fold definitions stored in `results/walk_forward_folds_<dataset>.json`.

### No fixed leaderboard split
For RV/VIX we *do not* report a single fixed-split number as a headline. The walk-forward median across folds is the headline.

---

## 4. Leakage controls

- Scaler (`HPT_QRC_Multi._fit_scaler`, `multi_qrc.py:96-106`) is fit on the **training fold only** and applied to val/test — verified.
- HAR context (`multi_qrc.py:118-134`) uses only `t-1, t-2, …, t-22` — no `t+0` info. Verified at the loop bound `for t in range(pad, len(padded))` reading `padded[:t]`.
- For HARX/QRC-X, exogenous regressors (VIX, Fama-French) are lagged **1 day**. VIX is a 30-day forward implied-vol index; using contemporaneous VIX as a feature for next-day RV would be leakage. The lag is applied in `data_loader.load_sp500` exogenous build.
- log-RV bias correction (Patton 2011): when the target is log-RV, the back-transform is `exp(ŷ + σ̂²/2)` with σ̂² estimated on the training residuals. Applied in the QLIKE computation, not in the model.
- Seeds: every randomised component sets `numpy`, `torch`, and `random` seeds from a single per-run integer; the seed is logged.

---

## 5. Metrics

| Metric | Where | Note |
|---|---|---|
| MSE | All datasets | Sanity metric |
| MAE | All datasets | Robustness |
| NRMSE | NARMA, MG | Normalised by `std(y_test)` |
| QLIKE | RV, VIX | `y/ŷ - log(y/ŷ) - 1`; bias-corrected back-transform on log-RV |
| R²_OS | RV, VIX | Campbell–Thompson out-of-sample R² vs. historical mean |
| Linear MC | Memory analysis | Jaeger 2001 |
| IPC (degrees 1–4) | Memory analysis | Dambre et al. 2012, Hermite basis |
| Diebold–Mariano stat + p | All datasets | **Newey–West HAC**, Bartlett kernel, automatic bandwidth `m = ⌊4(T/100)^(2/9)⌋` (Andrews 1991) |
| Hansen MCS | All datasets, per metric | Stationary bootstrap, B = 5000, block length 20, `α = 0.10` for the survivor set |

All metrics are reported as **median ± IQR across walk-forward folds** for RV/VIX and **mean ± std across seeds** for NARMA/MG.

---

## 6. Statistical inference

### DM test
- Loss differential: `d_t = L_t(model A) - L_t(model B)` per timestep with the metric of interest.
- Variance: Newey–West HAC with Bartlett kernel and automatic bandwidth `m = ⌊4(T/100)^(2/9)⌋`, capped at `T-1`. Two-sided test (we have no directional prior model-vs-model).
- Output: lower-triangle = DM stat, upper-triangle = p-value, in `results/<dataset>_DM_<metric>_HAC.csv`.

### Hansen Model Confidence Set (MCS)
- Implementation following Hansen, Lunde & Nason (Econometrica 2011): iterative t-max elimination with bootstrapped quantiles.
- Stationary bootstrap (Politis & Romano 1994), B = 5000 resamples, expected block length 20.
- Output: per-metric per-dataset survivor table at `α = 0.10` in `results/<dataset>_MCS_<metric>.csv`, plus elimination order.
- No multiple-testing correction across datasets is applied; results are reported per-dataset.

### No selective reporting
- Every Optuna trial is logged to `results/optuna_logs/`. The best-by-val-loss configuration is the reported one.
- Every photon configuration in the ablation is reported, not just the best.
- Every walk-forward fold is reported, not just the average.

---

## 7. Ablations (locked)

### Architecture ablation
- `photon_list ∈ {[2], [3], [4], [2,3], [3,4], [2,4], [2,3,4]}` at **matched total output feature dimension** (cap `lex_out` per config so D_total is identical; pad with zeros if needed).
- Mode count `m ∈ {4, 6, 8, 10, 12}` at fixed `n_photons = 3`.
- Window `w ∈ {3, 5, 10, 15}`.

### Noise ablation
- Shot count `n_shots ∈ {10², 10³, 10⁴, 10⁵, ∞}`.
- Indistinguishability `V ∈ {0.7, 0.9, 1.0}`.

### Memory ablation
- ESN size `res_size ∈ {50, 100, 200, 500, 1000, 2000}` with the Optuna grid in §1.

---

## 8. Reproducibility

- Python 3.11, CPU. Versions pinned in `requirements.txt`.
- All seeds, configs, datasets, and output CSVs committed to the repo.
- Each benchmark script writes a `run_manifest.json` (timestamp, git hash, seed list, config) alongside its outputs.

---

## 9. What we will NOT claim

- "Quantum advantage" of any kind.
- "Outperforms LSTM" without specifying the matched-dimension RFF+Ridge column (which closes the regularisation-vs-deep-learning gap).
- "50× memory" or any order-of-magnitude claim against ESN without the matched-dim ESN sweep in this protocol.
- "First photonic QRC for finance" — preceded by Amanov & Azamov (arXiv:2603.10707) on photonic swaption surfaces and by Li et al. (arXiv:2505.13933) on Ising-QRC realised volatility.
- "Photonic" without a "(Perceval-simulated)" qualifier until the Quandela hardware execution is in.

---

## Deviation log

**DEVIATION 2026-07-29 — protocol v2.0. The v1.0 protocol above is superseded.**

Auditing the v1.0 results found that the model under test did not beat a plain ridge on a
window of the raw series, and that the benchmark protocol was not comparable to published
numbers. Both are addressed by changes large enough that v1.0 results cannot be carried
forward. Everything below is a deliberate, documented break from the pre-registration.

1. **Model changed.** The windowed feature map is replaced by a recurrent reservoir with
   state feedback (`src/temporal_qrc.py`). The v1.0 model had no state; its measured linear
   memory capacity equalled its window length. The old model is retained as
   `src/multi_qrc.py` so v1/v2 numbers reproduce.

2. **Task protocol changed.** NARMA is now driven by its exogenous input, the standard
   reservoir-computing protocol. v1.0 fed the model the target's own history, which is a
   materially easier task and makes the numbers incomparable to the literature. The old
   variant is retained as `narma10_autoregressive`.

3. **Primary metric changed** to NRMSE (RMSE ÷ std of target), the metric the reservoir
   literature reports, so results can be placed against published values directly. The
   §5 QLIKE inconsistency (summed in some scripts, averaged in others, so cross-file numbers
   were not comparable) is resolved by not using QLIKE as a headline metric; it remains
   available in `src/dm_mcs.py` for the volatility task.

4. **Ridge penalty is now selected per model** on a validation slice rather than fixed at
   `1e-4`. A shared penalty favours whichever feature count it happens to suit, across
   models whose dimensions differ by an order of magnitude.

5. **Baseline corrected.** The v1.0 ESN had no input-scaling parameter. With one it reaches
   NRMSE 0.183 on NARMA-10, matching the literature's ≈0.185. Several v1.0 comparisons were
   therefore against a misconfigured opponent.

6. **Datasets added:** NARMA-5, NARMA-20, Lorenz-63. Lorenz at horizon 1 was dropped as a
   benchmark — a tuned ESN and a plain ridge both score ~1e-4, so it discriminates nothing;
   horizon 20 is used instead. S&P 500 RV is **retained despite being the dataset where the
   model does not win**, per §9.

7. **Ablation added:** feedback on/off at otherwise identical configuration, and a
   matched-capacity sweep (`experiments/matched_capacity.py`) reporting NRMSE against
   feature dimension for every model family. The latter is required for any claim about the
   photonic feature map, since the tuned configuration uses more features than the baselines.

8. **Noise study added** (not in v1.0): Perceval `NoiseModel` with threshold detectors at
   the Ascella and Belenos operating points read from the cloud API.

Deviations from v1.0 that remain, and are not fixed:

- **Optuna budget is 150 trials, not the pre-registered 100** — raised, applied equally to
  every model including all baselines.
- **Hansen MCS uses B = 5000** as pre-registered. Note the MCS retains almost every model on
  every task here; with 200–600 test points it has little power, so DM-HAC is the informative
  test and both are reported.
- **`R²_OS` and the Patton log-RV bias correction are still not implemented.** They were
  specified in §5 and are still missing. Any S&P 500 claim should be read with that in mind —
  though as no model is distinguishable there, no claim is made.
- **Artifacts promised in §8 (`dataset_hashes.txt`, `optuna_logs/`, `run_manifest.json`,
  `walk_forward_folds_*.json`) do not exist.** Seeds are fixed and datasets are generated
  deterministically from them, so runs are reproducible, but the manifest files were never
  produced.

---

**DEVIATION 2026-08-01 — protocol v2.1.** Changes to the *search* procedure and to two
claims, all made after v2.0 results were in hand and all documented here rather than folded
in silently.

9. **The search objective averages over several estimates, not one validation split.**
   Selecting on a single split overfits it once the trial count is high, measured twice:
   NARMA-20 at 250 trials improved validation 0.191 → 0.163 while test degraded 0.180 →
   0.223; Santa Fe from 100 to 300 trials improved validation 0.0318 → 0.0294 while test
   degraded 0.0601 → 0.0705. Synthetic tasks now average the objective over fresh data
   realisations (`--objective-seeds`); recorded series, which cannot be resampled, use
   rolling-origin validation windows (`--val-blocks`). The test slice is untouched by both.

   *Sequence, stated because it matters:* the weakness was demonstrated on NARMA-20 before
   Santa Fe was re-run, but the decision to add rolling-origin validation came **after**
   seeing Santa Fe's test number degrade. Both sets of numbers are in `results/CHANGELOG.md`.

10. **Trial budgets differ between datasets** (100–300), and `--max-dim` prunes photonic
    trials projected above 30 000 features. Both are compute budgets, not modelling choices:
    the top of the search space reaches ~4×10⁵ features, which on a 4000-step series is a
    12 GB feature matrix, and one Santa Fe search spent five hours at 7 % CPU thrashing.
    Within every dataset all five models receive the identical budget, which is the
    comparison §1's equal-compute rule exists to protect.

11. **Multi-timescale integration added** to the photonic search space (`n_scales`,
    `scale_ratio`): the measured probabilities are integrated at several leak rates at once.
    This is post-processing of an already-collected sequence and costs no extra shots.

12. **The architecture ablation in §7 is replaced by a gain sweep.** The locked ablation was
    feedback on/off at otherwise identical configuration. That comparison turned out to be
    uninformative in exactly the cases it mattered: the search chooses `feedback=False`
    outright on 8 of 11 datasets, so the ablation and the model are the same object there and
    the measured gap is zero by construction. Where feedback is kept, the gain sits at the
    floor of its range (0.010–0.016 against a ceiling of 3.0).
    `experiments/feedback_strength.py` sweeps the gain instead. The on/off ablation is still
    reported in every benchmark table, marked with a dagger where the tuned configuration
    already had feedback off.

13. **Retracted: "recurrence is what carries the result."** The sweep measures 3.46× on
    NARMA-5, 3.06× on Mackey-Glass (both at the saturation floor), 1.07× on NARMA-10 and
    1.00× on NARMA-20 — where the best gain is zero and the no-feedback ablation scores
    marginally better than the tuned model. What the sweep does establish is a failure
    boundary: past `g_fb ≈ 0.6` every task sits at NRMSE ≈ 1, the echo state property
    breaking.

14. **Hardware executed** (§ not in the pre-registration, which assumed simulation only).
    126 timesteps on `qpu:belenos`, 2 photons in 10 modes. Reported as feature-level
    agreement with simulation (correlation 0.805–0.844), **not** as an accuracy result: the
    stitched run has 66 training rows against 65 features.

15. **Three datasets report searches that predate this deviation.** `santa_fe`, `henon` and
    `parity_d3` are reported under their original 100-trial single-split searches. Their
    re-searches degraded test while improving validation, and the rolling-origin replacement
    was stopped before completing. Reporting a configuration selected by a procedure this
    document calls inadequate, or a half-finished one, would be worse than reporting the last
    complete and internally consistent search. `sp500_rv` is the exception — its
    rolling-origin re-tune completed for all five models and is reported.
