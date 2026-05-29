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

(Add `DEVIATION YYYY-MM-DD:` blocks here when the protocol changes.)
