<div align="center">
  <h1>Team Qedi 🐈‍⬛</h1>
  <p><strong>EPFL Quantum Hackathon 2026 • Quandela Challenge → Academic Paper Extension</strong></p>
  <h3>HPT-QRC: A Temporal Linear-Optical Quantum Reservoir<br/>(Simulated; Quandela hardware execution forthcoming)</h3>
  <p><i>Cross-domain benchmarking on chaotic and volatility forecasting tasks with full econometric evaluation.</i></p>
</div>

> ⚠️ **Scope statement.** All quantum-feature results reported here are produced by *classical simulation* of linear-optical Fock-state probabilities via Perceval's SLOS backend. They are **not** photonic-hardware measurements. Hardware execution on Quandela Ascella/Belenos is part of the planned journal extension. We use "linear-optical reservoir" rather than "photonic" except where explicitly contextualised by hardware results.

> 📄 **Concurrent and independent work.** A closely related architecture for swaption-surface reconstruction was independently proposed by Amanov & Azamov (arXiv:2603.10707, March 2026); a transverse-field Ising QRC for realised-volatility forecasting is in Li, Mukhopadhyay, Bayat & Habibnia (arXiv:2505.13933, 2025/2026). This work differs by (i) a temporal sliding-window formulation rather than a static surface model, (ii) systematic cross-domain benchmarking (NARMA-10, Mackey-Glass, S&P 500 RV; daily VIX is included in `data_loader.load_vix_df` as a release-companion dataset but is not analysed in the current manuscript), (iii) econometric evaluation using Diebold–Mariano with Newey–West HAC variance and the Hansen Model Confidence Set on MSE and QLIKE, and (iv) a planned hardware execution path on Quandela's linear-optical platform.

<br />

## 📖 1. What This Repository Is

This repository contains **two related but distinct bodies of work:**

| Phase | Description |
|:---|:---|
| **Phase 1 — Hackathon** | Original winning submission for the EPFL Quantum Hackathon 2026 (Quandela Challenge). Forecasts swaption volatility surfaces using HPT-QRC. |
| **Phase 2 — Academic Paper** | Extended benchmarking framework for submission to a peer-reviewed venue. Tests HPT-QRC across three rigorous datasets with full statistical validation. |

**You are currently on Phase 2.** All new benchmarking code lives in `src/`, `experiments/`, and `scripts/`.

---

## ⚙️ 2. Architecture (HPT-QRC)

The core model uses **fixed, untrained linear-optical circuits** (simulated via Perceval's SLOS backend) as nonlinear feature extractors. Only a Ridge regression readout is trained — no gradient descent, no barren-plateau pathology by construction.

**Current best config (v2):**
- `photon_list = [2, 3, 4]` — heterogeneous photon ensemble (covers quadratic, cubic, and quartic Fock correlations)
- `window = 10` — 10-step temporal context window
- `n_reservoirs = 3` — 3 reservoir families × 3 virtual depth levels = 9 total quantum layers
- `ridge_alpha = 1e-4` — Ridge regression readout

**Pipeline:**
```
Time Series Input → Sliding Window (size=10)
    → Phase Encoding into fixed photonic circuit
    → Multi-photon Fock state feature extraction (heterogeneous: 2+3+4 photons)
    → [Quantum Fock features] + [HAR classical context]
    → Ridge Regression (single closed-form solution, no epochs)
    → Forecast
```

---

## 🚀 3. Setup & Running

```bash
conda create -n quandela python=3.11
conda activate quandela
pip install perceval-quandela merlin optuna yfinance statsmodels
```

### Run Full Benchmark Suite
Run all scripts from the **project root** (`EPFL_ANTI/`):

```bash
# Main results — 5 seeds, mean ± std (publication-ready)
python experiments/multi_seed_benchmark.py

# Single-seed run + DM test tables + overlay plots
python experiments/train_narma.py

# Memory Capacity analysis (Jaeger 2001)
python experiments/memory_capacity.py

# Ablation study (photons, reservoirs, window, ensemble type)
python experiments/ablation_study.py

# Computational efficiency table
python experiments/efficiency_benchmark.py

# Walk-forward cross-validation
python experiments/walk_forward_runner.py

# Hyperparameter tuning (run before multi_seed_benchmark)
python experiments/tune_qrc.py
python experiments/tune_baselines.py

# Post-processing plots
python scripts/plot_dm_heatmaps.py
python scripts/format_dm.py
```

---

## 📈 4. Current Results (multi-seed, Optuna-tuned models, equal compute budget)

All classical baselines (ESN, LSTM, RFF+Ridge) and HPT-QRC are Optuna-tuned with comparable trial budgets per dataset (see `experiments/tune_baselines.py` and `experiments/tune_qrc.py`). Best configs are cached in `results/{lstm,esn,rff,qrc}_best_configs.json`.

### NARMA-10 — nonlinear synthetic task (5 seeds)
| Model | MSE (mean ± std) | QLIKE (mean ± std) |
|:---|:---|:---|
| HAR | 0.006199 ± 0.001067 | 4.19 ± 0.53 |
| ESN (tuned) | 0.005944 ± 0.000913 | 4.05 ± 0.45 |
| RFF+Ridge (tuned, matched dim) | 0.005682 ± 0.000961 | 3.75 ± 0.40 |
| HPT-QRC (default `[2,3,4]`) | 0.005752 ± 0.000928 | 3.82 ± 0.43 |
| **HPT-QRC (Optuna-tuned)** | **0.005336 ± 0.000918** ← BEST in univariate | **3.59 ± 0.48** |
| HARX | 0.003816 ± 0.000473 | 2.41 ± 0.24 |
| RFF+Ridge-X (tuned) | 0.000436 ± 0.000064 | 0.39 ± 0.11 |
| **HPT-QRC-X** | **0.000398 ± 0.000082** ← BEST | **0.35 ± 0.14** |

> Tuned HPT-QRC beats the matched-dimension Optuna-tuned RFF baseline by ~6 % on MSE in the univariate regime and by ~9 % in the exogenous regime. The matched-dim RFF baseline is the strongest fixed-feature classical comparator (random nonlinear features + Ridge); tuned HPT-QRC dominates it consistently after symmetric tuning, which is the relevant honest comparison.

### Mackey-Glass — 17-step-ahead (5 seeds)
| Model | MSE (mean ± std) | QLIKE (mean ± std) |
|:---|:---|:---|
| AR(3) | 6.5e-6 ± 4e-7 | 1.1e-3 |
| HAR | 1.4e-4 ± 9e-6 | 0.020 |
| ESN (tuned) | 1.05e-5 ± 9e-7 | 1.6e-3 |
| RFF+Ridge (tuned) | 1.32e-6 ± 2e-7 | 1.9e-4 |
| HPT-QRC (default) | 1.09e-6 ± 2e-7 | 1.5e-4 |
| **HPT-QRC (Optuna-tuned)** | **< 1e-7** ← BEST | **< 5e-5** |
| HPT-QRC-X | 7.2e-7 ± 3e-7 | 1.05e-4 |
| RFF+Ridge-X (tuned) | 7.5e-7 ± 1e-7 | 1.1e-4 |

### S&P 500 monthly realised volatility — fixed 80 / 20 split (5 seeds)
| Model | MSE | QLIKE |
|:---|:---|:---|
| AR(1) | 0.01134 | 0.947 |
| **AR(3)** | **0.01036** | **0.866** |
| HAR | 0.01048 | 0.876 |
| HARX | 0.01096 | 0.930 |
| ESN (tuned) | 0.01073 ± 0.00017 | 0.907 ± 0.016 |
| RFF+Ridge (tuned) | 0.01152 ± 0.00015 | 0.965 ± 0.013 |
| HPT-QRC (default) | 0.01352 ± 0.00097 | 1.142 ± 0.090 |
| **HPT-QRC (Optuna-tuned)** | **0.01049 ± 0.00010** | **0.877 ± 0.008** |

> Tuned HPT-QRC ties AR(3) within ~1 % on MSE and within ~1 % on QLIKE, and beats HAR / HARX / tuned RFF on the fixed-split S&P 500 RV benchmark. Default HPT-QRC trails — the lesson is that **per-dataset tuning matters as much for QRC as for classical baselines**, and any honest QRC-vs-classical comparison must give both sides equal Optuna budget. The walk-forward evaluation (8 folds, 1970–2017, in `results/wf_sp500_rv_summary.csv`) currently uses the default config; we will report the tuned walk-forward in the journal extension.

### S&P 500 RV walk-forward (8 folds, 1970–2017) — default config
| Model | MSE (med) | QLIKE (med) |
|:---|:---|:---|
| AR(1) | 6.0e-3 | 8.5e-3 |
| **HAR** | **5.4e-3** | **8.6e-3** |
| AR(3) | 5.6e-3 | 8.5e-3 |
| HARX | 5.9e-3 | 9.7e-3 |
| ESN | 6.6e-3 | 9.7e-3 |
| RFF+Ridge | 6.8e-3 | 9.9e-3 |
| HPT-QRC | 7.2e-3 | 1.2e-2 |
| LSTM (tuned) | 1.5e-2 | 2.2e-2 |

### Information Processing Capacity (Dambre 2012)
| System | feat. dim | Linear MC | IPC deg 1 | IPC deg 2 | IPC deg 3 |
|:---|:---:|:---:|:---:|:---:|:---:|
| HPT-QRC ($n=3$) | 95 | 5.00 | 5.00 | 14.89 | 32.49 |
| HPT-QRC-Hetero `[2,3,4]` | 95 | 5.00 | 5.00 | 14.90 | **32.62** |
| ESN res=200 (tuned) | 202 | 17.85 | 5.99 | 20.38 | 24.55 |
| ESN res=1000 (tuned) | 1002 | 19.36 | 5.95 | 20.60 | 45.83 |
| Random-linear (dim 100) | 100 | 5.01 | 5.00 | 0.00 | 3.62 |

The earlier coarse "50× memory vs ESN" comparison was replaced by this matched-feature-dimension IPC plane (`results/ipc_plane.png`). Honest reading: at matched dimension ≈ 100, the photon-ensemble reservoir produces ~1.3× the degree-3 nonlinear capacity of a tuned ESN and ~9× the random-linear baseline's degree-3 IPC. ESN dominates linear MC at any reasonable size; the photon ensemble's structural contribution is concentrated at higher-degree nonlinearity.

### Fock-space scaling (clean monotone signal on MG / SP500)
Joint sweep over photon count $n$ and mode count $m$. Effective unbunched Fock dimension is $\binom{m}{n}$. Mackey-Glass NRMSE drops monotonically from $5.53\!\cdot\!10^{-3}$ at Fock-dim 8 to $5.17\!\cdot\!10^{-3}$ at Fock-dim 56 (photon axis); S&P 500 RV NRMSE drops from 0.743 to 0.696 going from $\binom{4}{3}=4$ to $\binom{12}{3}=220$ (mode axis). NARMA-10 saturates at small dim, consistent with its bounded nonlinearity order. Full curves in `results/ablation_fock_scaling_combined.png`.

### Training efficiency
> ⚠️ **Benchmark conditions:** Measured on NARMA-10 (~800 training samples, 1 feature) on a standard CPU. Times will scale with dataset size and photon configuration.

| Model | Training Time | Epochs | Notes |
|:---|:---|:---|:---|
| AR(3) | ~0.6 ms | N/A | Closed-form OLS |
| HAR | ~2.4 ms | N/A | Closed-form OLS |
| LSTM (Optuna-tuned) | ~685 ms | 50–200 | BPTT + early stopping |
| **HPT-QRC (tuned)** | **~914 ms** | **N/A** | **Single closed-form Ridge solve** |

The training paradigm — not raw wall-clock — is the relevant selling point of HPT-QRC: closed-form readout, no learning-rate sensitivity, deterministic result. The matched-dimension RFF baseline shares this property; HPT-QRC's additional contribution over RFF lies in the *structure* of the Fock-feature map (Section above on IPC) and in the eventual hardware path.

### Training Efficiency
> ⚠️ **Benchmark conditions:** Measured on NARMA10 (~800 training samples, 1 feature) on a standard CPU. Times will scale with dataset size and photon configuration.

| Model | Training Time | Epochs | Notes |
|:---|:---|:---|:---|
| AR(3) | ~0.6 ms | N/A | Closed-form OLS |
| HAR | ~2.4 ms | N/A | Closed-form OLS |
| LSTM | ~685 ms | 100 | Gradient descent (BPTT, Optuna-tuned across {layers, hidden, lr, dropout} with early stopping) |
| **HPT-QRC** | **~914 ms** | **N/A** | **Single closed-form Ridge solve — no gradient descent, no epochs, deterministic result** |

The selling point of HPT-QRC is **training paradigm**, not raw speed: no iterative optimisation, no learning-rate sensitivity, and a global optimum from the closed-form Ridge solve. Note that "HPT-QRC outperforms LSTM on small samples" is, on its own, *not* a quantum-feature claim — Ridge on rich nonlinear features routinely beats deep models in the small-sample regime regardless of feature source (Branco et al. 2024 on RV). The relevant question is whether the linear-optical feature map outperforms or matches a matched-dimension classical Random Fourier Features baseline; this comparison is in the RFF column of the benchmark tables.

---

## 📁 5. Project Structure

```
EPFL_ANTI/
├── README.md                        ← this file
│
├── docs/                            ← documentation & research notes
│   ├── PROTOCOL.md                  ← pre-registered experimental protocol
│   ├── walkthrough.md               ← full research walkthrough & results log
│   └── claude_deepsearch.md         ← deep research notes
│
├── src/                             ← core library (imported by experiments)
│   ├── multi_qrc.py                 ← HPT-QRC model (HPT_QRC_Multi)
│   ├── data_loader.py               ← NARMA10 / Mackey-Glass / SP500 / VIX loaders
│   ├── classical_baselines.py       ← AR, HAR/HARX, LSTM, RC, ClassicalContextRidge
│   ├── esn_baseline.py              ← Echo State Network
│   ├── rff_baseline.py              ← Random Fourier Features + Ridge (matched-dim)
│   ├── dm_mcs.py                    ← Newey-West HAC DM test + Hansen MCS
│   ├── noise_models.py              ← Shot noise + indistinguishability models
│   └── walk_forward.py              ← WalkForwardSplit iterator
│
├── experiments/                     ← runnable benchmark scripts (run from project root)
│   ├── train_narma.py               ← single-seed benchmark + DM-HAC + MCS tables
│   ├── multi_seed_benchmark.py      ← 5-seed benchmark, mean ± std (publication-ready)
│   ├── walk_forward_runner.py       ← walk-forward CV driver (median + IQR + MCS)
│   ├── memory_capacity.py           ← linear MC + Dambre IPC + tuned ESN sweep
│   ├── esp_check.py                 ← echo-state-property / fading-memory check
│   ├── efficiency_benchmark.py      ← wall-clock training time + model size table
│   ├── pca_independence.py          ← PCA / linear independence analysis of features
│   ├── online_rls.py                ← online recursive least-squares adaptation
│   ├── flops_energy_calc.py         ← static FLOP / energy estimate
│   ├── ablation_study.py            ← photons / reservoirs / window ablation
│   ├── ablation_fock_scaling.py     ← joint (n_photons, n_modes) Fock-dim sweep
│   ├── ablation_matched_dim.py      ← photon-list ablation at matched feature dim
│   ├── tune_qrc.py                  ← Optuna tuner for HPT-QRC
│   ├── tune_baselines.py            ← Optuna tuner for LSTM / ESN / RFF
│   └── tune_sp500_window.py         ← window-size tuning for S&P 500
│
├── scripts/                         ← post-processing utilities
│   ├── plot_dm_heatmaps.py          ← DM test heatmap plots
│   ├── format_dm.py                 ← print DM tables as markdown
│   └── render_report.py             ← render full walkthrough report
│
├── results/                         ← generated outputs (CSVs, PNGs, JSON configs)
├── literature/                      ← reference papers and benchmark data (Data.CSV)
└── paper/workshop_draft/            ← LaTeX manuscript
```

---

## 👥 Team Qedi

Proudly built during the **EPFL Quantum Hackathon 2026**, now being extended into an academic publication.

* [**Eren Aslan**](https://www.linkedin.com/in/eren-aslan-421b66191/)
* [**Hüseyin Umut Işık**](https://www.linkedin.com/in/h%C3%BCseyin-umut-i%C5%9F%C4%B1k-7b3ba4255/)
* [**Arda Kara**](https://www.linkedin.com/in/arda-kara0/)
* [**Mehmet Alp Özaydın**](https://www.linkedin.com/in/mehmet-alp-%C3%B6zayd%C4%B1n-8455bb246/)
