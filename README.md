<div align="center">
  <h1>Team Qedi 🐈‍⬛</h1>
  <p><strong>EPFL Quantum Hackathon 2026 • Quandela Challenge → Academic Paper Extension</strong></p>
  <h3>Hybrid Photonic Temporal QRC (HPT-QRC)<br/>Time-Series Forecasting Benchmark Suite</h3>
  <p><i>Teaching photons to predict the market so we can finally sleep.</i></p>
</div>

<br />

## 📖 1. What This Repository Is

This repository contains **two related but distinct bodies of work:**

| Phase | Description |
|:---|:---|
| **Phase 1 — Hackathon** | Original winning submission for the EPFL Quantum Hackathon 2026 (Quandela Challenge). Forecasts swaption volatility surfaces using HPT-QRC. |
| **Phase 2 — Academic Paper** | Extended benchmarking framework for submission to a peer-reviewed venue. Tests HPT-QRC across three rigorous datasets with full statistical validation. |

**You are currently on Phase 2.** All new benchmarking code lives in `narma_experiment/`.

---

## ⚙️ 2. Architecture (HPT-QRC)

The core model uses **fixed, untrained photonic quantum circuits** as nonlinear feature extractors. Only a Ridge regression readout is trained — no gradient descent, no barren plateaus.

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
pip install -r requirements.txt
pip install yfinance   # for VIX dataset
```

### Run Full Benchmark Suite
```bash
cd narma_experiment/

# Main results — 5 seeds, mean ± std (publication-ready)
python multi_seed_benchmark.py

# Memory Capacity analysis (Jaeger 2001)
python memory_capacity.py

# Ablation study (photons, reservoirs, window, ensemble type)
python ablation_study.py

# Computational efficiency table
python efficiency_benchmark.py

# Single-seed run + DM test tables + overlay plots
python train_narma.py
```

---

## 📈 4. Current Results (v2: window=10, photon_list=[2,3,4], 5 seeds)

### NARMA10 — Nonlinear Synthetic Task
| Model | MSE (mean ± std) | QLIKE (mean ± std) |
|:---|:---|:---|
| HAR | 0.006199 ± 0.001067 | 4.19 ± 0.53 |
| HARX | 0.003816 ± 0.000473 | 2.41 ± 0.24 |
| Classical-Ridge (ablation) | 0.006870 ± 0.000923 | 4.60 ± 0.39 |
| HPT-QRC | 0.005752 ± 0.000928 | 3.82 ± 0.43 |
| **HPT-QRC-X** | **0.000398 ± 0.000082** ← **BEST** | **0.352 ± 0.138** ← **BEST** |

> HPT-QRC-X beats HARX by **9.6× on MSE** and **6.8× on QLIKE**.

### Mackey-Glass — 17-Step-Ahead (Memory Task)
| Model | MSE (mean ± std) | QLIKE (mean ± std) |
|:---|:---|:---|
| AR(3) | 7e-6 ± 0 | 0.0011 |
| HARX | 9.2e-5 ± 5e-6 | 0.0146 |
| Classical-Ridge (ablation) | 1.6e-3 ± 7.5e-5 | 0.273 |
| **HPT-QRC-X** | **1e-6 ± 0** ← **BEST** | **0.0001 ± 0** ← **BEST** |

### S&P 500 Realized Volatility (Window Tuning)
Financial datasets possess different memory topologies compared to synthetic chaos. Our explicit ablation shows that **Window=3** is the optimal temporal lag for the base HPT-QRC model on S&P 500:

| Window Size | QRC MSE | QRC-X MSE |
|:---:|:---:|:---:|
| 1 | 0.011666 | 0.016335 |
| **3** | **0.010727** ← **BEST** | 0.018455 |
| 5 | 0.010836 | 0.018025 |
| 10 | 0.013857 | 0.013774 |


### VIX — Generalizability Test (6,288 samples)
| Model | MSE | QLIKE |
|:---|:---|:---|
| AR(3) | 0.006039 | 3.987 |
| HAR | 0.006040 | 4.006 |
| **HPT-QRC** | **0.005977** ← **BEST** | **3.965** ← **BEST** |

### Memory Capacity (Jaeger 2001)
| Model | MC Score |
|:---|:---|
| **HPT-QRC** | **4.00** |
| Classical ESN (100 units) | 0.08 |

> Photonic reservoir provides **50× better temporal memory** than classical ESN.

### Training Efficiency
> ⚠️ **Benchmark conditions:** Measured on NARMA10 (~800 training samples, 1 feature) on a standard CPU. Times will scale with dataset size and photon configuration.

| Model | Training Time | Epochs | Notes |
|:---|:---|:---|:---|
| AR(3) | ~0.6 ms | N/A | Closed-form OLS |
| HAR | ~2.4 ms | N/A | Closed-form OLS |
| LSTM | ~685 ms | 100 | Gradient descent (BPTT) |
| **HPT-QRC** | **~914 ms** | **N/A** | **Single closed-form Ridge solve — no gradient descent, no epochs, deterministic result** |

The key advantage of HPT-QRC is not raw speed but **training paradigm**: no iterative optimisation, no hyperparameter sensitivity from learning rates or epochs, and a guaranteed global optimum from the closed-form Ridge solution. The full 5-seed × 3-dataset benchmark completes in under 1 minute on a standard CPU.

---

## 📁 5. Project Structure

```
EPFL_ANTI/
├── README.md                        ← this file
├── walkthrough.md                   ← full research walkthrough & results log
├── requirements.txt                 ← dependencies
│
├── narma_experiment/                ← Academic benchmark suite (Phase 2)
│   ├── multi_qrc.py                 ← HPT-QRC model (photon_list, get_features, etc.)
│   ├── classical_baselines.py       ← AR, HAR, HARX, LSTM, RC, ClassicalContextRidge
│   ├── data_loader.py               ← NARMA10, Mackey-Glass, S&P 500, VIX loaders
│   ├── esn_baseline.py              ← Echo State Network
│   ├── train_narma.py               ← Single-seed benchmark + DM tests + plots
│   ├── multi_seed_benchmark.py      ← 5-seed benchmark → mean ± std results
│   ├── memory_capacity.py           ← Jaeger (2001) MC analysis
│   ├── ablation_study.py            ← Architecture ablation sweeps
│   ├── efficiency_benchmark.py      ← Training time measurement
│   └── results/
│       ├── CHANGELOG.md             ← Version history of all experiments
│       ├── v1_window5_homo/         ← Baseline results (window=5, n_photons=3)
│       ├── v2_window10_hetero/      ← Current best (window=10, photon_list=[2,3,4])
│       ├── multi_seed_summary.csv   ← Main publication table
│       ├── mc_curve.png             ← Memory Capacity plot
│       ├── ablation_combined.png    ← Ablation study plot
│       └── efficiency_plot.png      ← Speed vs accuracy plot
│
└── [original hackathon files]       ← Phase 1 (swaption surface)
```

---

## 👥 Team Qedi

Proudly built during the **EPFL Quantum Hackathon 2026**, now being extended into an academic publication.

* [**Eren Aslan**](https://www.linkedin.com/in/eren-aslan-421b66191/)
* [**Hüseyin Umut Işık**](https://www.linkedin.com/in/h%C3%BCseyin-umut-i%C5%9F%C4%B1k-7b3ba4255/)
* [**Arda Kara**](https://www.linkedin.com/in/arda-kara0/)
* [**Mehmet Alp Özaydın**](https://www.linkedin.com/in/mehmet-alp-%C3%B6zayd%C4%B1n-8455bb246/)
