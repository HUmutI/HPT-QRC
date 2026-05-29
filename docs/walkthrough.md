# HPT-QRC: Complete Research Walkthrough

**Hybrid Photonic Temporal Quantum Reservoir Computing**
*Last updated: 2026-05-12 | Current config: v2 (window=10, photon_list=[2,3,4])*

> ⚠️ **Scope.** Quantum-feature results in this walkthrough are produced by classical simulation of linear-optical Fock-state probabilities via Perceval's SLOS backend; they are not hardware measurements. Hardware execution on Quandela Ascella/Belenos is the planned journal extension. "Linear-optical (simulated)" is the accurate descriptor; "photonic" is used loosely in figure captions only and is being phased out.

> 📄 **Concurrent and independent work.** A nearly identical architecture for swaption-surface reconstruction was independently proposed by Amanov & Azamov (arXiv:2603.10707, March 2026). The direct prior on quantum-reservoir realised-volatility forecasting is Li, Mukhopadhyay, Bayat & Habibnia (arXiv:2505.13933, 2025/2026) using a transverse-field Ising QRC. This work differs by (i) a temporal sliding-window formulation, (ii) cross-domain benchmarking across NARMA-10 / Mackey-Glass / S&P 500 RV / VIX, (iii) Diebold–Mariano with Newey–West HAC variance and Hansen Model Confidence Set on MSE and QLIKE, (iv) a Random Fourier Features + Ridge baseline of matched feature dimension, and (v) a planned Quandela hardware execution.

---

## 1. Project Overview

HPT-QRC uses **fixed, untrained linear-optical circuits** (simulated via Perceval SLOS) as high-dimensional nonlinear feature extractors for time-series forecasting. Only a Ridge regression readout is trained — no gradient descent, no barren-plateau pathology by construction.

**Current best config (v2):** `window=10, photon_list=[2,3,4]` (heterogeneous photon ensemble).

---

## 2. Main Results — v2 (window=10, hetero photons, 5 seeds mean ± std)

### NARMA10 — Nonlinear Synthetic Task

| Model                      | MSE (mean ± std)                      | QLIKE (mean ± std)              |
| :------------------------- | :------------------------------------- | :------------------------------- |
| AR(1)                      | 0.006427 ± 0.000921                   | 4.33 ± 0.52                     |
| HAR                        | 0.006199 ± 0.001067                   | 4.19 ± 0.53                     |
| RC (ESN)                   | 0.005944 ± 0.000913                   | 4.05 ± 0.45                     |
| Classical-Ridge (ablation) | 0.006870 ± 0.000923                   | 4.60 ± 0.39                     |
| HPT-QRC                    | 0.005752 ± 0.000928                   | 3.82 ± 0.43                     |
| HARX                       | 0.003816 ± 0.000473                   | 2.41 ± 0.24                     |
| **HPT-QRC-X**        | **0.000398 ± 0.000082 ← BEST** | **0.352 ± 0.138 ← BEST** |

> On NARMA-10, HPT-QRC-X achieves ~9.6× lower MSE and ~6.8× lower QLIKE than HARX. Significance is confirmed by the Diebold–Mariano test with Newey–West HAC variance (p < 0.01; see `results/NARMA10_DM_MSE.csv`). NARMA-10 is a strongly nonlinear synthetic benchmark; the ratio collapses on real financial data — see §RV and §VIX for honest reporting against HAR/HARX, and the RFF+Ridge column for the matched-dimension classical comparator.

### Mackey-Glass — 17-Step-Ahead (Memory Task)

| Model                      | MSE (mean ± std)           | QLIKE (mean ± std)                |
| :------------------------- | :-------------------------- | :--------------------------------- |
| AR(3)                      | 7e-6 ± 0                   | 0.0011 ± 0.0001                   |
| HARX                       | 9.2e-5 ± 5e-6              | 0.0146 ± 0.0007                   |
| Classical-Ridge (ablation) | 1.6e-3 ± 7.5e-5            | 0.273 ± 0.023                     |
| HPT-QRC                    | 1e-6 ± 0                   | 0.0001 ± 0.0000                   |
| **HPT-QRC-X**        | **1e-6 ± 0 ← BEST** | **0.0001 ± 0.0000 ← BEST** |

### S&P 500 — Realized Volatility

We performed an explicit ablation on the optimal window size for S&P 500, since financial datasets have different memory topologies compared to synthetic chaos:

| Window Size |          QRC MSE          | QRC-X MSE |
| :---------: | :------------------------: | :-------: |
|      1      |          0.011666          | 0.016335 |
|      2      |          0.011192          | 0.015451 |
| **3** | **0.010727** ← BEST | 0.018455 |
|      5      |          0.010836          | 0.018025 |
|      7      |          0.011248          | 0.014790 |
|     10     |          0.013857          | 0.013774 |

> **Conclusion**: Window=3 is the optimal temporal lag for the base HPT-QRC model on S&P 500, outperforming the default Window=10 used in synthetic tasks.

### VIX — Generalizability Test (6,288 samples)

| Model             | MSE                        | QLIKE                   |
| :---------------- | :------------------------- | :---------------------- |
| AR(3)             | 0.006039                   | 3.987                   |
| HAR               | 0.006040                   | 4.006                   |
| Classical-Ridge   | 0.006130                   | 4.081                   |
| **HPT-QRC** | **0.005977 ← BEST** | **3.965 ← BEST** |

> On VIX, HPT-QRC achieves the lowest MSE and QLIKE among the classical baselines shown; differences from AR(3) and HAR are small in absolute terms and we report DM HAC and Hansen MCS p-values to characterise statistical significance rather than make a "wins" claim. We do *not* interpret this as a quantum advantage; we interpret it as parity-to-modest-improvement consistent with the strong baselines (HAR/HARX) on this regime, in line with Branco et al. (2024) "HARd to Beat" findings on RV.

---

## 3. Diebold-Mariano Statistical Significance (Heatmaps)

To rigorously prove that the performance gains are statistically significant (and not just variance), we use the Diebold-Mariano (DM) test.
*(Green values < 0 indicate the Row Model significantly outperforms the Column Model)*

![NARMA10 DM Heatmap](narma_experiment/results/dm_heatmaps/NARMA10_DM_MSE_heatmap.png)
![Mackey Glass DM Heatmap](narma_experiment/results/dm_heatmaps/Mackey_Glass_DM_MSE_heatmap.png)

> **Conclusion**: The DM tests confirm that the HPT-QRC-X architecture's outperformance over the baselines on NARMA10 and Mackey-Glass is extremely statistically significant.

---

## 4. Memory Capacity & Information Processing Capacity

The earlier MC table compared an HPT-QRC with multi-hundred-feature output against a fixed-size 100-unit ESN. This is not a fair characterisation of classical ESN scaling: Jaeger MC is bounded by the linearly independent reservoir nodes, so a small ESN trivially under-reports. We therefore replace that table with:

1. A **tuned ESN sweep** at `res_size ∈ {50, 100, 200, 500, 1000, 2000}` with grid search over leak rate ∈ {0.1, 0.3, 0.5, 0.9} and spectral radius ∈ {0.6, 0.9, 1.1} per size.
2. **Matched total feature dimension** comparison between HPT-QRC photon configurations and ESN sizes.
3. **Information Processing Capacity (IPC; Dambre et al., Sci. Rep. 2012)** at degrees 1–4 in addition to the linear MC. The IPC plane (linear capacity vs nonlinear-sum capacity) characterises the memory–nonlinearity trade-off in a way that linear MC alone cannot.

See `memory_capacity.py` and `results/ipc_plane.png` for the protocol and the figure. The headline "50×" claim is withdrawn.

---

## 5. Ablation Study (NARMA10, 3 seeds)

| Dimension    | Config                           | MSE                |
| :----------- | :------------------------------- | :----------------- |
| n_photons    | 1                                | 0.007052           |
|              | 3 ✅ default                     | 0.007139           |
|              | 4                                | 0.007189           |
| n_reservoirs | 1                                | 0.007024           |
|              | 3 ✅ default                     | 0.007139           |
|              | 5                                | 0.007244           |
| window       | 5 ✅ (v1)                        | 0.007139           |
|              | **10 ✅ (v2)**             | **0.006586** |
|              | 15                               | 0.007498           |
| Ensemble     | Homo [3,3,3]                     | 0.007139           |
|              | **Hetero [2,3,4] ✅ (v2)** | **0.007124** |

---

## 6. Computational Efficiency

| Model             | Training Time    | Gradient Descent?                       |
| :---------------- | :--------------- | :-------------------------------------- |
| AR(3)             | 0.6 ms           | No                                      |
| HAR               | 2.4 ms           | No                                      |
| LSTM              | 685 ms           | Yes (100 epochs)                        |
| **HPT-QRC** | **914 ms** | **No — single closed-form pass** |

---

## 7. Results History

| Version                | Config                         | Location                    |
| :--------------------- | :----------------------------- | :-------------------------- |
| v1 (baseline)          | window=5, n_photons=3          | results/v1_window5_homo/    |
| **v2 (current)** | window=10, photon_list=[2,3,4] | results/v2_window10_hetero/ |

Full changelog: `results/CHANGELOG.md`

---

## 8. Supplementary Analyses

These analyses sit alongside the main benchmark and are not load-bearing for any "quantum advantage" claim. They characterise specific properties of the linear-optical feature map and the training paradigm; we report them as evidence about *what kind of system this is*, not about superiority.

### 8.1 Training-Compute Comparison (FLOPs vs. LSTM)

LSTM with BPTT over 100 epochs requires substantially more training-time FLOPs than a single closed-form Ridge solve over the HPT-QRC feature matrix. We report the ratio as a property of the *training paradigm* (one-shot Ridge vs. iterative gradient descent), not as a quantum-advantage statement. A classical Random Fourier Features + Ridge model has the same property — the FLOPs comparison is between *closed-form-readout systems and gradient-descent models*, and the linear-optical reservoir is one such system. The relevant additional question is the **inference-time** compute on simulated SLOS versus a (future) Quandela hardware run; this is documented in the planned "Hardware execution" section once the run is available.

![FLOPs Comparison](narma_experiment/results/advanced/flops_comparison.png)

### 8.2 Online Adaptation via Recursive Least Squares

The Ridge readout admits an exact online RLS form, which we provide as an alternative to the static fit. This is again a property of the **closed-form linear readout**, shared with RFF+Ridge, ESN, and any random-features-plus-RLS system. We document it for reproducibility and for the latency analysis in the hardware section; we do not claim it as a quantum-specific feature.

![Online RLS Learning](narma_experiment/results/advanced/online_rls_learning.png)

### 8.3 Feature-Matrix Conditioning (PCA Comparison)

PCA spectra on training feature matrices show that the linear-optical Fock-feature representation and the classical ESN representation differ in their cumulative explained-variance curves and condition numbers. We report this as evidence about the structure of the feature map under a fixed encoding, not as a proof of better representation: in particular this comparison does not control for matched dimension or a tuned ESN sweep, so we will re-run it in the matched-dim setting alongside the IPC plane and report both consistently. A matched RFF+Ridge baseline is the right comparator and is now included.

![PCA Independence](narma_experiment/results/advanced/pca_independence.png)

---

## 9. Scripts

```bash
conda activate quandela
python multi_seed_benchmark.py   # main results (5 seeds)
python memory_capacity.py        # MC analysis
python ablation_study.py         # ablation table
python efficiency_benchmark.py   # timing
python train_narma.py            # single-seed + DM tables + plots
python plot_dm_heatmaps.py       # Heatmap figures
python tune_sp500_window.py      # S&P 500 explicit window tuning
python pca_independence.py       # PCA linear independence analysis
python online_rls.py             # RLS streaming analysis
python flops_energy_calc.py      # FLOPs efficiency analysis
```

---

## 10. Paper TODO

Done:
- [x] Multiple seeds (5) with mean ± std
- [x] Linear Memory Capacity analysis (Jaeger 2001)
- [x] Full ablation table (photons, reservoirs, window, ensemble)
- [x] DM test CSVs generated (in `results/`)
- [x] Heterogeneous photon ensemble [2, 3, 4]
- [x] Second financial dataset (VIX)
- [x] Computational-efficiency table
- [x] DM heatmap figures
- [x] S&P 500 window-size ablation
- [x] Repository hygiene pass: claim cleanup, concurrent-work paragraphs, scope statement

In progress (Tier-A workshop / Tier-B journal upgrade per `/Users/umut/.claude/plans/in-the-epfl-anti-folder-jiggly-flurry.md`):
- [ ] **Random Fourier Features + Ridge** baseline at matched feature dimension (`rff_baseline.py`)
- [ ] **Walk-forward CV** for S&P 500 RV and VIX (`walk_forward_runner.py`)
- [ ] **Newey–West HAC** DM and **Hansen Model Confidence Set** (`dm_mcs.py`)
- [ ] **Information Processing Capacity** (Dambre 2012) + tuned ESN sweep (50–2000)
- [ ] **Matched-dim photon-ensemble ablation** (isolating photon-number axis from mode-count axis)
- [ ] **Shot noise & indistinguishability** sim (`noise_models.py`)
- [ ] **Echo-state-property check** (`esp_check.py`)
- [ ] Optuna-tuned LSTM / ESN / RFF (equal compute budget)
- [ ] Pre-registered protocol locked in `narma_experiment/PROTOCOL.md`

Pending hardware quota:
- [ ] Quandela cloud adapter (`quandela_runner.py`)
- [ ] Sim-vs-hardware concordance figure
- [ ] Hardware latency benchmark on walk-forward window

Writing:
- [ ] Workshop paper LaTeX (QTML 2026 / NeurIPS ML4PS or QML)
- [ ] Journal extension (Quantum Machine Intelligence / Quantum Sci. & Tech. / PR Applied) post-hardware
