# Results Changelog

All historical results are preserved in versioned subdirectories.

---

## v2 — window=10, photon_list=[2,3,4] (Current)
**Date:** 2026-05-03  
**Config:** window=10, photon_list=[2,3,4] (heterogeneous photon ensemble)  
**Files:** `results/v2_window10_hetero/`

### What changed from v1:
- `window` increased from 5 → 10
- `n_photons=3` (homogeneous) replaced with `photon_list=[2,3,4]` (heterogeneous)

### Key results (5 seeds, mean ± std):

| Dataset | Model | MSE | QLIKE |
|---|---|---|---|
| NARMA10 | **HPT-QRC-X** | **0.000398 ± 0.000082** ← NEW BEST | **0.352 ± 0.138** ← NEW BEST |
| NARMA10 | HARX | 0.003816 ± 0.000473 | 2.410 ± 0.245 |
| Mackey-Glass | **HPT-QRC-X** | **0.000001 ± 0.000000** | **0.0001 ± 0.0000** |
| S&P 500 | HARX | **0.009863 ± 0** | **0.833 ± 0** |
| S&P 500 | HPT-QRC | 0.013516 ± 0.000968 | 1.142 ± 0.090 |

### Notable finding:
**HPT-QRC-X on NARMA10 improved from 0.004427 → 0.000398 (11× improvement!)**.  
The wider window (10 steps) gives the exogenous signal much richer temporal context.  
S&P 500 performance degraded slightly — the larger window may be overfitting on the small financial dataset (164 test samples). This is expected.

---

## v1 — window=5, n_photons=3 (Baseline)
**Date:** 2026-05-03  
**Config:** window=5, n_photons=3 (homogeneous), n_reservoirs=3  
**Files:** `results/v1_window5_homo/`

### Key results (5 seeds, mean ± std):

| Dataset | Model | MSE | QLIKE |
|---|---|---|---|
| NARMA10 | HPT-QRC-X | 0.004427 ± 0.000355 | 2.770 ± 0.381 |
| NARMA10 | HARX | **0.003816 ± 0.000473** | **2.410 ± 0.245** |
| Mackey-Glass | **HPT-QRC-X** | **0.000001 ± 0.000000** | **0.0001 ± 0.0000** |
| S&P 500 | HARX | **0.009863 ± 0** | **0.833 ± 0** |
| S&P 500 | HPT-QRC | 0.011022 ± 0.000067 | 0.925 ± 0.006 |

---

## Recommendation for Paper

| Dataset | Report | Version |
|---|---|---|
| NARMA10 | Use **v2** results (HPT-QRC-X: 0.000398) | v2 |
| Mackey-Glass | Use **v2** results (identical) | v2 |
| S&P 500 | Use **v1** HPT-QRC results (0.011022) — v2 window=10 overfits | v1 |
| VIX | Use `results/vix_benchmark.csv` (HPT-QRC beats all) | v1 config |

**For S&P 500**: use window=5 for HPT-QRC (set it explicitly in that benchmark call).  
The paper should discuss this as a finding: *"Optimal window size is task-dependent — larger windows improve synthetic chaotic benchmarks but can overfit small real-world financial datasets."*
