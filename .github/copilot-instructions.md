# HPT-QRC Codebase — AI Agent Instructions

## Project Identity
**Hybrid Photonic Temporal Quantum Reservoir Computing (HPT-QRC)** — EPFL Quantum Hackathon 2026 winning submission, now extended into an academic paper benchmarking framework. The repo has two distinct phases:
- **Phase 1 (Hackathon):** `train_final.py`, `final_model.py`, `final_model_3d_vis.py` — swaption surface forecasting with PCA + Perceval QRC + Ridge or PennyLane Hybrid QNN.
- **Phase 2 (Academic):** `narma_experiment/` — multi-dataset benchmarking (NARMA10, Mackey-Glass, S&P 500 RV, VIX) with full statistical validation.

**Active work is in `narma_experiment/`.** Phase 1 files are reference/archive only.

---

## Architecture (HPT-QRC)

```
Time Series → Sliding Window (size=10)
    → Phase Encoding into fixed Perceval photonic circuit (Mach-Zehnder + PS)
    → Multi-photon Fock state feature extraction via MerLin QuantumLayer + LexGrouping
    → Heterogeneous ensemble: photon_list=[2,3,4] (3 reservoir families × 3 virtual depths = 9 layers)
    → [Quantum Fock features] ++ [HAR classical context (optional)]
    → Ridge regression readout (closed-form, no gradient descent)
    → Forecast
```

The **reservoirs are fixed** (no gradient flows through them). Only the Ridge readout is fitted. This is the defining property — no barren plateaus, sub-millisecond inference.

**Core class:** `narma_experiment/multi_qrc.py` → `HPT_QRC_Multi`
- `photon_list=[2,3,4]` enables the heterogeneous ensemble (each photon count = different Fock-space dimensionality and multi-photon interference statistics)
- `use_har_context=True` appends HAR(1,5,22) classical features → "HPT-QRC-X" variant (best performing)
- `n_virtual_nodes` controls circuit depth layers; combined with `photon_list` determines total feature dimension

**Key hyperparameters (current best v2):**
```python
HPT_QRC_Multi(window=10, photon_list=[2, 3, 4], n_virtual_nodes=3,
              lex_out=10, ridge_alpha=1e-4, use_har_context=True)
```
For S&P 500 RV specifically: `window=3` is optimal (financial memory topology differs from synthetic chaos).

---

## Critical Dependencies
- **`perceval-quandela`** — photonic circuit simulation (SLOS backend, permanent evaluation)
- **`merlinquantum`** — `QuantumLayer`, `ComputationSpace.UNBUNCHED`, `LexGrouping`; wraps Perceval for PyTorch integration
- **`pennylane`** — used only in Phase 1 (`train_final.py`) for variational Hybrid QNN
- Python 3.11 + conda env `quandela`

```bash
conda create -n quandela python=3.11
conda activate quandela
pip install -r requirements.txt
pip install yfinance   # for VIX dataset
```

---

## Developer Workflows

### Run Full Benchmark Suite (Phase 2)
```bash
cd narma_experiment/

python multi_seed_benchmark.py   # 5 seeds, mean±std — publication-ready
python memory_capacity.py        # Jaeger MC: HPT-QRC=4.0 vs ESN=0.08
python ablation_study.py         # photons / reservoirs / window / ensemble type
python efficiency_benchmark.py   # compute cost table
python train_narma.py            # single-seed + DM test tables + overlay plots
python tune_sp500_window.py      # window sweep for financial data
```

Results land in `narma_experiment/results/`. Heatmaps in `results/dm_heatmaps/`.

### Phase 1 (Hackathon) — Swaption Surface
```bash
python train_final.py            # trains ClassicalLSTM + HybridQNN, saves to logs/
python final_model.py            # Perceval QRC pipeline, predicts swaption surface
```
Expects data under `CHALLENGE RESOURCES/DATASETS/` (not in repo).

---

## Project-Specific Conventions

- **"HPT-QRC" vs "HPT-QRC-X":** base model has no classical context; `-X` appends HAR features (`use_har_context=True`). Always distinguish in result tables.
- **Metrics:** always report both **MSE** and **QLIKE** (volatility-appropriate loss). DM test p-values are required for paper claims.
- **"Quantum advantage" is forbidden language.** Use "photonic feature extractor" or "expressivity via Fock-space dimensionality." See `claude_deepsearch.md §5`.
- **Reproducibility:** all reservoir builds use `torch.manual_seed(42 + r_idx * 1000)` for determinism across seeds.
- **Preprocessing in `HPT_QRC_Multi`:** uses Winsorize → IQR scale → Min-max [0,1] (fitted on training data only). Never apply raw normalization to Perceval inputs.
- **Concurrent paper:** arXiv:2603.10707 (Amanov & Azamov) shares the same core architecture. Any paper contributions must be positioned relative to it — see `claude_deepsearch.md §2`.

---

## Key Files
| File | Purpose |
|---|---|
| `narma_experiment/multi_qrc.py` | Core `HPT_QRC_Multi` class — edit here for architecture changes |
| `narma_experiment/multi_seed_benchmark.py` | Main benchmark runner (5 seeds) |
| `narma_experiment/train_narma.py` | Single-seed + DM significance tests |
| `narma_experiment/ablation_study.py` | Ablation across photons/window/reservoirs |
| `narma_experiment/esn_baseline.py` | Classical ESN comparator |
| `walkthrough.md` | Up-to-date results summary and paper TODO |
| `claude_deepsearch.md` | Academic positioning, related work, claims to avoid |
| `narma_experiment/results/` | All CSVs and figures — do not delete |
