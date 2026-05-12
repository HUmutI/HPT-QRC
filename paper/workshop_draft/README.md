# Workshop paper draft

Source for the workshop submission of the HPT-QRC paper.

## Targets

- **Primary:** QTML 2026 (Quantum Techniques in Machine Learning).
- **Secondary:** NeurIPS 2026 *Machine Learning and the Physical Sciences* workshop.
- **Backup:** NeurIPS 2026 / ICML 2026 QML workshop.

## Build

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

## Figure dependencies

The placeholder table in `main.tex` references CSVs produced by:

- `narma_experiment/multi_seed_benchmark.py` → `results/multi_seed_summary.csv` (mean ± std across seeds).
- `narma_experiment/walk_forward_runner.py --dataset sp500` → `results/wf_sp500_rv_summary.csv` (median + IQR across folds).
- `narma_experiment/walk_forward_runner.py --dataset vix` → `results/wf_vix_summary.csv`.
- `narma_experiment/memory_capacity.py` → `results/ipc_plane.png` (IPC plane figure).
- `narma_experiment/train_narma.py` → `results/<Dataset>_MCS_{MSE,QLIKE}.csv` (Hansen MCS survivor tables).
- `narma_experiment/esp_check.py` → `results/esp_decay.png` (fading-memory diagnostic).
- `narma_experiment/noise_models.py --sweep both --dataset narma` → `results/noise_*_sweep_narma.csv`.

## Editorial guardrails (PROTOCOL.md §9)

- No "quantum advantage" claims anywhere in the paper.
- No "outperforms" without specifying RFF+Ridge matched-dim comparison.
- No order-of-magnitude memory-capacity claims without matched-dim ESN sweep.
- The "Concurrent and independent work" paragraph for arXiv:2603.10707 must remain prominent in the related-work section.
- "Photonic" must always appear with a "(Perceval-simulated)" qualifier until the hardware-execution section lands in the journal version.
