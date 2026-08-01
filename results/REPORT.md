# Results report
Generated from the result CSVs by `scripts/render_report.py`. See `README.md` for
interpretation and caveats.
## Benchmarks

NRMSE, mean ± std over seeds. Lower is better.

| Model | sp500_rv |
|---|---|
| Photonic (recurrent) | 0.7038 ± 0.0157 |
| Photonic (no feedback) | 0.6984 ± 0.0071 |
| Echo state network | 0.7388 ± 0.0276 |
| Random Fourier features | 0.7055 ± 0.0029 |
| Polynomial window | 0.6974 ± 0.00e+00 |
| Linear window (control) | 0.6957 ± 0.00e+00 |

- **sp500_rv** — Diebold-Mariano vs photonic: Photonic (no feedback) p=0.4097, Echo state network p=0.0213, Random Fourier features p=0.0391, Polynomial window p=0.7685, Linear window (control) p=0.7675

## Matched capacity (NARMA-10)

The tuned photonic configuration uses more features than the baselines, so this
compares every family in a common dimension range.

| Model | dim | NRMSE |
|---|---|---|
| Photonic (recurrent) | 1381 | 0.1092 |
| Photonic (recurrent) | 701 | 0.1643 |
| Random Fourier features | 1200 | 0.1876 |
| Random Fourier features | 600 | 0.2373 |
| Echo state network | 501 | 0.2660 |
| Echo state network | 1001 | 0.2676 |
| Polynomial window | 679 | 0.4566 |

## Memory and information processing capacity

| System | dim | Linear MC | IPC total | IPC per feature |
|---|---|---|---|---|
| photonic_no_feedback | 132 | 10.5 | 111.9 | 0.85 |
| linear_window_20 | 20 | 19.1 | 13.0 | 0.65 |
| esn_200 | 201 | 27.0 | 123.9 | 0.62 |
| photonic | 132 | 11.1 | 69.3 | 0.52 |
| esn_500 | 501 | 29.6 | 213.9 | 0.43 |

## Noise robustness

- noiseless: NRMSE 0.2479
- classical_control: NRMSE 0.3481

### Coincidences per timestep

| Coincidences per timestep | NRMSE |
|---|---|
| ∞ | 0.2748 ± 0.0279 |
| 30 | 0.4901 ± 0.0354 |
| 100 | 0.5340 ± 0.0520 |
| 300 | 0.4626 ± 0.0192 |
| 1000 | 0.4113 ± 0.0439 |
| 3000 | 0.4038 ± 0.0480 |
| 10000 | 0.3798 ± 0.0153 |
| 30000 | 0.3433 ± 0.0303 |

### Indistinguishability

| Indistinguishability | NRMSE |
|---|---|
| 0.5 | 0.2484 |
| 0.6 | 0.2482 |
| 0.7 | 0.2483 |
| 0.8 | 0.2468 |
| 0.85 | 0.2478 |
| 0.9 | 0.2486 |
| 0.95 | 0.2483 |
| 1 | 0.2479 |

### g2(0)

| g2(0) | NRMSE |
|---|---|
| 0 | 0.2479 |
| 0.02 | 0.2480 |
| 0.05 | 0.2482 |
| 0.1 | 0.2485 |
| 0.2 | 0.2492 |
| 0.3 | 0.2498 |
