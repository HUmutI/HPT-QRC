# Results report
Generated from the result CSVs by `scripts/render_report.py`. See `README.md` for
interpretation and caveats.
## Benchmarks

NRMSE, mean ± std over seeds. Lower is better.

| Model | santa_fe | channel_eq | parity_d3 | henon |
|---|---|---|---|---|
| Photonic (recurrent) | 0.0601 ± 0.0012 | 0.0849 ± 0.0041 | 5.91e-15 ± 5.39e-16 | 2.08e-05 ± 1.62e-05 |
| Photonic (no feedback) | 0.0601 ± 0.0012 | 0.0849 ± 0.0041 | 5.91e-15 ± 5.39e-16 | 2.08e-05 ± 1.62e-05 |
| Echo state network | 0.0618 ± 0.0052 | 0.1020 ± 0.0033 | 5.63e-04 ± 4.92e-05 | 0.0089 ± 0.0016 |
| Random Fourier features | 0.0644 ± 0.0038 | 0.1502 ± 0.0055 | 2.92e-14 ± 2.45e-15 | 8.80e-05 ± 2.34e-05 |
| Polynomial window | 0.0946 ± 0.00e+00 | 0.1535 ± 0.0044 | 4.70e-13 ± 1.86e-14 | 0.8428 ± 0.0225 |
| Linear window (control) | 0.4452 ± 0.00e+00 | 0.1729 ± 0.0050 | 1.0015 ± 0.0043 | 0.9838 ± 0.0128 |

- **santa_fe** — Diebold-Mariano vs photonic: Photonic (no feedback) p=1.0000, Echo state network p=0.9035, Random Fourier features p=0.1349, Polynomial window p=0.0103, Linear window (control) p=0.0000
- **channel_eq** — Diebold-Mariano vs photonic: Photonic (no feedback) p=1.0000, Echo state network p=0.0293, Random Fourier features p=0.0000, Polynomial window p=0.0000, Linear window (control) p=0.0000
- **parity_d3** — Diebold-Mariano vs photonic: Photonic (no feedback) p=1.0000, Echo state network p=0.0000, Random Fourier features p=0.0000, Polynomial window p=0.0000, Linear window (control) p=0.0000
- **henon** — Diebold-Mariano vs photonic: Photonic (no feedback) p=1.0000, Echo state network p=0.0005, Random Fourier features p=0.0161, Polynomial window p=0.0000, Linear window (control) p=0.0000

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
