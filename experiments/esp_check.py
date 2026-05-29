"""
esp_check.py
============
Empirical Echo-State-Property (ESP) / fading-memory check for HPT-QRC.

The ESP says that the reservoir state is, asymptotically, a function only
of the input history (not of the initial state). A common empirical test:
inject two trajectories that share the same input from time t0 onward but
have *different* states/inputs before t0, and verify that the reservoir
features converge.

Here we go further and follow the "perturbation decay" diagnostic used in
classical RC papers:

  1. Take a length-T random input u of fixed distribution.
  2. Build u' identical to u except for a small perturbation in u[0..k].
  3. Run HPT_QRC_Multi.get_features on both to get F, F'.
  4. Plot d(t) = || F[t] - F'[t] || vs t.
  5. ESP holds if d(t) decays toward 0; fails if d(t) grows or stays flat.

The sliding-window encoding in HPT_QRC_Multi places past inputs in the
*input* slot, so for windows of size W the perturbation should be visible
for exactly W steps and decay to 0 thereafter. If decay is monotonic with
no further long-range residual, the model behaves as a *quantum extreme
learning machine* (QELM) over W-windows, not a true RC. This file makes
that determination empirically and prints the label.

Outputs:
  results/esp_decay.csv     (per-t distance values for several seeds)
  results/esp_decay.png     (curve)
  results/esp_label.txt     (RC / QELM verdict)
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pathlib as _pathlib, sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent / 'src'))
del _pathlib, _sys
from multi_qrc import HPT_QRC_Multi


def perturbation_decay(qrc: HPT_QRC_Multi, T: int = 500,
                        k: int = 5, eps: float = 0.1,
                        seed: int = 0) -> np.ndarray:
    """Return d(t) = || F[t] - F'[t] || for t = 1..T-1."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 1.0, T).reshape(-1, 1)
    u_perturbed = u.copy()
    u_perturbed[:k] = np.clip(u_perturbed[:k] + eps, 0.0, 1.0)

    F = qrc.get_features(u)
    Fp = qrc.get_features(u_perturbed)

    n = min(len(F), len(Fp))
    diff = np.linalg.norm(F[:n] - Fp[:n], axis=1)
    return diff


def classify_label(curve: np.ndarray, window: int) -> str:
    """Return 'RC' if decay extends well beyond `window`, else 'QELM-windowed'."""
    if len(curve) < window + 5:
        return "INCONCLUSIVE"
    peak = curve.max()
    if peak < 1e-10:
        return "INCONCLUSIVE (no observable perturbation)"
    after_window_max = curve[window + 1:].max() / (peak + 1e-30)
    if after_window_max < 1e-2:
        return f"QELM-windowed (perturbation decays within window={window})"
    if curve[-len(curve) // 4:].mean() > 0.1 * peak:
        return "ESP-VIOLATING (long-range residual)"
    return "RC (decay extends beyond window)"


def main(out_dir: str = "results", n_seeds: int = 5, T: int = 500,
         k: int = 5, eps: float = 0.1):
    os.makedirs(out_dir, exist_ok=True)
    qrc = HPT_QRC_Multi(in_size=1, window=10, seed=42,
                         photon_list=[2, 3, 4], use_har_context=False)

    curves = []
    for s in range(n_seeds):
        curves.append(perturbation_decay(qrc, T=T, k=k, eps=eps, seed=s))
    L = min(len(c) for c in curves)
    arr = np.stack([c[:L] for c in curves], axis=0)
    median = np.median(arr, axis=0)
    q1 = np.quantile(arr, 0.25, axis=0)
    q3 = np.quantile(arr, 0.75, axis=0)

    df = pd.DataFrame({
        "t": np.arange(L),
        "median_dist": median,
        "q1_dist": q1,
        "q3_dist": q3,
    })
    df.to_csv(f"{out_dir}/esp_decay.csv", index=False)

    label = classify_label(median, window=qrc.window)
    with open(f"{out_dir}/esp_label.txt", "w") as fh:
        fh.write(label + "\n")
    print(f"[ESP] window={qrc.window}, peak={median.max():.3e}")
    print(f"[ESP] verdict: {label}")

    fig, ax = plt.subplots(figsize=(8, 5))
    t = np.arange(L)
    ax.plot(t, median, color="#3b82f6", linewidth=2, label="median over seeds")
    ax.fill_between(t, q1, q3, alpha=0.25, color="#3b82f6", label="IQR")
    ax.axvline(qrc.window, color="#ef4444", linestyle="--",
               label=f"window = {qrc.window}")
    ax.set_xlabel("t (timestep)")
    ax.set_ylabel(r"$\|F(t) - F'(t)\|$")
    ax.set_title("Perturbation decay (ESP check)", fontweight="bold")
    ax.set_yscale("log")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/esp_decay.png", dpi=150)
    plt.close()
    print(f"[ESP] saved {out_dir}/esp_decay.png")


if __name__ == "__main__":
    main()
