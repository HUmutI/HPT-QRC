"""Sim-vs-hardware comparison figure (the paper figure, PLAN.md section 6).

Reads a subset result JSON produced by run_hw_subset.py and renders:
  (a) forecast traces: y_true vs sim-feature prediction vs HW-feature prediction
  (b) per-step feature correlation (sim vs HW LexGrouping features)
  (c) pooled scatter of sim vs HW feature values
  (d) MSE bars

Usage:
    python hardware/compare_sim_hw.py hardware/results/subset_local_slos_narma10.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(path):
    d = json.loads(Path(path).read_text())
    y = np.array(d["y_test"]).ravel()
    sim_pred = np.array(d["sim_pred"]).ravel()
    hw_pred = np.array(d["hw_pred"]).ravel()
    Q_sim = np.array(d["Q_sim"])
    Q_hw = np.array(d["Q_hw"])
    corr = np.array([np.corrcoef(Q_hw[i], Q_sim[i])[0, 1] for i in range(len(Q_hw))])

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle(f"HPT-QRC on {d['platform']} — {d['dataset']}, "
                 f"{d['steps']} steps × {d['shots']} shots, {d['photons']} photons")

    ax = axes[0, 0]
    ax.plot(y, "k-", lw=1.2, label="true")
    ax.plot(sim_pred, "-", lw=1.0, label=f"sim features (MSE {d['mse_sim']:.2e})")
    ax.plot(hw_pred, "-", lw=1.0, label=f"HW features (MSE {d['mse_hw']:.2e})")
    ax.set_title("Forecast, sim-trained readout")
    ax.set_xlabel("test step")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(corr, ".-", ms=3, lw=0.7)
    ax.axhline(np.mean(corr), color="r", ls="--", lw=0.8,
               label=f"mean {np.mean(corr):.3f}")
    ax.set_title("Per-step feature correlation (sim vs HW)")
    ax.set_xlabel("test step")
    ax.set_ylim(min(0, corr.min()) - 0.05, 1.02)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(Q_sim.ravel(), Q_hw.ravel(), ".", ms=2, alpha=0.4)
    lim = [0, max(Q_sim.max(), Q_hw.max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_title("Feature values, pooled")
    ax.set_xlabel("simulated"); ax.set_ylabel("hardware")

    ax = axes[1, 1]
    bars = ax.bar(["sim features", "HW features"], [d["mse_sim"], d["mse_hw"]],
                  color=["tab:blue", "tab:orange"])
    ax.bar_label(bars, fmt="%.2e", fontsize=8)
    ax.set_title(f"Test MSE (ratio {d['mse_hw'] / d['mse_sim']:.2f}×)")

    fig.tight_layout()
    out = Path(path).with_suffix(".png")
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
