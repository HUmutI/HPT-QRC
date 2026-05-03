"""
ablation_study.py
=================
Full ablation study for HPT-QRC architecture choices on NARMA10.

Sweeps:
  - n_photons      : [1, 2, 3, 4]           (homogeneous)
  - n_reservoirs   : [1, 2, 3, 5]
  - window_size    : [3, 5, 10, 15]
  - photon ensemble: homogeneous [3,3,3] vs heterogeneous [2,3,4]

Saves:
  results/ablation_photons.csv / .png
  results/ablation_reservoirs.csv / .png
  results/ablation_window.csv / .png
  results/ablation_ensemble.csv / .png
  results/ablation_combined.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from data_loader import load_narma10
from multi_qrc import HPT_QRC_Multi

os.makedirs("results", exist_ok=True)

N_SEEDS = 3
SEEDS   = [42, 0, 1]

def run_config(cfg, seeds=SEEDS):
    """Run a single HPT-QRC config over multiple seeds. Returns mean/std MSE."""
    mses = []
    for seed in seeds:
        X_tr, y_tr, X_te, y_te = load_narma10(seed=seed)
        qrc = HPT_QRC_Multi(**cfg, seed=seed)
        qrc.fit(y_tr)
        p = qrc.predict(y_te)
        mses.append(mean_squared_error(y_te, p))
    return np.mean(mses), np.std(mses)


def bar_plot(labels, means, stds, title, xlabel, save_path, highlight_idx=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#3b82f6"] * len(labels)
    if highlight_idx is not None:
        colors[highlight_idx] = "#10b981"   # green = current default
    bars = ax.bar(labels, means, yerr=stds, capsize=5,
                  color=colors, edgecolor="white", linewidth=1.2)
    ax.bar_label(bars, labels=[f"{m:.5f}" for m in means],
                 padding=3, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("MSE (NARMA10)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# -----------------------------------------------------------------------
# 1. n_photons sweep
# -----------------------------------------------------------------------
print("=== Ablation: n_photons ===")
photon_vals = [1, 2, 3, 4]
rows = []
for n_ph in photon_vals:
    base = dict(in_size=1, window=5, n_photons=n_ph, n_reservoirs=3, n_virtual_nodes=3)
    m, s = run_config(base)
    print(f"  n_photons={n_ph}  MSE={m:.6f} ± {s:.6f}")
    rows.append({"n_photons": n_ph, "MSE_mean": m, "MSE_std": s})

df_ph = pd.DataFrame(rows)
df_ph.to_csv("results/ablation_photons.csv", index=False)
bar_plot([str(v) for v in photon_vals], df_ph.MSE_mean, df_ph.MSE_std,
         "Ablation: Number of Photons", "n_photons",
         "results/ablation_photons.png", highlight_idx=2)

# -----------------------------------------------------------------------
# 2. n_reservoirs sweep
# -----------------------------------------------------------------------
print("\n=== Ablation: n_reservoirs ===")
res_vals = [1, 2, 3, 5]
rows = []
for n_res in res_vals:
    base = dict(in_size=1, window=5, n_photons=3, n_reservoirs=n_res, n_virtual_nodes=3)
    m, s = run_config(base)
    print(f"  n_reservoirs={n_res}  MSE={m:.6f} ± {s:.6f}")
    rows.append({"n_reservoirs": n_res, "MSE_mean": m, "MSE_std": s})

df_res = pd.DataFrame(rows)
df_res.to_csv("results/ablation_reservoirs.csv", index=False)
bar_plot([str(v) for v in res_vals], df_res.MSE_mean, df_res.MSE_std,
         "Ablation: Number of Reservoirs", "n_reservoirs",
         "results/ablation_reservoirs.png", highlight_idx=2)

# -----------------------------------------------------------------------
# 3. window size sweep
# -----------------------------------------------------------------------
print("\n=== Ablation: window size ===")
win_vals = [3, 5, 10, 15]
rows = []
for w in win_vals:
    base = dict(in_size=1, window=w, n_photons=3, n_reservoirs=3, n_virtual_nodes=3)
    m, s = run_config(base)
    print(f"  window={w}  MSE={m:.6f} ± {s:.6f}")
    rows.append({"window": w, "MSE_mean": m, "MSE_std": s})

df_win = pd.DataFrame(rows)
df_win.to_csv("results/ablation_window.csv", index=False)
bar_plot([str(v) for v in win_vals], df_win.MSE_mean, df_win.MSE_std,
         "Ablation: Window Size", "window",
         "results/ablation_window.png", highlight_idx=1)

# -----------------------------------------------------------------------
# 4. Homogeneous vs Heterogeneous photon ensemble
# -----------------------------------------------------------------------
print("\n=== Ablation: Ensemble Type ===")
ensembles = {
    "Homo [3,3,3]": dict(in_size=1, window=5, n_photons=3, n_reservoirs=3, n_virtual_nodes=3),
    "Hetero [2,3,4]": dict(in_size=1, window=5, photon_list=[2, 3, 4], n_virtual_nodes=3),
}
rows = []
for label, cfg in ensembles.items():
    m, s = run_config(cfg)
    print(f"  {label}  MSE={m:.6f} ± {s:.6f}")
    rows.append({"ensemble": label, "MSE_mean": m, "MSE_std": s})

df_ens = pd.DataFrame(rows)
df_ens.to_csv("results/ablation_ensemble.csv", index=False)
bar_plot(df_ens.ensemble, df_ens.MSE_mean, df_ens.MSE_std,
         "Ablation: Homogeneous vs Heterogeneous Photons", "Ensemble Type",
         "results/ablation_ensemble.png")

# -----------------------------------------------------------------------
# Combined 2×2 summary figure
# -----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

def _panel(ax, df, x_col, title, xlabel, default_val=None):
    xs = df[x_col].astype(str).tolist()
    colors = ["#10b981" if str(v) == str(default_val) else "#3b82f6"
              for v in df[x_col]]
    bars = ax.bar(xs, df.MSE_mean, yerr=df.MSE_std, capsize=5,
                  color=colors, edgecolor="white")
    ax.bar_label(bars, labels=[f"{m:.5f}" for m in df.MSE_mean],
                 padding=2, fontsize=8)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("MSE (NARMA10)")
    ax.grid(axis="y", alpha=0.3)

_panel(axes[0, 0], df_ph,  "n_photons",   "n_photons Sweep",   "n_photons",   default_val=3)
_panel(axes[0, 1], df_res, "n_reservoirs","n_reservoirs Sweep","n_reservoirs",default_val=3)
_panel(axes[1, 0], df_win, "window",       "Window Size Sweep", "window",       default_val=5)

ax_ens = axes[1, 1]
bar_colors = ["#3b82f6", "#8b5cf6"]
bars = ax_ens.bar(df_ens.ensemble, df_ens.MSE_mean, yerr=df_ens.MSE_std,
                  capsize=5, color=bar_colors, edgecolor="white")
ax_ens.bar_label(bars, labels=[f"{m:.5f}" for m in df_ens.MSE_mean],
                 padding=2, fontsize=9)
ax_ens.set_title("Ensemble Type", fontweight="bold")
ax_ens.set_ylabel("MSE (NARMA10)")
ax_ens.grid(axis="y", alpha=0.3)

# Green = our chosen default
legend_patches = [
    plt.Rectangle((0,0),1,1, color="#10b981", label="Current default"),
    plt.Rectangle((0,0),1,1, color="#3b82f6", label="Other configs"),
]
fig.legend(handles=legend_patches, loc="lower center", ncol=2, fontsize=11,
           bbox_to_anchor=(0.5, 0.0))
plt.suptitle("HPT-QRC Ablation Study (NARMA10, mean±std over 3 seeds)",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("results/ablation_combined.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n[✓] Saved all ablation plots and CSVs to results/")
