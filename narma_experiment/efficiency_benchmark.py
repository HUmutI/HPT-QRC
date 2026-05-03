"""
efficiency_benchmark.py
========================
Measures wall-clock training time and model size for each model.

Saves:
  results/efficiency_table.csv
  results/efficiency_plot.png
"""

import os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from data_loader import load_narma10, load_sp500
from classical_baselines import ARModel, HARModel, HARXModel, LSTMWrapper, RCModel, ClassicalContextRidge
from multi_qrc import HPT_QRC_Multi

os.makedirs("results", exist_ok=True)

# Use NARMA10 for timing (fixed dataset)
X_tr, y_tr, X_te, y_te = load_narma10(seed=42)

def time_model(name, fit_fn, predict_fn, n_reps=3):
    """Average training time over n_reps runs."""
    fit_times, mses = [], []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        model = fit_fn()
        fit_times.append(time.perf_counter() - t0)
        p = predict_fn(model)
        mses.append(mean_squared_error(y_te, p))
    return np.mean(fit_times), np.std(fit_times), np.mean(mses)


rows = []

print("Timing AR(3)...")
m, s, mse = time_model("AR3",
    lambda: ARModel(lags=3).fit(y_tr),
    lambda m: m.predict(y_te))
rows.append({"Model": "AR(3)", "Train_s_mean": m, "Train_s_std": s, "MSE": mse, "Epochs": "N/A (closed-form)"})

print("Timing HAR...")
m, s, mse = time_model("HAR",
    lambda: HARModel().fit(y_tr),
    lambda m: m.predict(y_te))
rows.append({"Model": "HAR", "Train_s_mean": m, "Train_s_std": s, "MSE": mse, "Epochs": "N/A (closed-form)"})

print("Timing Classical-Ridge...")
m, s, mse = time_model("Classical-Ridge",
    lambda: ClassicalContextRidge(window=5).fit(y_tr),
    lambda m: m.predict(y_te))
rows.append({"Model": "Classical-Ridge", "Train_s_mean": m, "Train_s_std": s, "MSE": mse, "Epochs": "N/A (closed-form)"})

print("Timing RC (ESN)...")
m, s, mse = time_model("RC",
    lambda: RCModel(in_size=1, seed=42).fit(y_tr),
    lambda m: m.predict(y_te))
rows.append({"Model": "RC (ESN)", "Train_s_mean": m, "Train_s_std": s, "MSE": mse, "Epochs": "N/A (closed-form)"})

print("Timing LSTM...")
m, s, mse = time_model("LSTM",
    lambda: LSTMWrapper(input_dim=1).fit(y_tr, y_tr),
    lambda m: m.predict(y_te, y_te), n_reps=1)
rows.append({"Model": "LSTM", "Train_s_mean": m, "Train_s_std": 0.0, "MSE": mse, "Epochs": "100 epochs"})

print("Timing HPT-QRC...")
m, s, mse = time_model("HPT-QRC",
    lambda: HPT_QRC_Multi(in_size=1, window=5, seed=42).fit(y_tr),
    lambda m: m.predict(y_te), n_reps=2)
rows.append({"Model": "HPT-QRC (ours)", "Train_s_mean": m, "Train_s_std": s, "MSE": mse, "Epochs": "N/A (closed-form)"})

df = pd.DataFrame(rows)
df.to_csv("results/efficiency_table.csv", index=False)

print("\n=== Training Efficiency ===")
for _, row in df.iterrows():
    print(f"  {row['Model']:<22} {row['Train_s_mean']*1000:.1f} ms ± {row['Train_s_std']*1000:.1f} ms   MSE={row['MSE']:.6f}")

# -----------------------------------------------------------------------
# Plot: training time vs MSE (scatter) + bar chart
# -----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: horizontal bar chart of training time (log scale)
ax = axes[0]
colors = ["#ef4444" if "HPT-QRC" in r else "#3b82f6" for r in df["Model"]]
bars = ax.barh(df["Model"], df["Train_s_mean"] * 1000, xerr=df["Train_s_std"] * 1000,
               color=colors, edgecolor="white", capsize=4)
ax.set_xlabel("Training Time (ms)", fontsize=12)
ax.set_title("Training Speed Comparison", fontsize=13, fontweight="bold")
ax.set_xscale("log")
ax.grid(axis="x", alpha=0.3)
for bar, val in zip(bars, df["Train_s_mean"] * 1000):
    ax.text(val * 1.1, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}ms", va="center", fontsize=9)

# Right: MSE vs training time scatter
ax2 = axes[1]
for _, row in df.iterrows():
    color = "#ef4444" if "HPT-QRC" in row["Model"] else "#3b82f6"
    ax2.scatter(row["Train_s_mean"] * 1000, row["MSE"],
                color=color, s=100, zorder=5)
    ax2.annotate(row["Model"], (row["Train_s_mean"] * 1000, row["MSE"]),
                 textcoords="offset points", xytext=(5, 3), fontsize=8)
ax2.set_xlabel("Training Time (ms, log scale)", fontsize=12)
ax2.set_ylabel("MSE (NARMA10)", fontsize=12)
ax2.set_title("Speed vs Accuracy Trade-off", fontsize=13, fontweight="bold")
ax2.set_xscale("log")
ax2.grid(alpha=0.3)

legend_patches = [
    plt.scatter([], [], color="#ef4444", s=80, label="HPT-QRC (ours)"),
    plt.scatter([], [], color="#3b82f6", s=80, label="Classical baseline"),
]
ax2.legend(handles=legend_patches, fontsize=10)

plt.tight_layout()
plt.savefig("results/efficiency_plot.png", dpi=150)
plt.close()
print("\n[✓] Saved: results/efficiency_table.csv")
print("[✓] Saved: results/efficiency_plot.png")
