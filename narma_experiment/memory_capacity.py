"""
memory_capacity.py
==================
Computes Jaeger (2001) Memory Capacity (MC) for HPT-QRC vs Classical ESN.

MC = sum_{k=1}^{K}  R²( u[t-k],  readout( reservoir_state[t] ) )

A high MC proves the reservoir retains a long history of past inputs —
the core theoretical claim of reservoir computing papers.

Saves:
  results/mc_curve.png      — MC vs lag plot
  results/mc_scores.csv     — MC summary table
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from multi_qrc import HPT_QRC_Multi
from esn_baseline import EchoStateNetwork

os.makedirs("results", exist_ok=True)

# -----------------------------------------------------------------------
# MC helper
# -----------------------------------------------------------------------

def _extract_esn_states(esn, u, discard=100):
    """Run ESN forward and return reservoir states (T-discard, design_matrix_size)."""
    data = u.flatten().reshape(-1, 1)
    states, _ = esn._run_reservoir(data)
    return states[discard:]


def compute_mc(name, get_states_fn, u_train, u_test, max_lag=40, discard=100):
    """
    get_states_fn(u) -> np.array(T, n_features)
    Returns list of R² values per lag.
    """
    u_all = np.concatenate([u_train, u_test])
    states_all = get_states_fn(u_all)          # (T_all - 1, F)
    u_trimmed  = u_all[1:len(states_all)+1]    # align targets

    # Split back into train/test on the state axis
    n_tr = len(u_train) - 1 - discard

    mc_scores = []
    for k in range(1, max_lag + 1):
        # Target: u[t-k] aligned with states[t]
        # valid range: state index k..end
        valid = states_all[k:]
        target = u_trimmed[:len(valid) - k + (max_lag - k)]
        # Simplest: use the train portion only
        # state at t predicts u[t-k], so skip first k rows
        s_tr = states_all[discard + k : discard + k + n_tr]
        t_tr = u_all[discard : discard + n_tr]   # u[t-k] for t=discard+k..

        if len(s_tr) == 0 or len(t_tr) == 0:
            mc_scores.append(0.0)
            continue

        min_len = min(len(s_tr), len(t_tr))
        s_tr = s_tr[:min_len]
        t_tr = t_tr[:min_len]

        try:
            ridge = Ridge(alpha=1e-4)
            ridge.fit(s_tr, t_tr)
            # Evaluate on test states
            s_te = states_all[discard + k + n_tr : discard + k + n_tr + (len(u_test) - k)]
            t_te = u_all[discard + n_tr : discard + n_tr + len(s_te)]
            min_te = min(len(s_te), len(t_te))
            if min_te < 2:
                mc_scores.append(0.0)
                continue
            p_te = ridge.predict(s_te[:min_te])
            r2 = max(0.0, r2_score(t_te[:min_te], p_te))
            mc_scores.append(r2)
        except Exception:
            mc_scores.append(0.0)

    return mc_scores


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    np.random.seed(42)
    N_TRAIN = 1500
    N_TEST  = 300
    MAX_LAG = 40
    DISCARD = 100

    u_train = np.random.uniform(0, 1, N_TRAIN)
    u_test  = np.random.uniform(0, 1, N_TEST)

    # --- HPT-QRC states ---
    print("Computing HPT-QRC features for MC analysis...")
    qrc = HPT_QRC_Multi(in_size=1, window=5, seed=42, use_har_context=False)

    def qrc_states(u):
        y = u.reshape(-1, 1)
        return qrc.get_features(y)

    mc_qrc = compute_mc("HPT-QRC", qrc_states, u_train, u_test,
                         max_lag=MAX_LAG, discard=DISCARD)

    # --- HPT-QRC Heterogeneous (2+3+4 photons) ---
    print("Computing HPT-QRC-Hetero features for MC analysis...")
    qrc_hetero = HPT_QRC_Multi(in_size=1, window=5, seed=42,
                                use_har_context=False, photon_list=[2, 3, 4])

    def qrc_hetero_states(u):
        y = u.reshape(-1, 1)
        return qrc_hetero.get_features(y)

    mc_hetero = compute_mc("HPT-QRC-Hetero", qrc_hetero_states, u_train, u_test,
                            max_lag=MAX_LAG, discard=DISCARD)

    # --- Classical ESN states ---
    print("Computing Classical ESN states for MC analysis...")
    esn = EchoStateNetwork(in_size=1, res_size=100, seed=42)

    def esn_states(u):
        return _extract_esn_states(esn, u, discard=DISCARD)

    mc_esn = compute_mc("ESN", esn_states, u_train, u_test,
                         max_lag=MAX_LAG, discard=DISCARD)

    # --- Random Linear Projection (baseline) ---
    np.random.seed(0)
    W_rand = np.random.randn(100, 5)

    def rand_states(u):
        y = u.reshape(-1, 1)
        # simple sliding window → random projection
        n = len(y)
        feats = []
        for t in range(5, n):
            win = y[t-5:t].flatten()
            feats.append(W_rand @ win)
        return np.array(feats)

    mc_rand = compute_mc("Random-Linear", rand_states, u_train, u_test,
                          max_lag=MAX_LAG, discard=DISCARD)

    # -----------------------------------------------------------------------
    # Summarise
    # -----------------------------------------------------------------------
    lags = np.arange(1, MAX_LAG + 1)

    total_mc = {
        "HPT-QRC (3 photons)":      sum(mc_qrc),
        "HPT-QRC-Hetero (2+3+4)":   sum(mc_hetero),
        "Classical ESN (100 units)": sum(mc_esn),
        "Random-Linear":             sum(mc_rand),
    }

    print("\n=== Memory Capacity Summary ===")
    for name, mc in total_mc.items():
        print(f"  {name:<30}  MC = {mc:.4f}")

    df_mc = pd.DataFrame({
        "Lag": lags,
        "HPT-QRC":       mc_qrc,
        "HPT-QRC-Hetero": mc_hetero,
        "ESN":           mc_esn,
        "Random-Linear": mc_rand,
    })
    df_mc.to_csv("results/mc_scores.csv", index=False)

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: MC curve per lag
    ax = axes[0]
    ax.plot(lags, mc_qrc,    label="HPT-QRC (3 photons)",      color="#3b82f6", linewidth=2)
    ax.plot(lags, mc_hetero, label="HPT-QRC-Hetero (2+3+4)",   color="#8b5cf6", linewidth=2, linestyle="--")
    ax.plot(lags, mc_esn,    label="Classical ESN (100 units)", color="#f59e0b", linewidth=2)
    ax.plot(lags, mc_rand,   label="Random-Linear",             color="#6b7280", linewidth=1.5, linestyle=":")
    ax.set_xlabel("Lag k", fontsize=12)
    ax.set_ylabel("R² (recall accuracy)", fontsize=12)
    ax.set_title("Memory Capacity per Lag", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    # Right: total MC bar chart
    ax2 = axes[1]
    names = list(total_mc.keys())
    vals  = list(total_mc.values())
    colors = ["#3b82f6", "#8b5cf6", "#f59e0b", "#6b7280"]
    bars = ax2.bar(names, vals, color=colors, edgecolor="white", linewidth=1.2)
    ax2.bar_label(bars, fmt="%.2f", padding=3, fontsize=11)
    ax2.set_ylabel("Total Memory Capacity (MC)", fontsize=12)
    ax2.set_title("Total MC Comparison", fontsize=13, fontweight="bold")
    ax2.set_ylim(0, max(vals) * 1.2)
    ax2.tick_params(axis="x", rotation=15)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/mc_curve.png", dpi=150)
    plt.close()
    print("\n[✓] Saved: results/mc_curve.png")
    print("[✓] Saved: results/mc_scores.csv")


if __name__ == "__main__":
    main()
