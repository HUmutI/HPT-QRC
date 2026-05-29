"""
ablation_matched_dim.py
=======================
Phase 2.1 ablation per PROTOCOL.md §7.

Isolates the photon-number-ensemble axis from feature-dimension scaling
by reporting all configs at *matched total Fock-feature dimension*.
Sweeps in two directions:

  A. photon_list \in {[2], [3], [4], [2,3], [3,4], [2,4], [2,3,4]}
     mode count fixed (n_modes = 8), virtual depth tuned so total
     output dim is constant (target dim = D_target).

  B. mode count m \in {6, 8, 10, 12} at fixed n_photons = 3,
     virtual depth fixed at 3, lex_out fixed at 10.

NRMSE on NARMA-10 + Mackey-Glass, QLIKE on S&P 500 RV.

Saves:
  results/ablation_matched_photons.csv
  results/ablation_matched_modes.csv
  results/ablation_matched_photons.png
  results/ablation_matched_modes.png
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

import pathlib as _pathlib, sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent / 'src'))
del _pathlib, _sys
from data_loader import load_mackey_glass, load_narma10, load_sp500
from multi_qrc import HPT_QRC_Multi


def _qlike(yt, yp, eps=1e-8):
    y = np.abs(np.asarray(yt).flatten()) + eps
    p = np.abs(np.asarray(yp).flatten()) + eps
    r = y / p
    return float(np.mean(r - np.log(r) - 1.0))


def _nrmse(yt, yp):
    yt = np.asarray(yt).flatten()
    yp = np.asarray(yp).flatten()
    s = yt.std()
    return float(np.sqrt(np.mean((yt - yp) ** 2)) / (s + 1e-12))


def _run_qrc(photon_list, n_modes, n_virtual_nodes, lex_out,
             dataset, seeds=(42, 0, 1), use_har_context=False, log_space=False):
    scores = []
    for seed in seeds:
        if dataset == "NARMA10":
            _, y_tr, _, y_te = load_narma10(seed=seed)
        elif dataset == "Mackey_Glass":
            _, y_tr, _, y_te = load_mackey_glass(seed=seed)
        else:  # SP500
            y_tr, y_te, _, _ = load_sp500()
        qrc = HPT_QRC_Multi(
            in_size=1, window=10,
            photon_list=photon_list,
            n_virtual_nodes=n_virtual_nodes,
            lex_out=lex_out,
            seed=seed,
            use_har_context=use_har_context,
        )
        # Override n_modes (post-init), so output dim follows the photon/mode setup.
        # `_build_temporal_circuit` and `_build_reservoirs` are called in __init__,
        # so we re-init with the wanted mode count by piggybacking on the API.
        if n_modes is not None and n_modes != qrc.n_modes:
            qrc.n_modes = n_modes
            qrc.n_input_modes = min(qrc.n_input_modes, n_modes)
            qrc.n_memory_modes = max(1, n_modes - qrc.n_input_modes)
            qrc.reservoirs = qrc._build_reservoirs(seed)
        feature_dim = len(qrc.reservoirs) * lex_out
        qrc.fit(y_tr)
        pred = qrc.predict(y_te)
        if log_space:
            y_te_eval = np.exp(y_te)
            pred_eval = np.exp(pred)
        else:
            y_te_eval = y_te
            pred_eval = pred
        scores.append({
            "MSE": float(mean_squared_error(y_te_eval, pred_eval)),
            "NRMSE": _nrmse(y_te_eval, pred_eval),
            "QLIKE": _qlike(y_te_eval, pred_eval),
            "feature_dim": feature_dim,
        })
    df = pd.DataFrame(scores)
    return {
        "MSE_mean": df.MSE.mean(), "MSE_std": df.MSE.std(),
        "NRMSE_mean": df.NRMSE.mean(), "NRMSE_std": df.NRMSE.std(),
        "QLIKE_mean": df.QLIKE.mean(), "QLIKE_std": df.QLIKE.std(),
        "feature_dim": int(df.feature_dim.iloc[0]),
    }


def _matched_dim_grid(photon_lists, target_dim: int, n_virtual_nodes: int = 3):
    """Pick lex_out per config so n_res * n_virtual_nodes * lex_out = target_dim."""
    out = []
    for ph in photon_lists:
        n_res = len(ph)
        lex = max(1, round(target_dim / (n_res * n_virtual_nodes)))
        # Achieved dim:
        out.append((ph, lex, n_res * n_virtual_nodes * lex))
    return out


def ablation_photons(out_dir: str = "results", target_dim: int = 90,
                     seeds=(42, 0, 1)):
    """
    Sweep photon_list at matched total output dim ~= target_dim.
    """
    configs = _matched_dim_grid(
        photon_lists=[[2], [3], [4], [2, 3], [3, 4], [2, 4], [2, 3, 4]],
        target_dim=target_dim,
    )
    rows = []
    for ph, lex, achieved in configs:
        for dataset, log_sp, use_har in [
            ("NARMA10", False, False),
            ("Mackey_Glass", False, False),
            ("SP500_RV", True, True),
        ]:
            print(f"  photon_list={ph}, lex_out={lex}, dim≈{achieved}, dataset={dataset}")
            res = _run_qrc(ph, n_modes=None,
                            n_virtual_nodes=3, lex_out=lex,
                            dataset=dataset, seeds=seeds,
                            use_har_context=use_har, log_space=log_sp)
            rows.append({"photon_list": str(ph), "lex_out": lex, "dataset": dataset,
                         **res})
    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(f"{out_dir}/ablation_matched_photons.csv", index=False)
    _plot_photon_ablation(df, f"{out_dir}/ablation_matched_photons.png")
    return df


def ablation_modes(out_dir: str = "results", mode_vals=(4, 6, 8, 10),
                    seeds=(42, 0, 1)):
    rows = []
    for m in mode_vals:
        for dataset, log_sp, use_har in [
            ("NARMA10", False, False),
            ("Mackey_Glass", False, False),
            ("SP500_RV", True, True),
        ]:
            print(f"  n_modes={m}, dataset={dataset}")
            res = _run_qrc([3], n_modes=int(m),
                            n_virtual_nodes=3, lex_out=10,
                            dataset=dataset, seeds=seeds,
                            use_har_context=use_har, log_space=log_sp)
            rows.append({"n_modes": int(m), "dataset": dataset, **res})
    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(f"{out_dir}/ablation_matched_modes.csv", index=False)
    _plot_mode_ablation(df, f"{out_dir}/ablation_matched_modes.png")
    return df


def _plot_photon_ablation(df: pd.DataFrame, save_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, ds, ycol in zip(
        axes,
        ["NARMA10", "Mackey_Glass", "SP500_RV"],
        ["NRMSE_mean", "NRMSE_mean", "QLIKE_mean"],
    ):
        sub = df[df.dataset == ds].copy()
        labels = sub["photon_list"].tolist()
        means = sub[ycol].values
        stds = sub[ycol.replace("mean", "std")].values
        bars = ax.bar(labels, means, yerr=stds, capsize=4,
                      color="#3b82f6", edgecolor="white")
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(ycol.replace("_mean", ""))
        ax.set_title(f"{ds} (matched dim)")
        ax.grid(axis="y", alpha=0.3)
        for b, v in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, v,
                    f"{v:.3g}", ha="center", va="bottom", fontsize=7)
    fig.suptitle("Photon-Ensemble Ablation at Matched Feature Dimension",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _plot_mode_ablation(df: pd.DataFrame, save_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, ds, ycol in zip(
        axes,
        ["NARMA10", "Mackey_Glass", "SP500_RV"],
        ["NRMSE_mean", "NRMSE_mean", "QLIKE_mean"],
    ):
        sub = df[df.dataset == ds].copy().sort_values("n_modes")
        ax.errorbar(sub.n_modes.astype(int), sub[ycol],
                     yerr=sub[ycol.replace("mean", "std")],
                     marker="o", linewidth=2)
        ax.set_xlabel("Number of modes")
        ax.set_ylabel(ycol.replace("_mean", ""))
        ax.set_title(f"{ds}")
        ax.grid(alpha=0.3)
    fig.suptitle("Mode-Count Ablation (fixed n=3, virtual=3, lex_out=10)",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["photons", "modes", "both"], default="both")
    ap.add_argument("--target_dim", type=int, default=90)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 1])
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    seeds = tuple(args.seeds)
    if args.phase in ("photons", "both"):
        print("=== Photon-list sweep (matched dim) ===")
        ablation_photons(out_dir=args.out, target_dim=args.target_dim, seeds=seeds)
    if args.phase in ("modes", "both"):
        print("\n=== Mode-count sweep ===")
        ablation_modes(out_dir=args.out, seeds=seeds)


if __name__ == "__main__":
    main()
