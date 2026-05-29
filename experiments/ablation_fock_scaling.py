"""
ablation_fock_scaling.py
========================
Clean scaling story for HPT-QRC: how does forecast skill scale with the
combinatorial size of the Fock-space output?

We jointly sweep (n_photons, n_modes) along two axes:

  A. Vary n_photons in {1, 2, 3, 4, 5} at fixed n_modes = 8.
     Output Fock-space dim in the *unbunched* subspace is
        binom(m, n) = binom(8, n)
     Sequence: 8, 28, 56, 70, 56  (n=1..5).

  B. Vary n_modes in {4, 6, 8, 10, 12} at fixed n_photons = 3.
     binom(m, 3) = 4, 20, 56, 120, 220.

For each (n, m), report NRMSE on NARMA-10 + Mackey-Glass and QLIKE on
S&P 500 RV. Plot NRMSE vs. Fock-space dim on log-x to show monotonic
scaling (the previous m-only sweep at fixed n=3 was flat because at
fixed photon count varying mode count above ~8 has marginal effect on
the *useful* Fock-subspace dimension; jointly varying n is what moves
the dial).

Outputs:
  results/ablation_fock_scaling_photon.{csv,png}
  results/ablation_fock_scaling_mode.{csv,png}
  results/ablation_fock_scaling_combined.png
"""

from __future__ import annotations

import argparse
import os
import math

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


def _qlike(y, p, eps=1e-8):
    y = np.abs(np.asarray(y).flatten()) + eps
    p = np.abs(np.asarray(p).flatten()) + eps
    r = y / p
    return float(np.mean(r - np.log(r) - 1.0))


def _nrmse(y, p):
    y = np.asarray(y).flatten()
    p = np.asarray(p).flatten()
    s = y.std()
    return float(np.sqrt(np.mean((y - p) ** 2)) / (s + 1e-12))


def _fock_dim(n_photons: int, n_modes: int) -> int:
    return math.comb(n_modes, n_photons)


def _run(photon_list, n_modes, dataset, seed, lex_out=10,
         n_virtual_nodes=3, use_har=False, log_space=False):
    if dataset == "NARMA10":
        _, y_tr, _, y_te = load_narma10(seed=seed)
    elif dataset == "Mackey_Glass":
        _, y_tr, _, y_te = load_mackey_glass(seed=seed)
    else:
        y_tr, y_te, _, _ = load_sp500()
    qrc = HPT_QRC_Multi(
        in_size=1, window=10,
        photon_list=photon_list,
        n_virtual_nodes=n_virtual_nodes,
        lex_out=lex_out,
        seed=seed, use_har_context=use_har,
    )
    if n_modes is not None and n_modes != qrc.n_modes:
        qrc.n_modes = n_modes
        qrc.n_input_modes = min(qrc.n_input_modes, n_modes)
        qrc.n_memory_modes = max(1, n_modes - qrc.n_input_modes)
        qrc.reservoirs = qrc._build_reservoirs(seed)
    qrc.fit(y_tr)
    pred = qrc.predict(y_te)
    if log_space:
        y_te_eval = np.exp(y_te)
        pred_eval = np.exp(pred)
    else:
        y_te_eval = y_te
        pred_eval = pred
    return {
        "MSE": float(mean_squared_error(y_te_eval, pred_eval)),
        "NRMSE": _nrmse(y_te_eval, pred_eval),
        "QLIKE": _qlike(y_te_eval, pred_eval),
    }


def _sweep(rows_axis, seeds=(42, 0, 1)):
    """Aggregate mean/std across seeds for each row config."""
    out = []
    for cfg in rows_axis:
        scores = []
        for seed in seeds:
            s = _run(
                photon_list=cfg["photon_list"],
                n_modes=cfg["n_modes"],
                dataset=cfg["dataset"],
                seed=seed,
                use_har=cfg["use_har"],
                log_space=cfg["log_space"],
            )
            scores.append(s)
        df = pd.DataFrame(scores)
        out.append({
            **{k: v for k, v in cfg.items() if k not in ("use_har", "log_space")},
            "fock_dim": _fock_dim(cfg["photon_list"][0], cfg["n_modes"] or 8),
            "MSE_mean": df.MSE.mean(), "MSE_std": df.MSE.std(),
            "NRMSE_mean": df.NRMSE.mean(), "NRMSE_std": df.NRMSE.std(),
            "QLIKE_mean": df.QLIKE.mean(), "QLIKE_std": df.QLIKE.std(),
        })
    return pd.DataFrame(out)


def ablation_photon(out_dir: str, seeds, n_modes: int = 8):
    cfgs = []
    for n_ph in [1, 2, 3, 4, 5]:
        if n_ph > n_modes:
            continue
        for ds, use_har, log_sp in [
            ("NARMA10", False, False),
            ("Mackey_Glass", False, False),
            ("SP500_RV", True, True),
        ]:
            cfgs.append({
                "photon_list": [n_ph],
                "n_modes": n_modes,
                "dataset": ds,
                "use_har": use_har,
                "log_space": log_sp,
            })
    df = _sweep(cfgs, seeds=seeds)
    df.to_csv(f"{out_dir}/ablation_fock_scaling_photon.csv", index=False)
    return df


def ablation_mode(out_dir: str, seeds, n_photons: int = 3):
    cfgs = []
    for m in [4, 6, 8, 10, 12]:
        if n_photons > m:
            continue
        for ds, use_har, log_sp in [
            ("NARMA10", False, False),
            ("Mackey_Glass", False, False),
            ("SP500_RV", True, True),
        ]:
            cfgs.append({
                "photon_list": [n_photons],
                "n_modes": m,
                "dataset": ds,
                "use_har": use_har,
                "log_space": log_sp,
            })
    df = _sweep(cfgs, seeds=seeds)
    df.to_csv(f"{out_dir}/ablation_fock_scaling_mode.csv", index=False)
    return df


def _plot_axis(ax, df, ds, ycol, xlabel, title):
    sub = df[df.dataset == ds].sort_values("fock_dim")
    ax.errorbar(sub.fock_dim, sub[ycol],
                 yerr=sub[ycol.replace("mean", "std")],
                 marker="o", linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ycol.replace("_mean", ""))
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")


def plot(df_photon: pd.DataFrame, df_mode: pd.DataFrame, out_dir: str):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    # Photon-axis row
    for ax, ds, ycol, lab in zip(
        axes[0],
        ["NARMA10", "Mackey_Glass", "SP500_RV"],
        ["NRMSE_mean", "NRMSE_mean", "QLIKE_mean"],
        ["NRMSE", "NRMSE", "QLIKE"],
    ):
        _plot_axis(ax, df_photon, ds, ycol,
                    f"Fock dim $\\binom{{m}}{{n}}$ (vary $n$, $m=8$)",
                    f"{ds} — photon scan")
    # Mode-axis row
    for ax, ds, ycol in zip(
        axes[1],
        ["NARMA10", "Mackey_Glass", "SP500_RV"],
        ["NRMSE_mean", "NRMSE_mean", "QLIKE_mean"],
    ):
        _plot_axis(ax, df_mode, ds, ycol,
                    f"Fock dim $\\binom{{m}}{{n}}$ (vary $m$, $n=3$)",
                    f"{ds} — mode scan")
    fig.suptitle("Fock-space dimension scaling of HPT-QRC", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/ablation_fock_scaling_combined.png", dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 1])
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    seeds = tuple(args.seeds)
    os.makedirs(args.out, exist_ok=True)
    print("=== Photon-count scan at fixed m=8 ===")
    df_p = ablation_photon(args.out, seeds, n_modes=8)
    print(df_p[["photon_list", "n_modes", "dataset", "fock_dim",
                "NRMSE_mean", "QLIKE_mean"]].to_string(index=False))
    print("\n=== Mode-count scan at fixed n=3 ===")
    df_m = ablation_mode(args.out, seeds, n_photons=3)
    print(df_m[["photon_list", "n_modes", "dataset", "fock_dim",
                "NRMSE_mean", "QLIKE_mean"]].to_string(index=False))
    plot(df_p, df_m, args.out)
    print(f"\n[✓] {args.out}/ablation_fock_scaling_combined.png")


if __name__ == "__main__":
    main()
