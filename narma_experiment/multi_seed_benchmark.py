"""
multi_seed_benchmark.py
=======================
Runs the full HPT-QRC benchmark across multiple random seeds and reports
mean ± std for every model/metric combination.

Usage:
    conda run -n quandela python multi_seed_benchmark.py

This is the primary script for generating publication-ready numbers.
Results are saved to results/multi_seed_summary.csv and printed as a
LaTeX-ready table.
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

from data_loader import load_narma10, load_mackey_glass, load_sp500
from classical_baselines import (
    ARModel, ARMAXModel, HARModel, HARXModel,
    LSTMWrapper, RCModel, ClassicalContextRidge
)
from multi_qrc import HPT_QRC_Multi

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def qlike_loss(y_true, y_pred):
    eps = 1e-8
    y_t = np.abs(np.array(y_true).flatten()) + eps
    y_p = np.abs(np.array(y_pred).flatten()) + eps
    ratio = y_t / y_p
    return np.sum(ratio - np.log(ratio) - 1)


# ---------------------------------------------------------------------------
# Single-seed benchmark runner
# ---------------------------------------------------------------------------

def run_one_seed(seed):
    """
    Run the full benchmark (NARMA10, Mackey-Glass, S&P 500) for one seed.
    Returns a dict: { dataset_name -> pd.DataFrame of (Model, MSE, QLIKE) }
    """
    results = {}

    # ------------------------------------------------------------------
    # 1. NARMA10
    # ------------------------------------------------------------------
    X_tr, y_tr, X_te, y_te = load_narma10(seed=seed)

    def qlike_narma(yt, yp): return qlike_loss(yt, yp)

    rows = []
    preds = {}

    ar1 = ARModel(lags=1).fit(y_tr);            p = ar1.predict(y_te);         preds["AR1"] = p;             rows.append(("AR1",            mean_squared_error(y_te, p), qlike_narma(y_te, p)))
    ar3 = ARModel(lags=3).fit(y_tr);            p = ar3.predict(y_te);         preds["AR3"] = p;             rows.append(("AR3",            mean_squared_error(y_te, p), qlike_narma(y_te, p)))
    har = HARModel().fit(y_tr);                 p = har.predict(y_te);         preds["HAR"] = p;             rows.append(("HAR",            mean_squared_error(y_te, p), qlike_narma(y_te, p)))
    rc  = RCModel(in_size=1, seed=seed).fit(y_tr); p = rc.predict(y_te);      preds["RC"] = p;              rows.append(("RC",             mean_squared_error(y_te, p), qlike_narma(y_te, p)))
    ccr = ClassicalContextRidge(window=5).fit(y_tr); p = ccr.predict(y_te);   preds["Classical-Ridge"] = p; rows.append(("Classical-Ridge",mean_squared_error(y_te, p), qlike_narma(y_te, p)))
    qrc = HPT_QRC_Multi(in_size=1, window=10, photon_list=[2,3,4], seed=seed).fit(y_tr); p = qrc.predict(y_te); preds["HPT-QRC"] = p; rows.append(("HPT-QRC", mean_squared_error(y_te, p), qlike_narma(y_te, p)))

    # Exogenous
    harx = HARXModel().fit(y_tr, X_tr); p = harx.predict(y_te, X_te); preds["HARX"] = p; rows.append(("HARX", mean_squared_error(y_te, p), qlike_narma(y_te, p)))
    qrc_in = 1 + X_tr.shape[1]
    qrcx = HPT_QRC_Multi(in_size=qrc_in, window=10, photon_list=[2,3,4], seed=seed).fit(y_tr, X_tr)
    p = qrcx.predict(y_te, X_te); preds["HPT-QRC-X"] = p; rows.append(("HPT-QRC-X", mean_squared_error(y_te, p), qlike_narma(y_te, p)))

    results["NARMA10"] = pd.DataFrame(rows, columns=["Model", "MSE", "QLIKE"])

    # ------------------------------------------------------------------
    # 2. Mackey-Glass (17-step-ahead)
    # ------------------------------------------------------------------
    X_tr, y_tr, X_te, y_te = load_mackey_glass(seed=seed)

    rows = []
    ar1 = ARModel(lags=1).fit(y_tr);   p = ar1.predict(y_te);  rows.append(("AR1",            mean_squared_error(y_te, p), qlike_loss(y_te, p)))
    ar3 = ARModel(lags=3).fit(y_tr);   p = ar3.predict(y_te);  rows.append(("AR3",            mean_squared_error(y_te, p), qlike_loss(y_te, p)))
    har = HARModel().fit(y_tr);        p = har.predict(y_te);  rows.append(("HAR",            mean_squared_error(y_te, p), qlike_loss(y_te, p)))
    rc  = RCModel(in_size=1, seed=seed).fit(y_tr); p = rc.predict(y_te); rows.append(("RC", mean_squared_error(y_te, p), qlike_loss(y_te, p)))
    ccr = ClassicalContextRidge(window=5).fit(y_tr); p = ccr.predict(y_te); rows.append(("Classical-Ridge", mean_squared_error(y_te, p), qlike_loss(y_te, p)))
    qrc = HPT_QRC_Multi(in_size=1, window=10, photon_list=[2,3,4], seed=seed).fit(y_tr); p = qrc.predict(y_te); rows.append(("HPT-QRC", mean_squared_error(y_te, p), qlike_loss(y_te, p)))

    harx = HARXModel().fit(y_tr, X_tr); p = harx.predict(y_te, X_te); rows.append(("HARX", mean_squared_error(y_te, p), qlike_loss(y_te, p)))
    qrc_in = 1 + X_tr.shape[1]
    qrcx = HPT_QRC_Multi(in_size=qrc_in, window=10, photon_list=[2,3,4], seed=seed).fit(y_tr, X_tr)
    p = qrcx.predict(y_te, X_te); rows.append(("HPT-QRC-X", mean_squared_error(y_te, p), qlike_loss(y_te, p)))

    results["Mackey_Glass"] = pd.DataFrame(rows, columns=["Model", "MSE", "QLIKE"])

    # ------------------------------------------------------------------
    # 3. S&P 500 Realized Volatility
    # ------------------------------------------------------------------
    y_tr, y_te, X_tr, X_te = load_sp500()

    def qlike_sp(yt, yp):
        # Data is log-RV — exponentiate before QLIKE evaluation
        return qlike_loss(np.exp(yt), np.exp(yp))

    rows = []
    ar1 = ARModel(lags=1).fit(y_tr);   p = ar1.predict(y_te);  rows.append(("AR1",            mean_squared_error(y_te, p), qlike_sp(y_te, p)))
    ar3 = ARModel(lags=3).fit(y_tr);   p = ar3.predict(y_te);  rows.append(("AR3",            mean_squared_error(y_te, p), qlike_sp(y_te, p)))
    har = HARModel().fit(y_tr);        p = har.predict(y_te);  rows.append(("HAR",            mean_squared_error(y_te, p), qlike_sp(y_te, p)))
    rc  = RCModel(in_size=1, seed=seed).fit(y_tr); p = rc.predict(y_te); rows.append(("RC", mean_squared_error(y_te, p), qlike_sp(y_te, p)))
    ccr = ClassicalContextRidge(window=5).fit(y_tr); p = ccr.predict(y_te); rows.append(("Classical-Ridge", mean_squared_error(y_te, p), qlike_sp(y_te, p)))
    qrc = HPT_QRC_Multi(in_size=1, window=10, photon_list=[2,3,4], seed=seed, use_har_context=True).fit(y_tr)
    p = qrc.predict(y_te); rows.append(("HPT-QRC", mean_squared_error(y_te, p), qlike_sp(y_te, p)))

    harx = HARXModel().fit(y_tr, X_tr); p = harx.predict(y_te, X_te); rows.append(("HARX", mean_squared_error(y_te, p), qlike_sp(y_te, p)))
    qrc_in = 1 + X_tr.shape[1]
    qrcx = HPT_QRC_Multi(in_size=qrc_in, window=10, photon_list=[2,3,4], seed=seed, use_har_context=True).fit(y_tr, X_tr)
    p = qrcx.predict(y_te, X_te); rows.append(("HPT-QRC-X", mean_squared_error(y_te, p), qlike_sp(y_te, p)))

    results["SP500_RV"] = pd.DataFrame(rows, columns=["Model", "MSE", "QLIKE"])

    return results


# ---------------------------------------------------------------------------
# Multi-seed aggregation
# ---------------------------------------------------------------------------

def aggregate(all_runs, dataset):
    """Compute mean ± std across seeds for a given dataset."""
    dfs = [run[dataset] for run in all_runs]
    models = dfs[0]["Model"].tolist()
    rows = []
    for model in models:
        mses   = [df[df["Model"] == model]["MSE"].values[0]   for df in dfs]
        qlikes = [df[df["Model"] == model]["QLIKE"].values[0] for df in dfs]
        rows.append({
            "Model":      model,
            "MSE_mean":   np.mean(mses),
            "MSE_std":    np.std(mses),
            "QLIKE_mean": np.mean(qlikes),
            "QLIKE_std":  np.std(qlikes),
        })
    return pd.DataFrame(rows)


def print_table(df, dataset_name):
    print(f"\n{'='*72}")
    print(f"  {dataset_name}  —  mean ± std over {N_SEEDS} seeds")
    print(f"{'='*72}")
    print(f"{'Model':<20} {'MSE (mean±std)':<28} {'QLIKE (mean±std)':<28}")
    print("-" * 72)
    for _, row in df.iterrows():
        mse_str   = f"{row['MSE_mean']:.6f} ± {row['MSE_std']:.6f}"
        qlike_str = f"{row['QLIKE_mean']:.4f}   ± {row['QLIKE_std']:.4f}"
        marker = " ◀ best" if row["MSE_mean"] == df["MSE_mean"].min() else ""
        print(f"{row['Model']:<20} {mse_str:<28} {qlike_str:<28}{marker}")
    print("-" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

N_SEEDS = 5
SEEDS   = [42, 0, 1, 2, 3]

def main():
    t0 = time.time()
    print(f"Running multi-seed benchmark with seeds={SEEDS}")

    all_runs = []
    for i, seed in enumerate(SEEDS):
        print(f"\n>>> Seed {seed}  ({i+1}/{N_SEEDS}) ...")
        run = run_one_seed(seed)
        all_runs.append(run)

    # Aggregate & display
    os.makedirs("results", exist_ok=True)
    all_summaries = []

    for dataset in ["NARMA10", "Mackey_Glass", "SP500_RV"]:
        summary = aggregate(all_runs, dataset)
        print_table(summary, dataset)
        summary.insert(0, "Dataset", dataset)
        all_summaries.append(summary)

    final = pd.concat(all_summaries, ignore_index=True)
    final.to_csv("results/multi_seed_summary.csv", index=False)
    print(f"\n[✓] Saved to results/multi_seed_summary.csv")
    print(f"[✓] Total runtime: {(time.time()-t0)/60:.1f} minutes")


if __name__ == "__main__":
    main()
