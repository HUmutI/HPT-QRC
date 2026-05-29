"""
walk_forward_runner.py
======================
Drive any forecaster through a `WalkForwardSplit` and emit per-fold + median+IQR results.

Model interface (duck-typed):
  model.fit(y_train, X_exog=None)   # X_exog optional
  model.predict(y_test, X_exog=None) -> 1D / 2D predictions

Outputs:
  results/wf_<dataset>_<metric>_per_fold.csv
  results/wf_<dataset>_summary.csv
  results/wf_<dataset>_MCS_<metric>.csv  (pooled across folds)
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

import pathlib as _pathlib, sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent / 'src'))
del _pathlib, _sys
from walk_forward import WalkForwardSplit, sp500_monthly_split, vix_daily_split
from data_loader import load_sp500_df, load_vix_df
from dm_mcs import loss_per_step, mcs_from_predictions, mcs_to_dataframe


@dataclass
class ModelSpec:
    name: str
    build: Callable[[], object]   # zero-arg constructor returning a fresh model
    uses_exog: bool = False       # whether to pass X to fit/predict
    log_space: bool = False       # back-transform predictions before scoring
    lstm_style: bool = False      # LSTMWrapper.fit(train_data, y_train) signature


def _qlike(y, p, eps=1e-8):
    y = np.abs(np.asarray(y).flatten()) + eps
    p = np.abs(np.asarray(p).flatten()) + eps
    r = y / p
    return float(np.mean(r - np.log(r) - 1.0))


def _evaluate(y_true, y_pred):
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "QLIKE": _qlike(y_true, y_pred),
    }


def run_walk_forward(
    dataset_name: str,
    df: pd.DataFrame,
    splitter: WalkForwardSplit,
    models: list[ModelSpec],
    out_dir: str = "results",
    mcs_alpha: float = 0.10,
    mcs_B: int = 2000,
    mcs_block_len: int = 20,
):
    """Run every model on every fold and pool per-step losses for an MCS test."""
    os.makedirs(out_dir, exist_ok=True)
    y_col = "y"
    x_cols = [c for c in df.columns if c.startswith("X_")]

    per_fold_rows = []
    pooled_preds: dict[str, list[np.ndarray]] = {m.name: [] for m in models}
    pooled_y: list[np.ndarray] = []

    folds = list(splitter.split(df))
    print(f"[walk-forward] {dataset_name}: {len(folds)} folds")

    for fold in folds:
        tr, va, te = fold.slice_df(df)
        y_tr = tr[y_col].values.reshape(-1, 1)
        y_va = va[y_col].values.reshape(-1, 1)
        y_te = te[y_col].values.reshape(-1, 1)
        X_tr = tr[x_cols].values if x_cols else None
        X_va = va[x_cols].values if x_cols else None
        X_te = te[x_cols].values if x_cols else None

        # Combine train + val for the final fit (CV-style); some baselines may
        # tune internally on val; for the simple wired-in versions we just fit
        # on train+val and evaluate on test.
        y_fit = np.vstack([y_tr, y_va])
        X_fit = (
            np.vstack([X_tr, X_va]) if X_tr is not None else None
        )

        for spec in models:
            m = spec.build()
            try:
                if spec.lstm_style:
                    # LSTMWrapper expects (train_data, y_train) with seq_length-windowing.
                    m.fit(y_fit, y_fit)
                    pred = m.predict(y_te, y_te)
                elif spec.uses_exog and X_fit is not None:
                    m.fit(y_fit, X_fit)
                    pred = m.predict(y_te, X_te)
                else:
                    m.fit(y_fit)
                    pred = m.predict(y_te)
            except Exception as exc:
                print(f"  fold {fold.fold_id} {spec.name}: FAILED ({exc})")
                continue

            pred = np.asarray(pred).flatten()
            y_eval = y_te.flatten()
            if spec.log_space:
                y_eval = np.exp(y_eval)
                pred = np.exp(pred)
            scores = _evaluate(y_eval, pred)
            per_fold_rows.append({
                "fold": fold.fold_id,
                "model": spec.name,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                **scores,
            })
            pooled_preds[spec.name].append(pred)
        pooled_y.append(y_te.flatten() if not any(s.log_space for s in models) else np.exp(y_te.flatten()))

    df_fold = pd.DataFrame(per_fold_rows)
    df_fold.to_csv(f"{out_dir}/wf_{dataset_name}_per_fold.csv", index=False)

    summary = (
        df_fold.groupby("model")[["MSE", "MAE", "QLIKE"]]
        .agg(["median", lambda s: s.quantile(0.75) - s.quantile(0.25), "mean", "std"])
    )
    summary.columns = [f"{m}_{stat}" for m, stat in summary.columns]
    summary.to_csv(f"{out_dir}/wf_{dataset_name}_summary.csv")

    # Hansen MCS on pooled per-step losses
    if pooled_y:
        y_pool = np.concatenate(pooled_y)
        for metric in ["mse", "qlike", "mae"]:
            preds_dict = {}
            for name, plist in pooled_preds.items():
                if not plist:
                    continue
                p_pool = np.concatenate(plist)
                if len(p_pool) != len(y_pool):
                    continue
                preds_dict[name] = p_pool
            if len(preds_dict) < 2:
                continue
            res = mcs_from_predictions(
                y_pool, preds_dict, loss=metric, alpha=mcs_alpha,
                B=mcs_B, block_len=mcs_block_len,
            )
            tab = mcs_to_dataframe(res)
            tab.to_csv(f"{out_dir}/wf_{dataset_name}_MCS_{metric.upper()}.csv", index=False)
            with open(f"{out_dir}/wf_{dataset_name}_MCS_{metric.upper()}.json", "w") as fh:
                json.dump({k: v for k, v in res.items() if k != "elimination"} |
                          {"elimination": [list(e) for e in res["elimination"]]}, fh, indent=2)

    return df_fold, summary


# ---------------------------------------------------------------------------
# Default model registry for the walk-forward bench
# ---------------------------------------------------------------------------
def default_models(include_qrc: bool = True, include_rff: bool = True,
                   matched_dim: int = 90) -> list[ModelSpec]:
    """Construct the standard baseline list with leak-safe wrappers."""
    from classical_baselines import ARModel, HARModel, HARXModel, LSTMWrapper, RCModel
    specs: list[ModelSpec] = [
        ModelSpec("AR1", lambda: ARModel(lags=1)),
        ModelSpec("AR3", lambda: ARModel(lags=3)),
        ModelSpec("HAR", lambda: HARModel()),
        ModelSpec("HARX", lambda: HARXModel(), uses_exog=True),
        ModelSpec("ESN", lambda: RCModel(in_size=1)),
        ModelSpec("LSTM", lambda: LSTMWrapper(input_dim=1)),
    ]
    if include_rff:
        from rff_baseline import RFFRidge
        specs.append(ModelSpec(
            "RFF+Ridge",
            lambda: RFFRidge(in_size=1, window=10, output_dim=matched_dim,
                              gamma=0.1, use_har_context=True),
        ))
    # Patch the spec list so the LSTM entry uses lstm_style.
    for spec in specs:
        if spec.name == "LSTM":
            spec.lstm_style = True
    if include_qrc:
        from multi_qrc import HPT_QRC_Multi
        specs.append(ModelSpec(
            "HPT-QRC",
            lambda: HPT_QRC_Multi(in_size=1, window=10, photon_list=[2, 3, 4],
                                   use_har_context=True),
        ))
    return specs


def main_sp500(out_dir: str = "results", max_folds: int = 8):
    df = load_sp500_df()
    splitter = sp500_monthly_split(max_folds=max_folds)
    models = default_models()
    return run_walk_forward("sp500_rv", df, splitter, models, out_dir=out_dir)


def main_vix(out_dir: str = "results", max_folds: int = 6):
    df = load_vix_df()
    splitter = vix_daily_split(max_folds=max_folds)
    models = default_models()
    return run_walk_forward("vix", df, splitter, models, out_dir=out_dir)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["sp500", "vix"], default="sp500")
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--out", default="results")
    args = p.parse_args()
    if args.dataset == "sp500":
        main_sp500(out_dir=args.out, max_folds=args.folds)
    else:
        main_vix(out_dir=args.out, max_folds=args.folds)
