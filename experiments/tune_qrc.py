"""
tune_qrc.py
===========
Equal-compute-budget Optuna search for HPT-QRC hyperparameters, mirroring
`tune_baselines.py` for the classical baselines (LSTM, ESN, RFF). This
removes the unfair asymmetry where the classical baselines are tuned and
HPT-QRC is reported at fixed defaults.

Search space:
  - photon_list \in {[2], [3], [4], [2,3], [3,4], [2,4], [2,3,4]}
  - window \in {5, 10, 15}
  - n_virtual_nodes \in {1, 2, 3, 4}
  - lex_out \in {6, 10, 14}
  - ridge_alpha \in [1e-6, 1e-1] log-spaced

Trial budget: 20 by default (override --trials). HPT-QRC training is slower
than the classical baselines (~10s/fit), so 20 trials is a sane upper bound
on CPU. Best configs cached in results/qrc_best_configs.json.

Usage:
    python tune_qrc.py --trials 20
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import optuna
from sklearn.metrics import mean_squared_error

import pathlib as _pathlib, sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent / 'src'))
del _pathlib, _sys
from data_loader import load_mackey_glass, load_narma10, load_sp500
from multi_qrc import HPT_QRC_Multi

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _dataset(name: str, seed: int = 42):
    if name == "NARMA10":
        X_tr, y_tr, X_te, y_te = load_narma10(seed=seed)
        return y_tr, y_te, X_tr, X_te, False
    if name == "Mackey_Glass":
        X_tr, y_tr, X_te, y_te = load_mackey_glass(seed=seed)
        return y_tr, y_te, X_tr, X_te, False
    if name == "SP500_RV":
        y_tr, y_te, X_tr, X_te = load_sp500()
        return y_tr, y_te, X_tr, X_te, True
    raise ValueError(name)


def _split_train_val(y, X=None, val_frac: float = 0.2):
    n_val = max(20, int(len(y) * val_frac))
    y_lo, y_va = y[:-n_val], y[-n_val:]
    if X is not None:
        X_lo, X_va = X[:-n_val], X[-n_val:]
    else:
        X_lo, X_va = None, None
    return y_lo, y_va, X_lo, X_va


PHOTON_CHOICES = [
    "[2]", "[3]", "[4]", "[2,3]", "[3,4]", "[2,4]", "[2,3,4]",
]


def _parse(spec: str) -> list[int]:
    return [int(x) for x in spec.strip("[]").split(",")]


def tune_qrc(dataset: str, n_trials: int, seed: int = 42, exog: bool = False) -> dict:
    y_tr_full, _, X_tr_full, _, log_space = _dataset(dataset, seed=seed)
    use_har = (dataset == "SP500_RV")
    y_tr, y_va, X_tr, X_va = _split_train_val(y_tr_full, X_tr_full if exog else None)

    def objective(trial: optuna.Trial):
        photon_spec = trial.suggest_categorical("photon_list", PHOTON_CHOICES)
        photon_list = _parse(photon_spec)
        window = trial.suggest_categorical("window", [5, 10, 15])
        n_vd = trial.suggest_categorical("n_virtual_nodes", [1, 2, 3])
        lex = trial.suggest_categorical("lex_out", [6, 10, 14])
        ridge = trial.suggest_float("ridge_alpha", 1e-6, 1e-1, log=True)
        try:
            in_dim = 1 + (X_tr.shape[1] if (exog and X_tr is not None) else 0)
            m = HPT_QRC_Multi(
                in_size=in_dim,
                window=int(window),
                photon_list=photon_list,
                n_virtual_nodes=int(n_vd),
                lex_out=int(lex),
                ridge_alpha=float(ridge),
                use_har_context=use_har,
                seed=seed,
            )
            if exog and X_tr is not None:
                m.fit(y_tr, X_tr)
                pred = m.predict(y_va, X_va)
            else:
                m.fit(y_tr)
                pred = m.predict(y_va)
            return float(mean_squared_error(y_va, pred))
        except Exception:
            return float("inf")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {
        "best_params": study.best_params,
        "best_val_mse": study.best_value,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument(
        "--datasets",
        nargs="+",
        choices=["NARMA10", "Mackey_Glass", "SP500_RV"],
        default=["NARMA10", "Mackey_Glass", "SP500_RV"],
    )
    ap.add_argument("--exog", action="store_true",
                    help="Tune the X variant (uses exogenous regressors)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    results = {}
    for dataset in args.datasets:
        print(f"[tune-qrc] {dataset} ({args.trials} trials, exog={args.exog})")
        res = tune_qrc(dataset, args.trials, seed=42, exog=args.exog)
        print(f"   best: val_MSE={res['best_val_mse']:.6e}  params={res['best_params']}")
        results[dataset] = res

    fname = "qrc_best_configs_x.json" if args.exog else "qrc_best_configs.json"
    with open(f"{args.out}/{fname}", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[tune-qrc] saved {args.out}/{fname}")


if __name__ == "__main__":
    main()
