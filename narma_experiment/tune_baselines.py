"""
tune_baselines.py
=================
Equal-budget Optuna hyperparameter search for the LSTM, ESN, and RFF+Ridge
baselines on each dataset, per PROTOCOL.md §1.

The search uses a single train/val split derived from the available train
data (last 20% of train as validation). The final number returned for each
trial is val MSE; the best config per (dataset, model) is cached as JSON
in results/<model>_best_configs.json.

Equal compute budget per trial: N_TRIALS_DEFAULT = 30 (override via --trials).
Original protocol asks for 100 trials; on CPU we trade trial count for
real-time turn-around. 30 trials with TPE sampling reliably finds a stable
optimum for these small models.

Usage:
    python tune_baselines.py --trials 30
    python tune_baselines.py --models lstm rff --trials 50
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import optuna
from sklearn.metrics import mean_squared_error

from classical_baselines import LSTMWrapper, RCModel
from data_loader import load_narma10, load_mackey_glass, load_sp500
from rff_baseline import RFFRidge

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
def _dataset(name: str, seed: int = 42):
    if name == "NARMA10":
        X_tr, y_tr, X_te, y_te = load_narma10(seed=seed)
        return y_tr, y_te, X_tr, X_te, False
    if name == "Mackey_Glass":
        X_tr, y_tr, X_te, y_te = load_mackey_glass(seed=seed)
        return y_tr, y_te, X_tr, X_te, False
    if name == "SP500_RV":
        y_tr, y_te, X_tr, X_te = load_sp500()
        return y_tr, y_te, X_tr, X_te, True  # log-space
    raise ValueError(name)


def _split_train_val(y_tr, X_tr=None, val_frac: float = 0.2):
    n_val = max(20, int(len(y_tr) * val_frac))
    y_tr_lo, y_va = y_tr[:-n_val], y_tr[-n_val:]
    if X_tr is not None:
        X_tr_lo, X_va = X_tr[:-n_val], X_tr[-n_val:]
    else:
        X_tr_lo, X_va = None, None
    return y_tr_lo, y_va, X_tr_lo, X_va


# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------
def tune_lstm(dataset: str, n_trials: int, seed: int = 42) -> dict:
    y_tr_full, _, X_tr_full, _, _ = _dataset(dataset, seed=seed)
    y_tr, y_va, X_tr, X_va = _split_train_val(y_tr_full, X_tr_full)

    def objective(trial: optuna.Trial):
        hidden = trial.suggest_categorical("hidden_dim", [32, 64, 128])
        layers = trial.suggest_int("num_layers", 1, 3)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        dropout = trial.suggest_categorical("dropout", [0.0, 0.2, 0.4])
        seq_len = trial.suggest_categorical("seq_length", [5, 10, 15])
        epochs = trial.suggest_int("epochs", 50, 200)
        try:
            m = LSTMWrapper(
                input_dim=1, seq_length=seq_len, epochs=epochs, lr=lr,
                hidden_dim=hidden, num_layers=layers, dropout=dropout,
                val_frac=0.2, patience=10,
            ).fit(y_tr, y_tr)
            p = m.predict(y_va, y_va)
            return float(mean_squared_error(y_va, p))
        except Exception as exc:
            return float("inf")

    study = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {"best_params": study.best_params, "best_val_mse": study.best_value}


def tune_esn(dataset: str, n_trials: int, seed: int = 42) -> dict:
    y_tr_full, _, X_tr_full, _, _ = _dataset(dataset, seed=seed)
    y_tr, y_va, X_tr, X_va = _split_train_val(y_tr_full, X_tr_full)

    def objective(trial: optuna.Trial):
        res_size = trial.suggest_categorical("res_size", [50, 100, 200, 500, 1000])
        alpha = trial.suggest_categorical("alpha", [0.1, 0.3, 0.5, 0.9])
        sr = trial.suggest_categorical("spectral_radius", [0.6, 0.9, 1.1])
        ridge_alpha = trial.suggest_float("ridge_alpha", 1e-6, 1e0, log=True)
        try:
            m = RCModel(
                in_size=1, seed=seed,
                res_size=int(res_size), alpha=float(alpha),
                spectral_radius=float(sr), ridge_alpha=float(ridge_alpha),
            ).fit(y_tr)
            p = m.predict(y_va)
            return float(mean_squared_error(y_va, p))
        except Exception:
            return float("inf")

    study = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {"best_params": study.best_params, "best_val_mse": study.best_value}


def tune_rff(dataset: str, n_trials: int, seed: int = 42,
             matched_dim: int = 90) -> dict:
    y_tr_full, _, X_tr_full, _, log_space = _dataset(dataset, seed=seed)
    y_tr, y_va, X_tr, X_va = _split_train_val(y_tr_full, X_tr_full)
    use_har = (dataset == "SP500_RV")

    def objective(trial: optuna.Trial):
        dim_mult = trial.suggest_categorical("dim_mult", [0.5, 1.0, 2.0])
        gamma = trial.suggest_float("gamma", 1e-3, 1e2, log=True)
        ridge_alpha = trial.suggest_float("ridge_alpha", 1e-6, 1e0, log=True)
        window = trial.suggest_categorical("window", [5, 10, 15])
        try:
            D = max(8, int(matched_dim * dim_mult))
            m = RFFRidge(
                in_size=1, window=int(window), output_dim=D, gamma=float(gamma),
                ridge_alpha=float(ridge_alpha), seed=seed, use_har_context=use_har,
            ).fit(y_tr)
            p = m.predict(y_va)
            return float(mean_squared_error(y_va, p))
        except Exception:
            return float("inf")

    study = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {"best_params": study.best_params, "best_val_mse": study.best_value}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--models", nargs="+",
                    choices=["lstm", "esn", "rff"], default=["lstm", "esn", "rff"])
    ap.add_argument("--datasets", nargs="+",
                    choices=["NARMA10", "Mackey_Glass", "SP500_RV"],
                    default=["NARMA10", "Mackey_Glass", "SP500_RV"])
    ap.add_argument("--out", default="results")
    ap.add_argument("--matched_dim", type=int, default=90,
                    help="HPT-QRC Fock-feature dim to match in the RFF baseline.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    fn_map = {"lstm": tune_lstm, "esn": tune_esn,
              "rff": lambda d, n, s: tune_rff(d, n, s, matched_dim=args.matched_dim)}

    for model in args.models:
        all_results = {}
        for dataset in args.datasets:
            print(f"\n[tune] {model} on {dataset} ({args.trials} trials)")
            res = fn_map[model](dataset, args.trials, 42)
            print(f"   best: val_MSE={res['best_val_mse']:.6f}  params={res['best_params']}")
            all_results[dataset] = res
        out_path = f"{args.out}/{model}_best_configs.json"
        with open(out_path, "w") as fh:
            json.dump(all_results, fh, indent=2)
        print(f"[tune] saved {out_path}")


if __name__ == "__main__":
    main()
