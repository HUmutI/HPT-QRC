"""Optuna search over the recurrent photonic reservoir and its baselines.

Every model is tuned under the identical protocol from ``src.rc_protocol``: the search
objective is validation NRMSE, the test slice is never touched during the search, and the
ridge penalty is selected inside :func:`~src.rc_protocol.fit_readout` for every trial. A
model that is tuned against a baseline that is not is not a comparison, so the baselines
get the same trial budget as the photonic model.

Usage::

    python experiments/tune_temporal.py --dataset narma10 --model photonic --trials 150
    python experiments/tune_temporal.py --dataset narma10 --model all --trials 150
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rc_protocol import Split, fit_readout, nrmse  # noqa: E402
from src.tasks import TASKS, load_task  # noqa: E402
from src.temporal_qrc import TemporalPhotonicQRC  # noqa: E402
from src.baselines_rc import (  # noqa: E402
    esn_features,
    lag_features,
    polynomial_features,
    rff_features,
)

warnings.filterwarnings("ignore")
RESULTS = Path(__file__).resolve().parents[1] / "results" / "tuning"


def _val_score(features: np.ndarray, y: np.ndarray, split: Split) -> float:
    """Validation NRMSE under the shared protocol -- the search objective."""
    _, _, val = fit_readout(features, y, split)
    return val


def _photonic_features(trial, u, split):
    params = {
        "n_modes": trial.suggest_categorical("n_modes", [8, 10, 12, 16, 20, 24]),
        "photon_list": tuple(
            json.loads(trial.suggest_categorical("photon_list", ["[2]", "[3]", "[2,3]", "[2,3,4]"]))
        ),
        "reservoirs_per_photon": trial.suggest_int("reservoirs_per_photon", 1, 8),
        "depth": trial.suggest_int("depth", 1, 3),
        "leak": trial.suggest_float("leak", 0.01, 1.0, log=True),
        "g_in": trial.suggest_float("g_in", 0.01, 3.0, log=True),
        "g_fb": trial.suggest_float("g_fb", 0.01, 3.0, log=True),
        "encode_window": trial.suggest_int("encode_window", 1, 25),
        "window": trial.suggest_int("window", 1, 40),
        "feedback": trial.suggest_categorical("feedback", [True, False]),
    }
    model = TemporalPhotonicQRC(washout=split.washout, seed=42, **params)
    return model.build_features(u, n_train=split.n_train), params


def _esn_features(trial, u, split):
    params = {
        "res_size": trial.suggest_categorical("res_size", [100, 200, 500, 1000]),
        "leak": trial.suggest_float("leak", 0.01, 1.0, log=True),
        "spectral_radius": trial.suggest_float("spectral_radius", 0.1, 1.5),
        "input_scaling": trial.suggest_float("input_scaling", 0.01, 3.0, log=True),
        "sparsity": trial.suggest_float("sparsity", 0.0, 0.95),
    }
    return esn_features(u, n_train=split.n_train, seed=42, **params), params


def _rff_features(trial, u, split):
    params = {
        "n_features": trial.suggest_categorical("n_features", [100, 300, 600, 1200]),
        "gamma": trial.suggest_float("gamma", 1e-3, 1e2, log=True),
        "window": trial.suggest_int("window", 2, 40),
    }
    return rff_features(u, n_train=split.n_train, seed=42, **params), params


def _linear_features(trial, u, split):
    params = {"window": trial.suggest_int("window", 1, 60)}
    return lag_features(u, n_train=split.n_train, **params), params


def _poly_features(trial, u, split):
    params = {
        "window": trial.suggest_int("window", 2, 30),
        "degree": trial.suggest_int("degree", 2, 3),
    }
    return polynomial_features(u, n_train=split.n_train, **params), params


BUILDERS = {
    "photonic": _photonic_features,
    "esn": _esn_features,
    "rff": _rff_features,
    "linear": _linear_features,
    "poly": _poly_features,
}


def tune(dataset: str, model: str, trials: int, seed: int = 0) -> dict:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    u, y, split = load_task(dataset)
    builder = BUILDERS[model]

    def objective(trial):
        try:
            features, _ = builder(trial, u, split)
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise optuna.TrialPruned() from exc
        score = _val_score(features, y, split)
        return score if np.isfinite(score) else float("inf")

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=trials, show_progress_bar=False)

    # Re-evaluate the winner once on test, after the search is over.
    best_trial = study.best_trial
    features, params = builder(optuna.trial.FixedTrial(best_trial.params), u, split)
    pred, alpha, val = fit_readout(features, y, split)
    test = nrmse(y[split.test_slice], pred)

    return {
        "dataset": dataset,
        "model": model,
        "trials": trials,
        "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()},
        "raw_params": best_trial.params,
        "ridge_alpha": alpha,
        "val_nrmse": val,
        "test_nrmse": test,
        "feature_dim": int(features.shape[1]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="narma10", choices=sorted(TASKS))
    ap.add_argument("--model", default="photonic", choices=sorted(BUILDERS) + ["all"])
    ap.add_argument("--trials", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    models = sorted(BUILDERS) if args.model == "all" else [args.model]
    for name in models:
        out = tune(args.dataset, name, args.trials, args.seed)
        path = RESULTS / f"{args.dataset}_{name}.json"
        path.write_text(json.dumps(out, indent=2))
        print(
            f"[{args.dataset}/{name}] val {out['val_nrmse']:.4f}  test {out['test_nrmse']:.4f}  "
            f"dim {out['feature_dim']}  -> {path.name}",
            flush=True,
        )


if __name__ == "__main__":
    main()
