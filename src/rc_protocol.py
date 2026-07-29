"""Shared evaluation protocol for input-driven reservoir benchmarks.

Every model in the comparison gets the same treatment:

* one continuous series, split into washout / train / validation / test;
* the ridge penalty chosen on the validation slice, never on test;
* the reservoir state carried across split boundaries (it has no trained parameters, so
  driving it over later data is not leakage -- only the readout sees targets);
* NRMSE as the headline metric, matching the reservoir-computing literature.

Selecting the ridge penalty per model matters more than it looks: a fixed ``alpha`` favours
whichever model happens to have the feature count that penalty suits, and the predecessor
code used a single hard-coded ``1e-4`` for models whose feature dimension differed by an
order of magnitude.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

__all__ = ["ALPHA_GRID", "nrmse", "mse", "Split", "fit_readout", "evaluate_features"]

# Spans under-regularised to heavily-regularised; reservoir readouts routinely need the
# high end when the feature count approaches the number of training rows.
ALPHA_GRID = np.logspace(-9, 5, 29)


def nrmse(y_true, y_pred) -> float:
    """Root mean squared error normalised by the standard deviation of the target."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)) / np.std(y_true))


def mse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return float(np.mean((y_true - y_pred) ** 2))


class Split:
    """Index boundaries for one continuous series."""

    def __init__(self, n_total: int, washout: int = 100, n_train: int = 700, n_val: int = 100):
        if washout + n_val >= n_train:
            raise ValueError("need washout + n_val < n_train")
        if n_train >= n_total:
            raise ValueError("need n_train < n_total")
        self.n_total = int(n_total)
        self.washout = int(washout)
        self.n_train = int(n_train)
        self.n_val = int(n_val)

    @property
    def fit_slice(self) -> slice:
        """Rows used to fit the readout when selecting alpha."""
        return slice(self.washout, self.n_train - self.n_val)

    @property
    def val_slice(self) -> slice:
        return slice(self.n_train - self.n_val, self.n_train)

    @property
    def full_train_slice(self) -> slice:
        """Rows used to refit the readout once alpha is chosen."""
        return slice(self.washout, self.n_train)

    @property
    def test_slice(self) -> slice:
        return slice(self.n_train, self.n_total)


class _RidgePath:
    """Ridge solutions for many penalties from a single decomposition.

    Refitting ``Ridge`` per penalty recomputes the same decomposition every time. With
    feature counts in the thousands -- routine for a photonic ensemble -- that dominates the
    entire hyperparameter search. One economy SVD of the centred training block gives every
    penalty on the grid for the price of one fit:

        w(alpha) = V diag(s / (s^2 + alpha)) U^T (y - ybar)

    Results are identical to ``Ridge(alpha=...)`` with ``fit_intercept=True`` up to
    floating-point error, which ``tests/test_protocol_and_model.py`` checks.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x_mean = x.mean(axis=0)
        self.y_mean = float(y.mean())
        centred = x - self.x_mean
        self.u, self.s, self.vt = np.linalg.svd(centred, full_matrices=False)
        self.uty = self.u.T @ (y - self.y_mean)

    def coefficients(self, alpha: float) -> np.ndarray:
        scale = self.s / (self.s**2 + alpha)
        return self.vt.T @ (scale * self.uty)

    def predict(self, x: np.ndarray, alpha: float) -> np.ndarray:
        return (x - self.x_mean) @ self.coefficients(alpha) + self.y_mean


def fit_readout(features: np.ndarray, y: np.ndarray, split: Split, standardize: bool = True):
    """Select the ridge penalty on validation, refit on all training rows, predict test.

    Returns ``(test_predictions, chosen_alpha, validation_nrmse)``.
    """
    features = np.asarray(features, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    if standardize:
        scaler = StandardScaler().fit(features[split.fit_slice])
        f_sel = scaler.transform(features)
    else:
        f_sel = features

    path = _RidgePath(f_sel[split.fit_slice], y[split.fit_slice])
    val_x, val_y = f_sel[split.val_slice], y[split.val_slice]
    best_alpha, best_score = float(ALPHA_GRID[0]), np.inf
    for alpha in ALPHA_GRID:
        score = nrmse(val_y, path.predict(val_x, alpha))
        if np.isfinite(score) and score < best_score:
            best_alpha, best_score = float(alpha), score

    # Refit on washout..n_train with the selected penalty, rescaling on the same rows.
    if standardize:
        scaler = StandardScaler().fit(features[split.full_train_slice])
        f_fin = scaler.transform(features)
    else:
        f_fin = features
    final = Ridge(alpha=best_alpha).fit(f_fin[split.full_train_slice], y[split.full_train_slice])
    return final.predict(f_fin[split.test_slice]), best_alpha, float(best_score)


def evaluate_features(features: np.ndarray, y: np.ndarray, split: Split) -> dict:
    """Run the protocol and report test metrics plus the selected penalty."""
    pred, alpha, val = fit_readout(features, y, split)
    truth = np.asarray(y, dtype=float).ravel()[split.test_slice]
    return {
        "nrmse": nrmse(truth, pred),
        "mse": mse(truth, pred),
        "alpha": alpha,
        "val_nrmse": val,
        "feature_dim": int(features.shape[1]),
        "predictions": pred,
    }
