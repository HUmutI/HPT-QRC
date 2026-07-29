"""Baseline feature maps, built to the same interface as the photonic reservoir.

Every builder takes the raw drive ``u`` and ``n_train``, scales the input on training rows
only, and returns a feature matrix for the whole series. The readout, the ridge-penalty
selection and the metrics are then identical across models -- the only thing that varies is
the feature map, which is what the comparison is supposed to isolate.

The baselines that matter here:

* :func:`lag_features` -- ridge on a window of the raw drive. This is the control the
  predecessor model never beat, so it appears in every table.
* :func:`esn_features` -- a leaky-integrator echo state network *with input scaling*. The
  repository's original ``EchoStateNetwork`` has no input-scaling knob, which cost it a
  factor of roughly three on NARMA-10 and made it far too easy to beat.
* :func:`rff_features` -- random Fourier features at matched dimension, the honest test of
  whether a random nonlinear feature map explains the photonic result.
* :func:`polynomial_features` -- explicit low-order polynomial expansion of the window.
  NARMA targets are polynomial in the drive, so this bounds how much of the photonic gain
  is just "some nonlinearity in the window".
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

__all__ = [
    "scale_input",
    "lag_features",
    "esn_features",
    "rff_features",
    "polynomial_features",
]


def _as_2d(u) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    return u.reshape(-1, 1) if u.ndim == 1 else u


def scale_input(u, n_train: int) -> np.ndarray:
    """Z-score the drive using training rows only."""
    u = _as_2d(u)
    return StandardScaler().fit(u[:n_train]).transform(u)


def _lags(u: np.ndarray, k: int) -> np.ndarray:
    t, d = u.shape
    pad = np.vstack([np.zeros((k - 1, d)), u])
    return np.stack([pad[i : i + k].ravel() for i in range(t)])


def lag_features(u, n_train: int, window: int = 20) -> np.ndarray:
    """Ridge control: a plain window of the drive."""
    return _lags(scale_input(u, n_train), max(1, int(window)))


def esn_features(
    u,
    n_train: int,
    res_size: int = 500,
    leak: float = 0.3,
    spectral_radius: float = 0.9,
    input_scaling: float = 1.0,
    sparsity: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    """Leaky echo state network states, concatenated with the drive.

    ``x_t = (1 - a) x_{t-1} + a * tanh(W_in [1; u_t] + W x_{t-1})`` with ``W`` rescaled to
    the requested spectral radius.
    """
    us = scale_input(u, n_train)
    rng = np.random.default_rng(seed)
    res_size = int(res_size)

    w_in = rng.uniform(-1.0, 1.0, size=(res_size, us.shape[1] + 1)) * float(input_scaling)
    w = rng.uniform(-1.0, 1.0, size=(res_size, res_size))
    if sparsity > 0:
        w *= rng.random(w.shape) >= sparsity
    radius = np.max(np.abs(np.linalg.eigvals(w)))
    if radius > 0:
        w *= float(spectral_radius) / radius

    state = np.zeros(res_size)
    states = np.empty((len(us), res_size))
    leak = float(leak)
    for t in range(len(us)):
        state = (1 - leak) * state + leak * np.tanh(w_in @ np.r_[1.0, us[t]] + w @ state)
        states[t] = state
    return np.hstack([us, states])


def rff_features(
    u, n_train: int, n_features: int = 300, gamma: float = 1.0, window: int = 20, seed: int = 42
) -> np.ndarray:
    """Random Fourier features (Rahimi & Recht 2007) of a window of the drive."""
    lagged = _lags(scale_input(u, n_train), max(1, int(window)))
    rng = np.random.default_rng(seed)
    n_features = int(n_features)
    w = rng.normal(0.0, np.sqrt(2.0 * float(gamma)), size=(lagged.shape[1], n_features))
    b = rng.uniform(0.0, 2 * np.pi, size=n_features)
    return np.sqrt(2.0 / n_features) * np.cos(lagged @ w + b)


def polynomial_features(u, n_train: int, window: int = 10, degree: int = 2) -> np.ndarray:
    """Window of the drive plus all monomials up to ``degree`` (interactions included)."""
    from sklearn.preprocessing import PolynomialFeatures

    lagged = _lags(scale_input(u, n_train), max(1, int(window)))
    poly = PolynomialFeatures(degree=int(degree), include_bias=False)
    poly.fit(lagged[:n_train])
    return poly.transform(lagged)
