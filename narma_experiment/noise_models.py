"""
noise_models.py
===============
Two noise models for bridging simulated SLOS Fock-state probabilities to
realistic photonic-hardware measurements:

  1. Shot-noise (finite-sample) resampling. Treats the exact SLOS output
     probability vector p \in R^D as a categorical distribution and replaces
     it with the empirical histogram of `n_shots` multinomial draws.
     This is the dominant noise channel on Quandela's cloud platform
     after detection efficiency and dark counts are factored out.

  2. Partial-photon indistinguishability. Linearly interpolates between
     the ideal indistinguishable-photon probability vector (p_ideal) and
     the fully-distinguishable probability vector (p_dist):
        p_V = V * p_ideal + (1 - V) * p_dist
     with V in [0, 1] (V=1 is ideal). This is the standard mean-field
     model for partial distinguishability (Tichy 2014). A full MPS / r-fold
     permanent expansion is left for future work.

API
---
  apply_shot_noise(probs, n_shots, seed=None)
  apply_indistinguishability(p_ideal, p_dist, V)
  noisy_features(qrc, y, n_shots=None, V=None, p_dist=None, seed=None)

The `noisy_features` helper monkey-patches `HPT_QRC_Multi._quantum_features_batch`
on a copy of the model so existing benchmark code (`fit`/`predict`) can be
re-run end-to-end under noise without code surgery elsewhere.

Example
-------
  qrc = HPT_QRC_Multi(...).fit(y_train)
  preds_clean = qrc.predict(y_test)

  qrc_noisy = wrap_with_noise(qrc, n_shots=1000, V=0.9, seed=0)
  preds_noisy = qrc_noisy.predict(y_test)
"""

from __future__ import annotations

import copy

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------
def apply_shot_noise(probs: np.ndarray, n_shots: int,
                     rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Resample each row of `probs` (shape (T, D)) via multinomial(`n_shots`, row).
    Returns empirical frequencies of the same shape. Rows that do not sum to 1
    are renormalised after clipping to non-negative.
    """
    if rng is None:
        rng = np.random.default_rng()
    probs = np.asarray(probs, dtype=float)
    P = np.clip(probs, 0.0, None)
    sums = P.sum(axis=-1, keepdims=True)
    sums = np.where(sums <= 0, 1.0, sums)
    P = P / sums
    out = np.empty_like(P)
    for i in range(P.shape[0]):
        counts = rng.multinomial(n_shots, P[i])
        out[i] = counts / n_shots
    return out


def apply_indistinguishability(p_ideal: np.ndarray,
                                p_dist: np.ndarray,
                                V: float) -> np.ndarray:
    """
    Convex combination of ideal and fully-distinguishable probabilities.

    p_V = V * p_ideal + (1 - V) * p_dist

    V = 1.0 is the ideal limit (single source, perfect mode overlap).
    V = 0.0 is the fully distinguishable limit (no two-photon interference).
    """
    if not 0.0 <= V <= 1.0:
        raise ValueError(f"V must be in [0,1], got {V}")
    p_ideal = np.asarray(p_ideal, dtype=float)
    p_dist  = np.asarray(p_dist,  dtype=float)
    if p_ideal.shape != p_dist.shape:
        raise ValueError("shape mismatch between p_ideal and p_dist")
    return V * p_ideal + (1.0 - V) * p_dist


def distinguishable_baseline(probs: np.ndarray) -> np.ndarray:
    """
    Build a 'fully distinguishable' baseline by mode-permutation averaging
    of the columns of `probs`. The cheap surrogate used here replaces each
    row by its uniform-over-support distribution restricted to the rows
    actually observed in the simulated p_ideal — i.e. the bunching peaks
    are flattened but the support is preserved. For a more rigorous version,
    replace this with the classical mean-field photon-bunching distribution
    computed via permanent-of-modulus-square (see Tichy 2014).
    """
    probs = np.asarray(probs, dtype=float)
    out = np.empty_like(probs)
    for i in range(probs.shape[0]):
        support = probs[i] > 1e-10
        if support.any():
            row = np.zeros_like(probs[i])
            row[support] = 1.0 / support.sum()
            out[i] = row
        else:
            out[i] = probs[i]
    return out


# ---------------------------------------------------------------------------
# HPT_QRC_Multi monkey-patch
# ---------------------------------------------------------------------------
def wrap_with_noise(qrc, n_shots: int | None = None,
                    V: float | None = None,
                    seed: int | None = None):
    """
    Return a deep copy of `qrc` (an `HPT_QRC_Multi`) whose
    `_quantum_features_batch` applies shot-noise and/or indistinguishability.

    Use *after* `qrc.fit(...)` so the wrapped instance retains the trained
    Ridge weights; this lets you study the *inference-time* hardware-noise
    impact independently of fit-time noise.

    If you want to refit under noise (i.e., train the readout to compensate),
    construct a fresh model, wrap with noise, then call `.fit`.
    """
    if n_shots is None and V is None:
        return qrc  # no-op

    new = copy.deepcopy(qrc)
    rng = np.random.default_rng(seed)

    original = new._quantum_features_batch

    def noisy_batch(X_win):
        # Reproduce the per-reservoir path so we have access to each
        # reservoir's raw output BEFORE LexGrouping aggregation. The output
        # of `r(xt)` is a probability-like tensor (Perceval/MerLin layer);
        # we can apply noise on the LexGrouped output as a first-order approx.
        xt = torch.tensor(X_win, dtype=torch.float32)
        feats = []
        with torch.no_grad():
            for r in new.reservoirs:
                out = r(xt)
                if out.is_complex():
                    out = out.real
                out_np = out.numpy()
                # Treat each row as a (non-negative) probability surrogate
                if V is not None:
                    p_dist = distinguishable_baseline(np.clip(out_np, 0, None))
                    out_np = V * out_np + (1.0 - V) * p_dist
                if n_shots is not None:
                    out_np = apply_shot_noise(np.clip(out_np, 0, None),
                                              n_shots=n_shots, rng=rng)
                feats.append(out_np)
        return np.concatenate(feats, axis=1)

    new._quantum_features_batch = noisy_batch
    return new


# ---------------------------------------------------------------------------
# Standalone sweep entry-point
# ---------------------------------------------------------------------------
def shot_sweep(qrc, y_train, y_test, X_train=None, X_test=None,
                shots=(100, 1000, 10000, 100000),
                refit_per_shot: bool = False,
                seed: int = 0) -> "pd.DataFrame":
    """
    Sweep n_shots and report MSE/QLIKE for predict-time noise (or refit-time
    if `refit_per_shot=True`). Returns a DataFrame.
    """
    import pandas as pd
    from sklearn.metrics import mean_squared_error
    rows = []

    def qlike(y, p, eps=1e-8):
        y = np.abs(np.asarray(y).flatten()) + eps
        p = np.abs(np.asarray(p).flatten()) + eps
        r = y / p
        return float(np.mean(r - np.log(r) - 1.0))

    # Baseline (no noise)
    qrc.fit(y_train) if X_train is None else qrc.fit(y_train, X_train)
    pred = qrc.predict(y_test) if X_test is None else qrc.predict(y_test, X_test)
    rows.append({
        "n_shots": "inf", "MSE": mean_squared_error(y_test, pred),
        "QLIKE": qlike(y_test, pred),
    })

    for n in shots:
        if refit_per_shot:
            fresh = copy.deepcopy(qrc)
            noisy = wrap_with_noise(fresh, n_shots=int(n), seed=seed)
            (noisy.fit(y_train) if X_train is None else noisy.fit(y_train, X_train))
        else:
            noisy = wrap_with_noise(qrc, n_shots=int(n), seed=seed)
        pred = noisy.predict(y_test) if X_test is None else noisy.predict(y_test, X_test)
        rows.append({
            "n_shots": int(n), "MSE": mean_squared_error(y_test, pred),
            "QLIKE": qlike(y_test, pred),
        })
    return pd.DataFrame(rows)


def indistinguishability_sweep(qrc, y_train, y_test, X_train=None, X_test=None,
                                Vs=(0.7, 0.8, 0.9, 0.95, 1.0),
                                refit_per_V: bool = False,
                                seed: int = 0) -> "pd.DataFrame":
    """Sweep indistinguishability V and report MSE/QLIKE."""
    import pandas as pd
    from sklearn.metrics import mean_squared_error
    rows = []

    def qlike(y, p, eps=1e-8):
        y = np.abs(np.asarray(y).flatten()) + eps
        p = np.abs(np.asarray(p).flatten()) + eps
        r = y / p
        return float(np.mean(r - np.log(r) - 1.0))

    qrc.fit(y_train) if X_train is None else qrc.fit(y_train, X_train)

    for V in Vs:
        if refit_per_V:
            fresh = copy.deepcopy(qrc)
            noisy = wrap_with_noise(fresh, V=float(V), seed=seed)
            (noisy.fit(y_train) if X_train is None else noisy.fit(y_train, X_train))
        else:
            noisy = wrap_with_noise(qrc, V=float(V), seed=seed)
        pred = noisy.predict(y_test) if X_test is None else noisy.predict(y_test, X_test)
        rows.append({
            "V": float(V), "MSE": mean_squared_error(y_test, pred),
            "QLIKE": qlike(y_test, pred),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import os
    from data_loader import load_narma10, load_sp500
    from multi_qrc import HPT_QRC_Multi

    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["narma", "sp500"], default="narma")
    p.add_argument("--sweep", choices=["shots", "V", "both"], default="both")
    p.add_argument("--out", default="results")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.dataset == "narma":
        X_tr, y_tr, X_te, y_te = load_narma10()
    else:
        y_tr, y_te, X_tr, X_te = load_sp500()

    if args.sweep in ("shots", "both"):
        qrc = HPT_QRC_Multi(in_size=1, window=10, photon_list=[2, 3, 4])
        df = shot_sweep(qrc, y_tr, y_te, seed=args.seed)
        df.to_csv(f"{args.out}/noise_shot_sweep_{args.dataset}.csv", index=False)
        print(df.to_string(index=False))

    if args.sweep in ("V", "both"):
        qrc = HPT_QRC_Multi(in_size=1, window=10, photon_list=[2, 3, 4])
        df = indistinguishability_sweep(qrc, y_tr, y_te, seed=args.seed)
        df.to_csv(f"{args.out}/noise_V_sweep_{args.dataset}.csv", index=False)
        print(df.to_string(index=False))
