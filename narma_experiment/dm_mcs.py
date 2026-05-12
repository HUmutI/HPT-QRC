"""
dm_mcs.py
=========
Econometric inference utilities for forecast comparison:

1. `dm_hac(y_true, pred_a, pred_b, loss="mse", bandwidth=None)`
   Diebold-Mariano test with Newey-West HAC variance (Bartlett kernel),
   automatic bandwidth m = floor(4 * (T/100)^(2/9)) capped at T-1
   (Andrews 1991 rule of thumb).

2. `generate_dm_hac_table(y_true, preds_dict, loss="mse")`
   Drop-in replacement for `train_narma.generate_dm_table` using HAC variance.

3. `hansen_mcs(loss_matrix, alpha=0.10, B=5000, block_len=20, seed=42)`
   Hansen-Lunde-Nason (Econometrica 2011) Model Confidence Set with
   the stationary bootstrap (Politis & Romano 1994) and iterative
   t-max elimination. Returns the survivor set at confidence (1-alpha),
   plus the elimination order and p-values.

References
----------
- Diebold & Mariano (1995), J. Business & Economic Statistics.
- Newey & West (1987), Econometrica.
- Andrews (1991), Econometrica (automatic bandwidth).
- Politis & Romano (1994), JASA (stationary bootstrap).
- Hansen, Lunde & Nason (2011), Econometrica (MCS).
- Patton (2011), Journal of Econometrics (QLIKE for volatility).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats


# ---------------------------------------------------------------------------
# Per-step loss functions
# ---------------------------------------------------------------------------
def _mse_loss_per_step(y_true, y_pred):
    return (np.asarray(y_true).flatten() - np.asarray(y_pred).flatten()) ** 2


def _mae_loss_per_step(y_true, y_pred):
    return np.abs(np.asarray(y_true).flatten() - np.asarray(y_pred).flatten())


def _qlike_loss_per_step(y_true, y_pred, eps=1e-8):
    y = np.abs(np.asarray(y_true).flatten()) + eps
    p = np.abs(np.asarray(y_pred).flatten()) + eps
    r = y / p
    return r - np.log(r) - 1.0


_LOSS_FNS = {
    "mse": _mse_loss_per_step,
    "mae": _mae_loss_per_step,
    "qlike": _qlike_loss_per_step,
}


def loss_per_step(y_true, y_pred, name="mse"):
    """Return per-timestep loss vector."""
    name = name.lower()
    if name not in _LOSS_FNS:
        raise ValueError(f"Unknown loss '{name}'. Choices: {list(_LOSS_FNS)}")
    return _LOSS_FNS[name](y_true, y_pred)


# ---------------------------------------------------------------------------
# Newey-West HAC variance
# ---------------------------------------------------------------------------
def _newey_west_bandwidth(T: int) -> int:
    """Andrews (1991) rule-of-thumb m = floor(4 * (T/100)^(2/9))."""
    m = int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))
    return max(1, min(m, T - 1))


def newey_west_variance(d: np.ndarray, bandwidth: int | None = None) -> float:
    """Long-run variance of the loss-differential series via Bartlett kernel."""
    d = np.asarray(d, dtype=float).flatten()
    T = d.size
    if T < 2:
        return 0.0
    if bandwidth is None:
        bandwidth = _newey_west_bandwidth(T)
    d_centered = d - d.mean()
    gamma0 = float(d_centered @ d_centered) / T
    s = gamma0
    for k in range(1, bandwidth + 1):
        cov_k = float(d_centered[k:] @ d_centered[:-k]) / T
        w = 1.0 - k / (bandwidth + 1)
        s += 2.0 * w * cov_k
    return max(s, 0.0)


# ---------------------------------------------------------------------------
# Diebold-Mariano with HAC variance
# ---------------------------------------------------------------------------
def dm_hac(
    y_true,
    pred_a,
    pred_b,
    loss: str = "mse",
    bandwidth: int | None = None,
):
    """
    DM test of H0: E[L_t(A)] = E[L_t(B)] vs H1: not equal.

    Returns (stat, p_value).
    Stat > 0 means A has higher mean loss than B (B preferred).
    """
    la = loss_per_step(y_true, pred_a, loss)
    lb = loss_per_step(y_true, pred_b, loss)
    d = la - lb
    T = d.size
    if T < 2:
        return 0.0, 1.0
    v_long_run = newey_west_variance(d, bandwidth=bandwidth)
    if v_long_run <= 0.0:
        return 0.0, 1.0
    stat = d.mean() / np.sqrt(v_long_run / T)
    p = 2.0 * (1.0 - scipy.stats.norm.cdf(abs(stat)))
    return float(stat), float(p)


def generate_dm_hac_table(
    y_true,
    preds_dict: dict,
    loss: str = "mse",
    bandwidth: int | None = None,
) -> pd.DataFrame:
    """
    Lower triangle = DM stat, upper triangle = p-value, diagonal = "".
    Drop-in replacement for `train_narma.generate_dm_table`.
    """
    models = list(preds_dict.keys())
    n = len(models)
    out = pd.DataFrame(index=models, columns=models, dtype=object)
    for i in range(n):
        for j in range(n):
            if i == j:
                out.iloc[i, j] = ""
                continue
            stat, p = dm_hac(
                y_true, preds_dict[models[i]], preds_dict[models[j]],
                loss=loss, bandwidth=bandwidth,
            )
            if i > j:
                out.iloc[i, j] = f"{stat:.3f}"
            else:
                out.iloc[i, j] = f"{p:.3f}"
    return out


# ---------------------------------------------------------------------------
# Stationary bootstrap (Politis & Romano 1994)
# ---------------------------------------------------------------------------
def _stationary_bootstrap_indices(T: int, block_len: int, rng: np.random.Generator):
    """Return a length-T array of indices for one stationary-bootstrap draw."""
    p = 1.0 / block_len  # restart probability per step
    idx = np.empty(T, dtype=np.int64)
    idx[0] = rng.integers(0, T)
    restarts = rng.random(T - 1) < p
    next_jumps = rng.integers(0, T, size=T - 1)
    for t in range(1, T):
        if restarts[t - 1]:
            idx[t] = next_jumps[t - 1]
        else:
            idx[t] = (idx[t - 1] + 1) % T
    return idx


# ---------------------------------------------------------------------------
# Hansen Model Confidence Set
# ---------------------------------------------------------------------------
def hansen_mcs(
    loss_matrix: np.ndarray,
    model_names: list[str] | None = None,
    alpha: float = 0.10,
    B: int = 5000,
    block_len: int = 20,
    seed: int = 42,
) -> dict:
    """
    Hansen-Lunde-Nason (Econometrica 2011) Model Confidence Set.

    Parameters
    ----------
    loss_matrix : (T, M) ndarray. Column m is the per-timestep loss series of model m.
    model_names : optional list of length M.
    alpha       : MCS confidence level (survivors retained at (1-alpha)).
    B           : number of stationary-bootstrap resamples.
    block_len   : expected block length.
    seed        : RNG seed.

    Returns dict with:
        survivors      : list of model names left in the MCS at level (1-alpha).
        elimination    : list of (model_eliminated, p_value) in order of elimination.
        per_model_p    : {model: p_value} of the iteration in which the model
                         would be added to the MCS (max over elim p_values seen so far).
    """
    loss_matrix = np.asarray(loss_matrix, dtype=float)
    T, M = loss_matrix.shape
    if model_names is None:
        model_names = [f"m{i}" for i in range(M)]
    assert len(model_names) == M, "model_names length mismatch"

    rng = np.random.default_rng(seed)

    # Pre-sample bootstrap indices once
    boot_idx = np.stack(
        [_stationary_bootstrap_indices(T, block_len, rng) for _ in range(B)], axis=0
    )  # (B, T)

    active = list(range(M))
    elimination = []
    per_model_p = {name: 1.0 for name in model_names}
    running_max_p = 0.0

    while len(active) > 1:
        sub = loss_matrix[:, active]               # (T, k)
        k = len(active)
        mean_loss = sub.mean(axis=0)               # (k,)
        grand_mean = mean_loss.mean()
        # d_i = mean_loss_i - grand_mean (positive => above average loss => worse)
        d_i = mean_loss - grand_mean

        # Bootstrap variance of d_i
        boot_means = np.empty((B, k))
        for b in range(B):
            boot_means[b] = sub[boot_idx[b]].mean(axis=0)
        d_boot = boot_means - boot_means.mean(axis=1, keepdims=True)
        var_di = d_boot.var(axis=0, ddof=0) + 1e-30  # avoid divide-by-zero
        t_stats = d_i / np.sqrt(var_di)

        # Observed test statistic: T_max = max_i t_i
        T_max_obs = t_stats.max()

        # Bootstrap distribution of T_max under H0 of equal predictive ability
        t_boot = (boot_means - mean_loss[None, :]) / np.sqrt(var_di)[None, :]
        # Re-centre about own bootstrap mean (already subtracted boot grand mean above);
        # to compare with d_i we centre with mean_loss[None,:].
        T_max_boot = t_boot.max(axis=1)
        p_value = float((T_max_boot >= T_max_obs).mean())

        running_max_p = max(running_max_p, p_value)

        if p_value > alpha:
            # All remaining models are in the MCS
            for idx in active:
                per_model_p[model_names[idx]] = running_max_p
            break

        # Eliminate the worst model (largest t-stat)
        worst_local = int(np.argmax(t_stats))
        worst_global = active[worst_local]
        per_model_p[model_names[worst_global]] = running_max_p
        elimination.append((model_names[worst_global], p_value))
        active.pop(worst_local)
    else:
        # Only one model left; it survives by definition
        per_model_p[model_names[active[0]]] = running_max_p

    survivors = [model_names[i] for i in active]
    return {
        "survivors": survivors,
        "elimination": elimination,
        "per_model_p": per_model_p,
        "alpha": alpha,
        "B": B,
        "block_len": block_len,
        "T": T,
        "M": M,
    }


def mcs_from_predictions(
    y_true,
    preds_dict: dict,
    loss: str = "mse",
    alpha: float = 0.10,
    B: int = 5000,
    block_len: int = 20,
    seed: int = 42,
) -> dict:
    """
    Convenience wrapper: build the loss matrix from a dict of predictions
    and run `hansen_mcs`.
    """
    models = list(preds_dict.keys())
    losses = [loss_per_step(y_true, preds_dict[m], loss) for m in models]
    T = min(len(l) for l in losses)
    L = np.stack([l[:T] for l in losses], axis=1)  # (T, M)
    return hansen_mcs(L, model_names=models, alpha=alpha, B=B,
                      block_len=block_len, seed=seed)


def mcs_to_dataframe(res: dict) -> pd.DataFrame:
    """One-row-per-model table with survivor flag and p-value."""
    rows = []
    elim_order = {name: i for i, (name, _) in enumerate(res["elimination"])}
    for m, p in res["per_model_p"].items():
        rows.append({
            "model": m,
            "p_value": p,
            "in_mcs": m in res["survivors"],
            "elimination_order": elim_order.get(m, -1),  # -1 means survivor
        })
    df = pd.DataFrame(rows).sort_values(
        by=["in_mcs", "p_value"], ascending=[False, False]
    ).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T = 500
    y = rng.standard_normal(T)
    # Model A: noise; Model B: noise + small bias; Model C: y itself + tiny noise (best).
    pa = y + rng.standard_normal(T)
    pb = y + rng.standard_normal(T) + 0.5
    pc = y + rng.standard_normal(T) * 0.1
    preds = {"A": pa, "B": pb, "C_truth": pc}

    print("=== DM HAC ===")
    print(generate_dm_hac_table(y, preds, loss="mse"))
    print()
    print("=== Hansen MCS ===")
    res = mcs_from_predictions(y, preds, loss="mse", B=2000, block_len=10)
    print(mcs_to_dataframe(res).to_string(index=False))
    print("\nsurvivors:", res["survivors"])
    print("elimination:", res["elimination"])
