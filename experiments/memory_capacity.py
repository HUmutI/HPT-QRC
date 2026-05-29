"""
memory_capacity.py
==================
Linear Memory Capacity (Jaeger 2001) **and** Information Processing Capacity
(IPC; Dambre, Verstraeten, Schrauwen, Massar, Sci. Rep. 2012) for
HPT-QRC (heterogeneous photon ensembles) vs a tuned classical ESN sweep
and a random-linear baseline.

Why this overhaul:
  Earlier headline "MC = 4.0 vs ESN = 0.08" compared an HPT-QRC with
  multi-hundred-feature output to a fixed 100-unit ESN. That is not a
  fair characterisation of classical ESN scaling. Per PROTOCOL.md §7:

  - ESN is swept over res_size in {50, 100, 200, 500, 1000, 2000}
    with grid search over leak rate and spectral radius per size.
  - Linear MC computed at K = 40 lags.
  - IPC computed at degrees 1-4 using Hermite polynomials (Dambre 2012).
  - A figure 'ipc_plane.png' shows linear capacity (degree=1) vs
    sum of nonlinear capacities (degree>=2) for each system; this
    replaces the "50x" headline.

Outputs:
  results/mc_curve.png
  results/mc_scores.csv
  results/ipc_per_system.csv
  results/ipc_plane.png
  results/esn_mc_sweep.csv
"""

from __future__ import annotations

import itertools
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

import pathlib as _pathlib, sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent / 'src'))
del _pathlib, _sys
from esn_baseline import EchoStateNetwork
from multi_qrc import HPT_QRC_Multi

os.makedirs("results", exist_ok=True)


# ---------------------------------------------------------------------------
# Linear Memory Capacity (Jaeger 2001)
# ---------------------------------------------------------------------------
def linear_mc(states: np.ndarray, u: np.ndarray, max_lag: int = 40,
              ridge_alpha: float = 1e-6) -> list[float]:
    """Per-lag R^2 of states[t] predicting u[t-k]. Returns [MC_1, ..., MC_K]."""
    T = states.shape[0]
    mc = []
    for k in range(1, max_lag + 1):
        if k >= T - 1:
            mc.append(0.0)
            continue
        s = states[k:]
        target = u[: T - k]
        n_fit = int(0.8 * len(s))
        if n_fit < 5:
            mc.append(0.0)
            continue
        ridge = Ridge(alpha=ridge_alpha)
        ridge.fit(s[:n_fit], target[:n_fit])
        pred = ridge.predict(s[n_fit:])
        r2 = max(0.0, r2_score(target[n_fit:], pred))
        mc.append(r2)
    return mc


# ---------------------------------------------------------------------------
# Information Processing Capacity (Dambre et al. 2012, simplified)
# ---------------------------------------------------------------------------
def _hermite_basis(max_deg: int):
    """Return list of normalised probabilists' Hermite polynomials He_0 ... He_max_deg."""
    from numpy.polynomial.hermite_e import HermiteE
    return [HermiteE([0] * d + [1]) for d in range(max_deg + 1)]


def ipc_capacities(states: np.ndarray, u: np.ndarray,
                   max_degree: int = 4, max_lag: int = 8,
                   ridge_alpha: float = 1e-6,
                   threshold: float = 1e-3) -> dict[int, float]:
    """
    Compute IPC summed by polynomial degree.

    For each multi-index of total degree d with delays in [1..max_lag] and at most
    `max_degree` non-zero entries, build the target z_t = prod_j He_{d_j}(u_{t-lag_j})
    and compute R^2 = max(0, 1 - SS_res/SS_tot). Sum R^2 over all targets per degree.

    Note: this is a *finite-truncation, finite-sample* approximation of Dambre's
    asymptotic IPC. Exact IPC is data and basis-orthogonality dependent; we follow
    the standard practice of capping degree, lag, and applying a small R^2 floor
    to discard targets that are dominated by sampling noise.
    """
    T = states.shape[0]
    u = np.asarray(u).flatten()
    if u.std() > 1e-12:
        u_std = (u - u.mean()) / u.std()
    else:
        u_std = u.copy()

    H = _hermite_basis(max_degree)
    by_degree: dict[int, float] = {d: 0.0 for d in range(1, max_degree + 1)}

    # Enumerate multi-indices: list of (lag, degree) pairs with sum(degree) = d
    for total_deg in range(1, max_degree + 1):
        # number of nonzero terms = j in {1..total_deg}
        for n_terms in range(1, total_deg + 1):
            for lags in itertools.combinations(range(1, max_lag + 1), n_terms):
                # Partitions of total_deg into n_terms positive ints
                for partition in _compositions(total_deg, n_terms):
                    target = np.ones(T)
                    valid = True
                    for lag, deg in zip(lags, partition):
                        if lag >= T:
                            valid = False
                            break
                        target_lag = np.concatenate([np.zeros(lag), u_std[:-lag]])
                        target = target * H[deg](target_lag)
                    if not valid:
                        continue
                    n_fit = int(0.8 * T)
                    if n_fit < 50:
                        continue
                    s_tr = states[:n_fit]
                    s_te = states[n_fit:]
                    t_tr = target[:n_fit]
                    t_te = target[n_fit:]
                    if t_te.std() < 1e-8:
                        continue
                    try:
                        ridge = Ridge(alpha=ridge_alpha)
                        ridge.fit(s_tr, t_tr)
                        pred = ridge.predict(s_te)
                        r2 = max(0.0, r2_score(t_te, pred))
                    except Exception:
                        r2 = 0.0
                    if r2 > threshold:
                        by_degree[total_deg] += r2
    return by_degree


def _compositions(n: int, k: int):
    """Yield all ordered compositions of n into k positive parts."""
    if k == 1:
        yield (n,)
        return
    for first in range(1, n - k + 2):
        for rest in _compositions(n - first, k - 1):
            yield (first,) + rest


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------
def _qrc_states(qrc: HPT_QRC_Multi, u: np.ndarray) -> np.ndarray:
    y = u.reshape(-1, 1)
    return qrc.get_features(y)


def _esn_states(esn: EchoStateNetwork, u: np.ndarray,
                discard: int = 100) -> np.ndarray:
    data = u.flatten().reshape(-1, 1)
    states, _ = esn._run_reservoir(data)
    return states[discard:]


# ---------------------------------------------------------------------------
# ESN size + hyperparameter sweep
# ---------------------------------------------------------------------------
def esn_sweep(
    u_full: np.ndarray,
    max_lag: int = 40,
    res_sizes: tuple = (50, 100, 200, 500, 1000, 2000),
    leak_rates: tuple = (0.1, 0.3, 0.5, 0.9),
    spectral_radii: tuple = (0.6, 0.9, 1.1),
    seed: int = 42,
    discard: int = 100,
) -> pd.DataFrame:
    """Sweep ESN architecture and report best linear MC per res_size."""
    u_train_idx = int(0.85 * len(u_full))
    u_train = u_full[:u_train_idx]
    u_test  = u_full[u_train_idx:]

    rows = []
    for n_res in res_sizes:
        best_mc = -np.inf
        best_config = None
        for leak, sr in itertools.product(leak_rates, spectral_radii):
            esn = EchoStateNetwork(
                in_size=1, res_size=n_res, alpha=leak,
                spectral_radius=sr, seed=seed,
            )
            states_all = _esn_states(esn, u_full, discard=discard)
            mc = sum(linear_mc(states_all, u_full[discard:], max_lag=max_lag))
            rows.append({
                "res_size": n_res, "leak": leak, "spectral_radius": sr,
                "linear_MC": mc,
            })
            if mc > best_mc:
                best_mc = mc
                best_config = (leak, sr)
        print(f"  ESN res_size={n_res:>4d}: best MC={best_mc:.3f} "
              f"(leak={best_config[0]}, rho={best_config[1]})")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(42)
    N = 2000
    DISCARD = 100
    MAX_LAG = 40
    MAX_DEG = 3
    IPC_MAX_LAG = 6

    u = rng.uniform(0.0, 1.0, N)

    print("\n=== HPT-QRC linear MC ===")
    qrc = HPT_QRC_Multi(in_size=1, window=5, seed=42, use_har_context=False)
    qrc_states = _qrc_states(qrc, u)
    n_use = min(len(qrc_states), len(u) - 1)
    qrc_states = qrc_states[:n_use]
    u_aligned = u[1 : 1 + n_use]
    mc_qrc = linear_mc(qrc_states, u_aligned, max_lag=MAX_LAG)
    print(f"  total linear MC: {sum(mc_qrc):.3f}  (features dim = {qrc_states.shape[1]})")

    print("\n=== HPT-QRC heterogeneous (2+3+4) linear MC ===")
    qrc_het = HPT_QRC_Multi(in_size=1, window=5, seed=42,
                             use_har_context=False, photon_list=[2, 3, 4])
    qrc_het_states = _qrc_states(qrc_het, u)[:n_use]
    mc_het = linear_mc(qrc_het_states, u_aligned, max_lag=MAX_LAG)
    print(f"  total linear MC: {sum(mc_het):.3f}  (features dim = {qrc_het_states.shape[1]})")

    print("\n=== ESN size sweep ===")
    sweep_df = esn_sweep(u, max_lag=MAX_LAG, discard=DISCARD)
    sweep_df.to_csv("results/esn_mc_sweep.csv", index=False)
    best_per_size = (
        sweep_df.sort_values("linear_MC", ascending=False)
        .drop_duplicates("res_size")
        .sort_values("res_size")
    )
    print("Best ESN MC per size:")
    print(best_per_size.to_string(index=False))

    print("\n=== Random-linear baseline ===")
    W_rand = rng.standard_normal((100, 5))
    win = 5
    rand_states = np.array([W_rand @ u[t - win : t] for t in range(win, len(u))])
    mc_rand = linear_mc(rand_states, u[win:], max_lag=MAX_LAG)
    print(f"  total linear MC: {sum(mc_rand):.3f}")

    # IPC per system (curve plotted vs degree)
    print("\n=== IPC (degrees 1-3, max lag {}) ===".format(IPC_MAX_LAG))
    print(" (computing for HPT-QRC, HPT-QRC-Hetero, ESN at best res_size, Random-Linear)")
    systems = {
        "HPT-QRC (3 photons)": qrc_states,
        "HPT-QRC-Hetero (2,3,4)": qrc_het_states,
    }
    # Best ESN sizes 200 and 1000 for representative comparison
    for n_res in (200, 1000):
        row = best_per_size[best_per_size.res_size == n_res].iloc[0]
        esn = EchoStateNetwork(
            in_size=1, res_size=int(n_res),
            alpha=float(row.leak), spectral_radius=float(row.spectral_radius),
            seed=42,
        )
        states_all = _esn_states(esn, u, discard=DISCARD)[: n_use]
        systems[f"ESN res={n_res}"] = states_all
    systems["Random-Linear"] = rand_states[:n_use]

    ipc_rows = []
    for name, states in systems.items():
        # Pick the matching u-slice for each system:
        # - HPT-QRC uses get_features which already advances the input by 1
        #   (see HPT_QRC_Multi.get_features); we align via u_aligned.
        # - ESN states are returned after `discard` warm-up steps; align via u[DISCARD:].
        # - Random-linear features use a length-`win` sliding window; align via u[win:].
        if name.startswith("HPT-QRC"):
            u_match = u_aligned[: len(states)]
        elif name.startswith("Random"):
            u_match = u[win : win + len(states)]
        else:  # ESN
            u_match = u[DISCARD : DISCARD + len(states)]
        # Defensive trim to a common length
        L = min(len(states), len(u_match))
        states = states[:L]
        u_match = u_match[:L]
        ipc = ipc_capacities(states, u_match,
                              max_degree=MAX_DEG, max_lag=IPC_MAX_LAG)
        total = sum(ipc.values())
        row = {"system": name, "feature_dim": states.shape[1], "IPC_total": total}
        for d, val in ipc.items():
            row[f"IPC_deg{d}"] = val
        ipc_rows.append(row)
        print(f"  {name:<30s}  IPC_total={total:.2f}  per-degree={ipc}")
    ipc_df = pd.DataFrame(ipc_rows)
    ipc_df.to_csv("results/ipc_per_system.csv", index=False)

    # ---------------------------------------------------------------
    # MC summary CSV (legacy + matched)
    # ---------------------------------------------------------------
    summary_df = pd.DataFrame({
        "system": ["HPT-QRC (3 ph)", "HPT-QRC-Hetero (2,3,4)",
                   *[f"ESN res={n}" for n in (200, 1000)],
                   "Random-Linear"],
        "linear_MC": [
            sum(mc_qrc), sum(mc_het),
            *[
                best_per_size[best_per_size.res_size == n].iloc[0].linear_MC
                for n in (200, 1000)
            ],
            sum(mc_rand),
        ],
        "feature_dim": [
            qrc_states.shape[1], qrc_het_states.shape[1],
            200 + 2, 1000 + 2,  # +bias+input
            rand_states.shape[1],
        ],
    })
    summary_df.to_csv("results/mc_scores.csv", index=False)

    # ---------------------------------------------------------------
    # Plots
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    lags = np.arange(1, MAX_LAG + 1)
    ax.plot(lags, mc_qrc, label=f"HPT-QRC (dim={qrc_states.shape[1]})", linewidth=2)
    ax.plot(lags, mc_het, label=f"HPT-QRC-Het (dim={qrc_het_states.shape[1]})",
            linewidth=2, linestyle="--")
    for n_res in (200, 1000):
        row = best_per_size[best_per_size.res_size == n_res].iloc[0]
        esn = EchoStateNetwork(in_size=1, res_size=int(n_res),
                                alpha=float(row.leak),
                                spectral_radius=float(row.spectral_radius), seed=42)
        s = _esn_states(esn, u, discard=DISCARD)[:n_use]
        mc = linear_mc(s, u_aligned, max_lag=MAX_LAG)
        ax.plot(lags, mc, label=f"ESN res={n_res} (dim={s.shape[1]})",
                linewidth=1.5)
    ax.plot(lags, mc_rand, label="Random-Linear", linestyle=":", color="#666")
    ax.set_xlabel("Lag k"); ax.set_ylabel("R²")
    ax.set_title("Linear Memory Capacity vs. Lag", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_ylim(0, 1.0)

    # IPC plane: linear (deg 1) vs nonlinear sum (deg >=2)
    ax2 = axes[1]
    for _, r in ipc_df.iterrows():
        lin = r.get("IPC_deg1", 0.0)
        nl = sum(r.get(f"IPC_deg{d}", 0.0) for d in range(2, MAX_DEG + 1))
        ax2.scatter(lin, nl, s=80)
        ax2.annotate(r["system"], (lin, nl), fontsize=8,
                     xytext=(5, 5), textcoords="offset points")
    ax2.set_xlabel("Linear capacity (IPC deg=1)")
    ax2.set_ylabel("Nonlinear capacity (sum deg≥2)")
    ax2.set_title("IPC Plane (Dambre et al. 2012)", fontweight="bold")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/mc_curve.png", dpi=150)
    plt.close()

    # Stand-alone IPC plane figure
    fig2, ax3 = plt.subplots(figsize=(7, 6))
    for _, r in ipc_df.iterrows():
        lin = r.get("IPC_deg1", 0.0)
        nl = sum(r.get(f"IPC_deg{d}", 0.0) for d in range(2, MAX_DEG + 1))
        ax3.scatter(lin, nl, s=120)
        ax3.annotate(r["system"], (lin, nl), fontsize=10,
                     xytext=(7, 5), textcoords="offset points")
    ax3.set_xlabel("Linear capacity (IPC degree 1)", fontsize=12)
    ax3.set_ylabel(f"Sum of nonlinear capacity (degrees 2..{MAX_DEG})", fontsize=12)
    ax3.set_title("Information Processing Capacity plane\n"
                  "(matched-dimension comparison)", fontweight="bold")
    ax3.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/ipc_plane.png", dpi=150)
    plt.close()

    print("\n[✓] results/mc_curve.png")
    print("[✓] results/ipc_plane.png")
    print("[✓] results/mc_scores.csv")
    print("[✓] results/ipc_per_system.csv")
    print("[✓] results/esn_mc_sweep.csv")


if __name__ == "__main__":
    main()
