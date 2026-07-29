"""Linear memory capacity and information processing capacity.

An accuracy table says which model wins; these two diagnostics say what it is winning with.

* **Linear memory capacity** (Jaeger 2001): how many past inputs the state linearly encodes.
  The predecessor model scored exactly 5.0 for a window of 5 -- the signature of a windowed
  feature map with no state at all. A genuine reservoir should exceed its encoding window.
* **Information processing capacity** (Dambre et al. 2012): the same idea extended to a
  complete orthogonal basis of nonlinear functions of the input history. Total capacity is
  bounded by the number of linearly independent features, so reporting capacity *per
  feature* is what makes families of different size comparable.

Both are measured on i.i.d. uniform input, the standard driving signal for these
diagnostics, using Legendre polynomials (orthogonal on the input distribution) so that
capacities of different degrees add up rather than double-counting.

Usage::

    python experiments/memory_ipc.py --n 3000 --max-delay 30
"""

from __future__ import annotations

import argparse
import sys
import warnings
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial import legendre
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines_rc import esn_features, lag_features  # noqa: E402
from src.temporal_qrc import TemporalPhotonicQRC  # noqa: E402

warnings.filterwarnings("ignore")
RESULTS = Path(__file__).resolve().parents[1] / "results" / "capacity"


def _r2(features: np.ndarray, target: np.ndarray, washout: int, ridge_alpha: float = 1e-6) -> float:
    """Coefficient of determination of a ridge fit from features to target.

    Capacities are defined by the *fit*, not by generalisation, so this is in-sample by
    construction. A small ridge keeps the solve stable when features outnumber samples;
    without it an over-complete feature set would report a spurious capacity of 1.
    """
    x = features[washout:]
    y = target[washout:]
    y = y - y.mean()
    var = np.mean(y**2)
    if var <= 0:
        return 0.0
    model = Ridge(alpha=ridge_alpha, fit_intercept=True).fit(x, y)
    residual = np.mean((y - model.predict(x)) ** 2)
    return float(max(0.0, 1.0 - residual / var))


def _legendre_products(u: np.ndarray, spec, washout: int) -> np.ndarray:
    """Target for one capacity term: a product of Legendre polynomials of lagged inputs.

    ``spec`` is a list of ``(delay, degree)`` pairs. Input is mapped to [-1, 1], where the
    Legendre family is orthogonal under the uniform measure.
    """
    scaled = 2.0 * u.ravel() - 1.0
    out = np.ones(len(scaled))
    for delay, degree in spec:
        coeffs = np.zeros(degree + 1)
        coeffs[degree] = 1.0
        shifted = np.roll(scaled, delay)
        shifted[:delay] = 0.0
        out = out * legendre.legval(shifted, coeffs)
    return out


def linear_memory_capacity(features, u, washout: int, max_delay: int) -> tuple[float, np.ndarray]:
    per_delay = np.array(
        [_r2(features, _legendre_products(u, [(d, 1)], washout), washout)
         for d in range(1, max_delay + 1)]
    )
    return float(per_delay.sum()), per_delay


def information_processing_capacity(
    features, u, washout: int, max_delay: int = 12, max_degree: int = 3, threshold: float = 1e-3
) -> dict:
    """Total capacity split by polynomial degree.

    Terms below ``threshold`` are discarded: with finite data every term picks up a small
    positive R^2 by chance, and summing thousands of them would manufacture capacity out of
    noise.
    """
    by_degree = {d: 0.0 for d in range(1, max_degree + 1)}
    for degree in range(1, max_degree + 1):
        # Distribute `degree` total polynomial order over distinct delays.
        for delays in combinations_with_replacement(range(1, max_delay + 1), degree):
            spec, counts = [], {}
            for d in delays:
                counts[d] = counts.get(d, 0) + 1
            spec = [(d, k) for d, k in counts.items()]
            score = _r2(features, _legendre_products(u, spec, washout), washout)
            if score > threshold:
                by_degree[degree] += score
    total = sum(by_degree.values())
    return {"total": total, **{f"degree_{d}": v for d, v in by_degree.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--washout", type=int, default=200)
    ap.add_argument("--max-delay", type=int, default=30)
    ap.add_argument("--ipc-delay", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []

    for seed in range(42, 42 + args.seeds):
        rng = np.random.default_rng(seed)
        u = rng.uniform(0.0, 1.0, size=(args.n, 1))
        n_train = args.n  # capacities are in-sample; the whole series is used

        systems = {
            "photonic": lambda: TemporalPhotonicQRC(
                n_modes=12, photon_list=(2,), reservoirs_per_photon=2, depth=1,
                leak=0.1, g_in=0.1, g_fb=0.3, encode_window=10, window=0,
                use_classical=False, washout=args.washout, seed=seed,
            ).build_features(u, n_train),
            "photonic_no_feedback": lambda: TemporalPhotonicQRC(
                n_modes=12, photon_list=(2,), reservoirs_per_photon=2, depth=1,
                leak=0.1, g_in=0.1, g_fb=0.3, encode_window=10, window=0,
                use_classical=False, feedback=False, washout=args.washout, seed=seed,
            ).build_features(u, n_train),
            "esn_200": lambda: esn_features(u, n_train, res_size=200, leak=1.0,
                                            spectral_radius=0.9, input_scaling=0.1, seed=seed),
            "esn_500": lambda: esn_features(u, n_train, res_size=500, leak=1.0,
                                            spectral_radius=0.9, input_scaling=0.1, seed=seed),
            "linear_window_20": lambda: lag_features(u, n_train, window=20),
        }

        for name, make in systems.items():
            features = make()
            mc, curve = linear_memory_capacity(features, u, args.washout, args.max_delay)
            ipc = information_processing_capacity(
                features, u, args.washout, max_delay=args.ipc_delay
            )
            dim = features.shape[1]
            rows.append(dict(system=name, seed=seed, feature_dim=dim, linear_mc=mc,
                             ipc_total=ipc["total"], ipc_per_feature=ipc["total"] / dim,
                             **{k: v for k, v in ipc.items() if k.startswith("degree_")}))
            print(f"  {name:<22} dim {dim:>5}  MC {mc:6.2f}  IPC {ipc['total']:7.2f}  "
                  f"(deg1 {ipc['degree_1']:.2f} deg2 {ipc['degree_2']:.2f} "
                  f"deg3 {ipc['degree_3']:.2f})", flush=True)

    frame = pd.DataFrame(rows)
    out = RESULTS / "memory_ipc.csv"
    frame.to_csv(out, index=False)
    summary = frame.groupby("system").mean(numeric_only=True).sort_values("ipc_total",
                                                                          ascending=False)
    print("\n=== memory and information processing capacity (mean over seeds) ===")
    print(summary.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
