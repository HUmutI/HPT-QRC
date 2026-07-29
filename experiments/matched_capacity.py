"""Is the photonic advantage the optics, or just more features?

The tuned photonic model reaches its best NRMSE with several thousand features, while the
echo state network's search was capped at 1000 units. Comparing those two numbers directly
would confound the feature map with the size of the feature map, which is exactly the
criticism this kind of result attracts.

This sweeps every model family across a common range of feature dimensions and reports
NRMSE as a function of dimension. Three questions come out of it:

* At matched dimension, does the photonic feature map still win?
* Where does each family saturate?
* Does the echo state network keep improving past the 1000 units its search allowed?

Usage::

    python experiments/matched_capacity.py --dataset narma10 --seeds 3
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines_rc import esn_features, lag_features, polynomial_features, rff_features  # noqa: E402
from src.rc_protocol import evaluate_features  # noqa: E402
from src.tasks import load_task  # noqa: E402
from src.temporal_qrc import TemporalPhotonicQRC  # noqa: E402

warnings.filterwarnings("ignore")
RESULTS = Path(__file__).resolve().parents[1] / "results" / "capacity"

# Photonic feature dimension is C(m, n) * reservoirs + window, so it is varied through the
# ensemble width at fixed mode count -- the quantity a hardware run would also scale.
PHOTONIC_ENSEMBLE = [1, 2, 4, 8, 16, 32]
ESN_SIZES = [50, 100, 200, 500, 1000, 2000, 4000]
RFF_SIZES = [50, 100, 300, 600, 1200, 2400, 4800]
POLY_WINDOWS = [(5, 2), (10, 2), (20, 2), (10, 3), (14, 3)]
LAG_WINDOWS = [5, 10, 20, 40, 60, 80]


MAX_FEATURE_DIM = 30_000


def photonic_dim(base: dict, ensemble: int) -> int:
    """Feature dimension a given ensemble width would produce."""
    from math import comb

    per_set = sum(comb(base["n_modes"], n) for n in base["photon_list"])
    return per_set * ensemble + base.get("window", 0)


def photonic(u, split, seed, ensemble, base):
    cfg = dict(base)
    cfg["reservoirs_per_photon"] = ensemble
    cfg["photon_list"] = tuple(cfg["photon_list"])
    return TemporalPhotonicQRC(washout=split.washout, seed=seed, **cfg).build_features(
        u, split.n_train
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="narma10")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    seeds = list(range(42, 42 + args.seeds))

    # The photonic model must enter this sweep with its *tuned* hyperparameters, varying
    # only the ensemble width. Sweeping an untuned base against a well-configured ESN would
    # measure the tuning, not the feature map.
    import json

    tuned = Path(__file__).resolve().parents[1] / "results" / "tuning" / f"{args.dataset}_photonic.json"
    if tuned.exists():
        base = json.loads(tuned.read_text())["params"]
        base.pop("reservoirs_per_photon", None)
        base.pop("feedback", None)
    else:
        base = dict(n_modes=12, photon_list=(2,), depth=1, leak=0.1, g_in=0.1, g_fb=0.3,
                    encode_window=10, window=20)

    def _tuned(model: str, fallback: dict, drop: str) -> dict:
        """Tuned hyperparameters for a baseline, with the size knob removed so it can vary."""
        path = tuned.with_name(f"{args.dataset}_{model}.json")
        params = json.loads(path.read_text())["params"] if path.exists() else dict(fallback)
        params.pop(drop, None)
        return params

    esn_base = _tuned("esn", dict(leak=1.0, spectral_radius=0.9, input_scaling=0.1), "res_size")
    rff_base = _tuned("rff", dict(gamma=0.1, window=20), "n_features")

    rows = []
    for seed in seeds:
        data_seed = None if args.dataset in {"sp500_rv", "vix"} else seed
        u, y, split = load_task(args.dataset, seed=data_seed)

        # Some tuned configurations are already very wide -- the Lorenz-63 winner uses 24
        # modes with photons {2,3,4}, i.e. 12926 features per reservoir set, so an ensemble
        # of 32 would ask for over 400k features. Cap the sweep by dimension rather than by
        # ensemble width, and report what was dropped instead of silently truncating.
        widths = [k for k in PHOTONIC_ENSEMBLE if photonic_dim(base, k) <= MAX_FEATURE_DIM]
        if not widths:
            widths = [1]
        dropped = [k for k in PHOTONIC_ENSEMBLE if k not in widths]
        if dropped and seed == seeds[0]:
            print(f"  (skipping photonic ensembles {dropped}: would exceed "
                  f"{MAX_FEATURE_DIM} features)", flush=True)

        jobs = (
            [("photonic", k, lambda k=k: photonic(u, split, seed, k, base))
             for k in widths]
            + [("esn", k, lambda k=k: esn_features(u, split.n_train, res_size=k, seed=seed,
                                                   **esn_base))
               for k in ESN_SIZES]
            + [("rff", k, lambda k=k: rff_features(u, split.n_train, n_features=k, seed=seed,
                                                   **rff_base))
               for k in RFF_SIZES]
            + [("poly", f"w{w}d{d}", lambda w=w, d=d: polynomial_features(u, split.n_train,
                                                                          window=w, degree=d))
               for w, d in POLY_WINDOWS]
            + [("classical_control", k, lambda k=k: lag_features(u, split.n_train, window=k))
               for k in LAG_WINDOWS]
        )

        for family, setting, make in jobs:
            try:
                result = evaluate_features(make(), y, split)
            except Exception as exc:
                print(f"  !! {family}/{setting}/seed{seed}: {type(exc).__name__}: {exc}",
                      flush=True)
                continue
            rows.append(dict(dataset=args.dataset, family=family, setting=str(setting),
                             seed=seed, feature_dim=result["feature_dim"],
                             nrmse=result["nrmse"]))
            print(f"  {family:<18} {str(setting):<6} dim {result['feature_dim']:>5}  "
                  f"NRMSE {result['nrmse']:.4f}", flush=True)

    frame = pd.DataFrame(rows)
    out = RESULTS / f"capacity_{args.dataset}.csv"
    frame.to_csv(out, index=False)

    summary = (
        frame.groupby(["family", "setting"])
        .agg(dim=("feature_dim", "mean"), nrmse=("nrmse", "mean"), std=("nrmse", "std"))
        .sort_values("dim")
    )
    print(f"\n=== {args.dataset}: NRMSE vs feature dimension ===")
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
