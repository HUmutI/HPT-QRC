"""Headline benchmark: every model, every task, multi-seed, with significance testing.

Each model is instantiated from the configuration its own Optuna study selected
(``results/tuning/<dataset>_<model>.json``), so no model is tuned against untuned
opponents. All of them then pass through the identical readout, penalty selection and
metric in ``src.rc_protocol``.

The comparison that matters is not the ranking but whether differences survive inference.
Two things are reported for that:

* Diebold-Mariano with Newey-West HAC standard errors, pairwise against the photonic model.
* Hansen's Model Confidence Set at alpha = 0.10. When several models share the MCS, the
  honest statement is that they are indistinguishable -- not that the best mean wins.

``classical_control`` (ridge on a window of the raw drive) is in every table by
construction: it is the model the predecessor architecture never actually beat, so it is
the reference any claim about quantum features has to clear.

Usage::

    python experiments/run_benchmarks.py --datasets narma5 narma10 narma20 --seeds 5
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines_rc import (  # noqa: E402
    esn_features,
    lag_features,
    polynomial_features,
    rff_features,
)
from src.dm_mcs import dm_hac, mcs_from_predictions, mcs_to_dataframe  # noqa: E402
from src.rc_protocol import evaluate_features  # noqa: E402
from src.tasks import load_task  # noqa: E402
from src.temporal_qrc import TemporalPhotonicQRC  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
TUNING = ROOT / "results" / "tuning"
RESULTS = ROOT / "results" / "benchmarks"

# Tasks backed by a fixed empirical series that cannot be regenerated per seed; for these,
# seed variation comes from the model's own random draws only.
FIXED_DATA = {"sp500_rv", "vix"}

MODELS = ["photonic", "photonic_no_feedback", "esn", "rff", "poly", "classical_control"]


def tuned_params(dataset: str, model: str) -> dict | None:
    path = TUNING / f"{dataset}_{model}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())["params"]


def build(model: str, dataset: str, u, split, seed: int):
    """Feature matrix for one model, using its tuned configuration when available."""
    params = tuned_params(dataset, model) or {}

    if model in ("photonic", "photonic_no_feedback"):
        cfg = dict(
            n_modes=12, photon_list=(2,), reservoirs_per_photon=2, depth=1,
            leak=0.1, g_in=0.1, g_fb=0.3, encode_window=10, window=20,
        )
        cfg.update(tuned_params(dataset, "photonic") or {})
        cfg["photon_list"] = tuple(cfg["photon_list"])
        if model == "photonic_no_feedback":
            # Ablation: identical feature map with the recurrence switched off, which is
            # exactly the predecessor architecture.
            cfg["feedback"] = False
        return TemporalPhotonicQRC(washout=split.washout, seed=seed, **cfg).build_features(
            u, split.n_train
        )

    if model == "esn":
        return esn_features(u, split.n_train, seed=seed, **(params or dict(res_size=500)))
    if model == "rff":
        return rff_features(u, split.n_train, seed=seed, **(params or dict(n_features=300)))
    if model == "poly":
        return polynomial_features(u, split.n_train, **(params or dict(window=10, degree=2)))
    if model == "classical_control":
        return lag_features(u, split.n_train, **(params or dict(window=40)))
    raise KeyError(model)


def run_dataset(dataset: str, seeds: list[int]) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    preds_by_seed: dict[int, dict] = {}

    for seed in seeds:
        data_seed = None if dataset in FIXED_DATA else seed
        u, y, split = load_task(dataset, seed=data_seed)
        preds_by_seed[seed] = {"__truth__": y[split.test_slice]}

        for model in MODELS:
            try:
                result = evaluate_features(build(model, dataset, u, split, seed), y, split)
            except Exception as exc:  # one model failing must not lose the whole sweep
                print(f"  !! {dataset}/{model}/seed{seed}: {type(exc).__name__}: {exc}", flush=True)
                continue
            preds_by_seed[seed][model] = result["predictions"]
            rows.append(
                dict(dataset=dataset, model=model, seed=seed, nrmse=result["nrmse"],
                     mse=result["mse"], mae=result["mae"], r2_oos=result["r2_oos"],
                     alpha=result["alpha"], feature_dim=result["feature_dim"])
            )
            print(f"  {dataset}/{model}/seed{seed}: NRMSE {result['nrmse']:.4f} "
                  f"(dim {result['feature_dim']})", flush=True)

    frame = pd.DataFrame(rows)

    # Inference is run on the first seed, where every model saw identical data.
    stats: dict = {}
    first = seeds[0]
    available = {k: v for k, v in preds_by_seed[first].items() if k != "__truth__"}
    truth = preds_by_seed[first]["__truth__"]
    if len(available) >= 2:
        if "photonic" in available:
            stats["dm_vs_photonic"] = {
                name: dict(
                    zip(("stat", "p_value"),
                        dm_hac(truth, available[name], available["photonic"], loss="mse"))
                )
                for name in available
                if name != "photonic"
            }
        try:
            mcs = mcs_from_predictions(truth, available, loss="mse", alpha=0.10, B=5000,
                                       block_len=20, seed=42)
            stats["mcs"] = mcs_to_dataframe(mcs).to_dict(orient="records")
        except Exception as exc:
            stats["mcs_error"] = f"{type(exc).__name__}: {exc}"
    return frame, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--datasets", nargs="+",
        default=["narma5", "narma10", "narma20", "mackey_glass_h17", "lorenz63", "sp500_rv"],
    )
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    seeds = list(range(42, 42 + args.seeds))
    all_frames = []

    for dataset in args.datasets:
        print(f"\n=== {dataset} ===", flush=True)
        frame, stats = run_dataset(dataset, seeds)
        if frame.empty:
            continue
        frame.to_csv(RESULTS / f"{dataset}_raw.csv", index=False)
        (RESULTS / f"{dataset}_stats.json").write_text(json.dumps(stats, indent=2, default=str))
        all_frames.append(frame)
        print(
            frame.groupby("model")["nrmse"].agg(["mean", "std", "min"])
            .sort_values("mean").to_string(),
            flush=True,
        )

    if all_frames:
        combined = pd.concat(all_frames)
        combined.to_csv(RESULTS / "all_raw.csv", index=False)
        pivot = combined.pivot_table(index="model", columns="dataset", values="nrmse",
                                     aggfunc="mean")
        pivot.to_csv(RESULTS / "summary_nrmse.csv")
        print("\n=== NRMSE (mean over seeds) ===")
        print(pivot.to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
