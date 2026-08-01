"""How much does the recurrence actually contribute?

The binary ablation -- tuned model against the same model with ``feedback=False`` -- was the
basis for the claim that recurrence carries the result. That claim needs re-examining, for
two reasons visible in the current tuned configurations:

* The search now chooses ``feedback=False`` outright on most tasks, so on those the
  "ablation" and the model are the same object and the comparison is vacuous.
* Where it chooses ``feedback=True``, it drives ``g_fb`` to the bottom of its search range
  (0.010-0.016 against an upper bound of 3.0). A binary flag cannot distinguish "the
  recurrence matters" from "the search wanted it nearly off but not quite".

Sweeping the feedback gain answers what the flag cannot: whether error varies smoothly with
recurrence strength, and where the optimum sits. Everything else is held at the tuned
configuration, so the only thing moving is how strongly the previous state perturbs the
encoding phases.

Usage::

    python experiments/feedback_strength.py --datasets narma5 narma20 --seeds 5
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rc_protocol import evaluate_features  # noqa: E402
from src.tasks import load_task  # noqa: E402
from src.temporal_qrc import TemporalPhotonicQRC  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "feedback"

# Spans the search range. 0.0 is the ablation; the tuned optima sit near the bottom end, so
# the grid is denser there.
GAINS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.3, 0.6, 1.0, 2.0, 3.0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["narma5", "narma20"])
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in args.datasets:
        tuned = ROOT / "results" / "tuning" / f"{dataset}_photonic.json"
        if not tuned.exists():
            print(f"  !! no tuned configuration for {dataset}", flush=True)
            continue
        base = json.loads(tuned.read_text())["params"]
        base["photon_list"] = tuple(base["photon_list"])
        if "extra_leaks" in base:
            base["extra_leaks"] = tuple(base["extra_leaks"])
        tuned_gain = base.get("g_fb")
        tuned_on = base.get("feedback", True)
        print(f"\n=== {dataset} (tuned: feedback={tuned_on}, g_fb={tuned_gain}) ===",
              flush=True)

        for seed in range(42, 42 + args.seeds):
            data_seed = None if dataset in {"sp500_rv", "vix", "santa_fe"} else seed
            u, y, split = load_task(dataset, seed=data_seed)
            for gain in GAINS:
                cfg = dict(base)
                # gain 0 is the ablation, and is expressed as the flag rather than as a zero
                # gain so it exercises the same code path the ablation table reports.
                cfg["feedback"] = gain > 0
                cfg["g_fb"] = gain if gain > 0 else base.get("g_fb", 0.3)
                try:
                    features = TemporalPhotonicQRC(
                        washout=split.washout, seed=seed, **cfg
                    ).build_features(u, split.n_train)
                    result = evaluate_features(features, y, split)
                except Exception as exc:
                    print(f"  !! {dataset}/g_fb={gain}/seed{seed}: "
                          f"{type(exc).__name__}: {exc}", flush=True)
                    continue
                rows.append(dict(dataset=dataset, g_fb=gain, seed=seed,
                                 nrmse=result["nrmse"]))
            done = [r for r in rows if r["dataset"] == dataset and r["seed"] == seed]
            print(f"  seed {seed}: " + "  ".join(
                f"{r['g_fb']:g}:{r['nrmse']:.4f}" for r in done), flush=True)

    if not rows:
        return
    frame = pd.DataFrame(rows)
    out = RESULTS / "feedback_strength.csv"
    frame.to_csv(out, index=False)

    for dataset, sub in frame.groupby("dataset"):
        med = sub.groupby("g_fb").nrmse.median()
        best = med.idxmin()
        ablation = med.get(0.0, np.nan)
        print(f"\n=== {dataset}: NRMSE vs feedback gain (median over seeds) ===")
        print(med.to_string(float_format=lambda v: f"{v:.4f}"))
        print(f"  best gain {best:g} -> {med[best]:.4f};  ablation (no feedback) "
              f"{ablation:.4f};  recurrence buys {ablation / med[best]:.2f}x")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
