"""Does the optimal encoding window track the task's own memory order?

NARMA-N's target contains the cross-lag product ``u_t * u_{t-N+1}``. A linear-optical map
can only form that product if both lags are present in the *same* encoding, so the prediction
is that the best ``encode_window`` tracks N rather than being a free-floating hyperparameter.

This sweeps ``encode_window`` on NARMA-5, NARMA-10 and NARMA-20 with everything else fixed,
and also ablates feedback at each setting so the two mechanisms -- history inside the
encoding versus history in the state -- can be separated.

Usage::

    python experiments/ablation_encoding.py --seeds 3
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rc_protocol import evaluate_features  # noqa: E402
from src.tasks import load_task  # noqa: E402
from src.temporal_qrc import TemporalPhotonicQRC  # noqa: E402

warnings.filterwarnings("ignore")
RESULTS = Path(__file__).resolve().parents[1] / "results" / "ablation"

WINDOWS = [1, 2, 3, 5, 8, 10, 12, 15, 20, 25]
TASK_ORDER = {"narma5": 5, "narma10": 10, "narma20": 20}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--datasets", nargs="+", default=list(TASK_ORDER))
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []

    for dataset in args.datasets:
        for seed in range(42, 42 + args.seeds):
            u, y, split = load_task(dataset, seed=seed)
            for encode_window in WINDOWS:
                for feedback in (True, False):
                    model = TemporalPhotonicQRC(
                        n_modes=12, photon_list=(2,), reservoirs_per_photon=2, depth=1,
                        leak=0.1, g_in=0.1, g_fb=0.3, encode_window=encode_window,
                        window=20, feedback=feedback, washout=split.washout, seed=seed,
                    )
                    result = evaluate_features(model.build_features(u, split.n_train), y, split)
                    rows.append(dict(dataset=dataset, order=TASK_ORDER[dataset], seed=seed,
                                     encode_window=encode_window, feedback=feedback,
                                     nrmse=result["nrmse"]))
            print(f"  {dataset} seed {seed} done", flush=True)

    frame = pd.DataFrame(rows)
    out = RESULTS / "encoding_window.csv"
    frame.to_csv(out, index=False)

    print("\n=== NRMSE vs encode_window (feedback on, mean over seeds) ===")
    pivot = (
        frame[frame.feedback]
        .pivot_table(index="encode_window", columns="dataset", values="nrmse")
    )
    print(pivot.to_string(float_format=lambda v: f"{v:.4f}"))

    print("\n=== best encode_window per task ===")
    for dataset in args.datasets:
        sub = frame[(frame.dataset == dataset) & frame.feedback]
        means = sub.groupby("encode_window")["nrmse"].mean()
        print(f"  {dataset:<9} task order {TASK_ORDER[dataset]:>2}  ->  "
              f"best encode_window {means.idxmin():>2}  (NRMSE {means.min():.4f})")

    print("\n=== feedback contribution at each task's best window ===")
    for dataset in args.datasets:
        sub = frame[frame.dataset == dataset]
        best = sub[sub.feedback].groupby("encode_window")["nrmse"].mean().idxmin()
        on = sub[(sub.encode_window == best) & sub.feedback]["nrmse"].mean()
        off = sub[(sub.encode_window == best) & ~sub.feedback]["nrmse"].mean()
        print(f"  {dataset:<9} w={best:<3} feedback on {on:.4f}  off {off:.4f}  "
              f"ratio {off / on:.2f}x")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
