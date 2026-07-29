"""Echo state property: does the reservoir forget its initial condition?

Calling something a reservoir is a claim that its state is a function of the input history
alone. That holds only if perturbations to the initial state decay. Two states are started
from different initial conditions, driven with identical input, and the distance between
them is tracked.

The feedback gain ``g_fb`` controls this. Too small and there is no memory worth having; too
large and the map stops contracting, the state depends on where it started, and the readout
is fitting something that will not reproduce. This locates the usable band and confirms the
operating point used in the benchmarks sits inside it.

This replaces an earlier script of the same name that tested the previous windowed model.
That model was stateless, so it satisfied the property trivially and the diagnostic said
nothing.

Usage::

    python experiments/esp_check.py --seeds 3
"""

from __future__ import annotations

import argparse
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
RESULTS = Path(__file__).resolve().parents[1] / "results" / "ablation"

GAINS = [0.0, 0.05, 0.1, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 5.0]


def classify(decay: np.ndarray) -> str:
    """Label the regime from the perturbation-decay curve."""
    peak = float(decay.max())
    if peak <= 1e-12:
        return "stateless"
    tail = float(decay[-max(len(decay) // 4, 1):].mean())
    if tail > 0.1 * peak:
        return "ESP-violating"
    if tail > 0.01 * peak:
        return "marginal"
    return "contracting"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="narma10")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []

    for seed in range(42, 42 + args.seeds):
        u, y, split = load_task(args.dataset, seed=seed)
        for gain in GAINS:
            model = TemporalPhotonicQRC(
                n_modes=12, photon_list=(2,), reservoirs_per_photon=2, depth=1,
                leak=0.1, g_in=0.1, g_fb=gain, encode_window=10, window=20,
                washout=split.washout, seed=seed,
            )
            score = evaluate_features(model.build_features(u, split.n_train), y, split)["nrmse"]
            decay = model.esp_decay(u, perturbation=0.1, seed=seed)
            peak = max(float(decay.max()), 1e-30)
            # Steps until the perturbation falls below 1% of its peak: the memory timescale.
            below = np.where(decay < 0.01 * peak)[0]
            rows.append(dict(
                g_fb=gain, seed=seed, nrmse=score, regime=classify(decay), peak=peak,
                tail=float(decay[-max(len(decay) // 4, 1):].mean()),
                memory_steps=int(below[0]) if len(below) else len(decay),
            ))
        print(f"  seed {seed} done", flush=True)

    frame = pd.DataFrame(rows)
    out = RESULTS / "esp_check.csv"
    frame.to_csv(out, index=False)

    summary = frame.groupby("g_fb").agg(
        nrmse=("nrmse", "mean"),
        memory_steps=("memory_steps", "mean"),
        regime=("regime", lambda s: s.mode().iloc[0]),
    )
    print("\n=== echo state property vs feedback gain ===")
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))
    best = summary["nrmse"].idxmin()
    print(f"\nbest NRMSE at g_fb = {best} (regime: {summary.loc[best, 'regime']}); "
          f"benchmarks use g_fb = 0.3")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
