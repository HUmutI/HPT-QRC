"""Hardware-native QRC score on NARMA-10.

Unlike run_hw_subset.py (sim-trained readout applied to HW features), here the
Ridge readout is trained ON hardware features: the physical chip is used as the
reservoir it actually is. This is the standard methodology of hardware-QRC
demonstrations and tolerates sim-vs-HW distribution mismatch by construction.

All train+test timesteps go to the QPU as ONE batched job (Sampler iterations).

Controls reported alongside the hardware score, on identical steps:
  - sim-feature QRC (same architecture, exact simulation)  -> the paper's setting
  - classical-only Ridge (raw window features, no quantum)  -> does the reservoir add anything?

Usage:
    python hardware/run_hw_native.py --local --train 60 --test 20 --shots 500   # free mechanics test
    python hardware/run_hw_native.py --train 300 --test 100 --shots 1000        # Belenos run
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import perceval as pcvl
from src.multi_qrc import HPT_QRC_Multi
from src.data_loader import load_narma10
from phase_export import export_reservoir
from hw_features import sample_batch
from run_hw_subset import build_windows

RESULTS_DIR = Path(__file__).parent / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", default="qpu:belenos")
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--train", type=int, default=300)
    ap.add_argument("--test", type=int, default=100)
    ap.add_argument("--shots", type=int, default=1000)
    ap.add_argument("--discard", type=int, default=50, help="warmup rows dropped from train")
    ap.add_argument("--ridge-alpha", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk", type=int, default=12,
                    help="timesteps per cloud job (Explorer kills jobs at 5 min ~ 14 steps)")
    args = ap.parse_args()
    platform = "local:slos" if args.local else args.platform

    # ---- data: one contiguous NARMA-10 stretch, train | test split ------
    _, y_tr_full, _, y_te_full = load_narma10(seed=args.seed)
    y = np.concatenate([y_tr_full.ravel(), y_te_full.ravel()])
    n_total = args.train + args.test
    y = y[: n_total + 1]                       # +1 for the next-step targets
    series = y[:-1].reshape(-1, 1)             # model inputs
    targets = y[1:]                            # next-step targets

    model = HPT_QRC_Multi(window=5, photon_list=[3], n_virtual_nodes=1, seed=args.seed)
    spec = export_reservoir(model, 0)
    X_win = build_windows(model, series)       # (n_total, n_enc)
    assert len(X_win) == n_total

    # ---- quantum features: hardware (one job) and exact sim -------------
    if args.local:
        proc = pcvl.Processor("SLOS", spec["n_modes"])
    else:
        from hw_backend import get_processor
        print(f"[hw-native] submitting 1 job: {n_total} iterations x {args.shots} shots "
              f"= {n_total * args.shots:,} samples on {platform}")
        proc = get_processor(platform)

    result = sample_batch(proc, spec, X_win, args.shots, platform=platform,
                          chunk_size=None if args.local else args.chunk)
    Q_hw = np.array([s["features"] for s in result["steps"]])

    layer = model.reservoirs[0]
    with torch.no_grad():
        Q_sim = layer(torch.tensor(X_win, dtype=torch.float32)).numpy()

    # ---- three readouts, identical protocol ------------------------------
    tr = slice(args.discard, args.train)
    te = slice(args.train, n_total)
    feats = {
        "hw-native":      np.hstack([Q_hw, X_win]),
        "sim":            np.hstack([Q_sim, X_win]),
        "classical-only": X_win,
    }
    scores = {}
    preds = {}
    for name, F in feats.items():
        r = Ridge(alpha=args.ridge_alpha).fit(F[tr], targets[tr])
        p = r.predict(F[te])
        preds[name] = p
        scores[name] = float(mean_squared_error(targets[te], p))
    corr = [float(np.corrcoef(Q_hw[i], Q_sim[i])[0, 1]) for i in range(n_total)]

    print(f"\n=== NARMA-10, {args.train} train / {args.test} test, "
          f"{args.shots} shots/step, {platform} ===")
    for name in ("sim", "hw-native", "classical-only"):
        print(f"  {name:15} MSE {scores[name]:.6f}")
    print(f"  hw/sim ratio: {scores['hw-native'] / scores['sim']:.2f}x")
    print(f"  quantum lift on HW: classical-only / hw-native = "
          f"{scores['classical-only'] / scores['hw-native']:.2f}x  (>1 means HW features help)")
    print(f"  feature corr sim-vs-hw: mean {np.mean(corr):.3f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"hw_native_{platform.replace(':', '_')}_narma10.json"
    out.write_text(json.dumps({
        "platform": platform, "train": args.train, "test": args.test,
        "shots": args.shots, "discard": args.discard, "seed": args.seed,
        "ridge_alpha": args.ridge_alpha, "scores": scores,
        "y_test_true": targets[te].tolist(),
        "preds": {k: np.ravel(v).tolist() for k, v in preds.items()},
        "feature_corr": corr, "wall_time_s": result["wall_time_s"],
        "drop_rate": [s["drop_rate"] for s in result["steps"]],
    }))
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
