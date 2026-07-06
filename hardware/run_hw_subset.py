"""Validation subset driver (PLAN.md section 3).

Trains the readout entirely in simulation (free), then re-extracts the quantum
features of the TEST window on hardware and compares forecasts:

    sim path:  test windows -> merlin exact p-vector -> LexGrouping -> Ridge
    hw  path:  test windows -> QPU shots -> histogram p_hat -> LexGrouping -> Ridge
                                            (same Ridge, same classical context)

The scientific claim: the sim-trained readout still forecasts when the quantum
features come from real hardware, i.e. the feature map survives shot noise,
loss, and partial distinguishability.

Usage:
    python hardware/run_hw_subset.py --local                 # free local SLOS end-to-end test
    python hardware/run_hw_subset.py --steps 200 --shots 1000  # real Ascella run (ASK FIRST)

Every cloud job is cached by hw_features.sample_batch — re-runs are free.
Writes hardware/results/subset_<platform>_<dataset>.json for compare_sim_hw.py.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import perceval as pcvl
from src.multi_qrc import HPT_QRC_Multi
from src.data_loader import load_narma10
from phase_export import export_reservoir
from hw_features import sample_batch

RESULTS_DIR = Path(__file__).parent / "results"


def build_windows(model, X):
    """Replicate HPT_QRC_Multi._create_features windowing exactly (no scaler —
    the NARMA pipeline feeds raw values as phases)."""
    X_dim = model.in_size
    padded = np.vstack([np.zeros((model.window - 1, X_dim)), np.asarray(X).reshape(-1, X_dim)])
    wins = []
    for t in range(model.window - 1, len(padded)):
        w = padded[t - model.window + 1: t + 1].flatten()
        if len(w) < model.n_enc_actual:
            w = np.pad(w, (0, model.n_enc_actual - len(w)))
        wins.append(w[: model.n_enc_actual])
    return np.array(wins)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", default="qpu:ascella")
    ap.add_argument("--local", action="store_true",
                    help="use a local SLOS processor — free, no cloud job")
    ap.add_argument("--dataset", default="narma10", choices=["narma10"])
    ap.add_argument("--steps", type=int, default=200, help="test steps to run on HW")
    ap.add_argument("--shots", type=int, default=1000)
    ap.add_argument("--photons", type=int, default=3)
    ap.add_argument("--vd", type=int, default=1, help="virtual depth (n_virtual_nodes)")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    platform = "local:slos" if args.local else args.platform

    # ---- 1. Simulated training (free) -----------------------------------
    _, y_train, _, y_test = load_narma10(seed=args.seed)
    y_test = y_test[: args.steps]
    model = HPT_QRC_Multi(window=args.window, photon_list=[args.photons],
                          n_virtual_nodes=args.vd, seed=args.seed)
    model.fit(y_train)
    sim_pred = model.predict(y_test)
    mse_sim = float(np.mean((y_test.ravel() - sim_pred.ravel()) ** 2))
    print(f"[subset] sim-only test MSE: {mse_sim:.6f}")

    # ---- 2. Rebuild the exact test windows model.predict() used ---------
    features = y_test
    concat = np.vstack([model.last_X, features[:-1]])
    all_wins = build_windows(model, concat)
    X_win = all_wins[-len(features):]

    # Sim reference features for the same windows (through LexGrouping)
    layer = model.reservoirs[0]
    with torch.no_grad():
        Q_sim = layer(torch.tensor(X_win, dtype=torch.float32)).numpy()

    # ---- 3. Hardware (or local-sampled) quantum features ----------------
    spec = export_reservoir(model, 0)
    if args.local:
        proc = pcvl.Processor("SLOS", spec["n_modes"])
    else:
        from hw_backend import get_processor
        est_shots = args.steps * args.shots
        print(f"[subset] about to submit 1 job: {args.steps} iterations x "
              f"{args.shots} shots = {est_shots:,} shots on {platform}")
        proc = get_processor(platform)

    result = sample_batch(proc, spec, X_win, args.shots, platform=platform)
    Q_hw = np.array([s["features"] for s in result["steps"]])

    # ---- 4. Forecast with the SAME sim-trained Ridge ---------------------
    classical = X_win  # NARMA path: raw window context (use_har_context=False)
    hw_pred = model.ridge.predict(np.hstack([Q_hw, classical]))
    mse_hw = float(np.mean((y_test.ravel() - hw_pred.ravel()) ** 2))

    feat_corr = [float(np.corrcoef(Q_hw[i], Q_sim[i])[0, 1]) for i in range(len(Q_hw))]
    drop = [s["drop_rate"] for s in result["steps"]]
    print(f"[subset] HW-feature test MSE: {mse_hw:.6f}  (sim {mse_sim:.6f}, "
          f"ratio {mse_hw / mse_sim:.2f}x)")
    print(f"[subset] feature corr sim-vs-hw: mean {np.mean(feat_corr):.4f}  "
          f"min {np.min(feat_corr):.4f}")
    print(f"[subset] drop rate: mean {np.mean(drop):.4f}")

    # ---- 5. Persist for compare_sim_hw.py --------------------------------
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"subset_{platform.replace(':', '_')}_{args.dataset}.json"
    out.write_text(json.dumps({
        "platform": platform, "dataset": args.dataset, "steps": args.steps,
        "shots": args.shots, "photons": args.photons, "vd": args.vd,
        "window": args.window, "seed": args.seed,
        "mse_sim": mse_sim, "mse_hw": mse_hw,
        "y_test": y_test.ravel().tolist(),
        "sim_pred": np.ravel(sim_pred).tolist(),
        "hw_pred": np.ravel(hw_pred).tolist(),
        "Q_sim": Q_sim.tolist(), "Q_hw": Q_hw.tolist(),
        "p_hat": [s["p_hat"] for s in result["steps"]],
        "drop_rate": drop, "wall_time_s": result["wall_time_s"],
    }))
    print(f"[subset] wrote {out}")


if __name__ == "__main__":
    main()
