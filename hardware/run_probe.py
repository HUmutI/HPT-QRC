"""Cost probe (PLAN.md section 4) — the FIRST thing to run on the cloud.

One cheap job: 1 reservoir ([3] photons, virtual depth 1), 10 NARMA10 timesteps,
1000 shots/step. Measures real latency, coincidence/loss rate, and sim-vs-HW
fidelity, then extrapolates the budget for the validation subset.

Usage:
    python hardware/run_probe.py --platform sim:slos      # cloud simulator first (cheap rehearsal)
    python hardware/run_probe.py                          # then the real QPU (qpu:ascella)
    python hardware/run_probe.py --shots 500 --steps 5    # smaller probe

Results append to hardware/run_log.csv; raw job output cached under hardware/cache/.
Read your credit balance off the Quandela dashboard before and after, and note it
in the log's `credits_note` column by hand.
"""
import argparse
import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from src.multi_qrc import HPT_QRC_Multi
from src.data_loader import load_narma10
from phase_export import export_reservoir
from hw_backend import get_processor, DEFAULT_PLATFORM
from hw_features import sample_batch
from compare_sim_local import encoded_windows

RUN_LOG = Path(__file__).parent / "run_log.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", default=DEFAULT_PLATFORM)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--shots", type=int, default=1000)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    # Probe reservoir: cheapest defensible config
    model = HPT_QRC_Multi(window=5, photon_list=[3], n_virtual_nodes=1, seed=42)
    spec = export_reservoir(model, 0)
    print(f"Probe: {args.platform}, {args.steps} steps x {args.shots} shots, "
          f"{spec['n_modes']} modes / {spec['photons']} photons")

    X_train, _, _, _ = load_narma10()
    X_win = encoded_windows(model, X_train, args.steps)

    # Simulated reference for fidelity comparison
    core = model.reservoirs[0][0]
    with torch.no_grad():
        p_sim = core(torch.tensor(X_win, dtype=torch.float32)).numpy()

    proc = get_processor(args.platform)
    t0 = time.time()
    result = sample_batch(proc, spec, X_win, args.shots,
                          platform=args.platform, use_cache=not args.no_cache)
    total_wall = time.time() - t0

    # --- Analysis --------------------------------------------------------
    rows = []
    for i, step in enumerate(result["steps"]):
        p_hat = np.array(step["p_hat"])
        l1 = np.abs(p_hat - p_sim[i]).sum()
        corr = np.corrcoef(p_hat, p_sim[i])[0, 1]
        rows.append({
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "platform": args.platform,
            "step": i,
            "shots_requested": args.shots,
            "raw_counts": step["raw_counts"],
            "unbunched_counts": step["unbunched_counts"],
            "drop_rate": round(step["drop_rate"], 4),
            "physical_perf": step.get("physical_perf", ""),
            "logical_perf": step.get("logical_perf", ""),
            "l1_vs_sim": round(l1, 4),
            "corr_vs_sim": round(corr, 4),
            "job_wall_s": round(result["wall_time_s"], 1),
            "credits_note": "",
        })

    write_header = not RUN_LOG.exists()
    with RUN_LOG.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    l1s = [r["l1_vs_sim"] for r in rows]
    corrs = [r["corr_vs_sim"] for r in rows]
    drops = [r["drop_rate"] for r in rows]
    print(f"\n=== Probe summary ({args.platform}) ===")
    print(f"wall time: {total_wall:.1f}s total, {result['wall_time_s']:.1f}s in job "
          f"({result['wall_time_s'] / args.steps:.1f}s/step incl. queue)")
    print(f"L1 vs sim:   mean {np.mean(l1s):.4f}  worst {np.max(l1s):.4f}")
    print(f"corr vs sim: mean {np.mean(corrs):.4f}  worst {np.min(corrs):.4f}")
    print(f"drop rate:   mean {np.mean(drops):.4f}")

    # Extrapolation to the validation subset (PLAN.md section 3)
    subset_steps = 200
    per_step = result["wall_time_s"] / args.steps
    print(f"\nValidation subset extrapolation ({subset_steps} steps, 1 reservoir):")
    print(f"  time  ~ {subset_steps * per_step / 3600:.1f} h wall-clock")
    print(f"  shots ~ {subset_steps * args.shots:,} requested")
    print(f"Log appended to {RUN_LOG}")


if __name__ == "__main__":
    main()
