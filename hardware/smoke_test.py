"""Minimal QPU smoke test: 1 timestep of the probe reservoir circuit, 1 shot.

Validates the full path — token, connection, circuit compilation on Ascella,
sampling, histogram — at the smallest possible quota cost.

Usage:
    python hardware/smoke_test.py [--shots 1] [--platform qpu:ascella]
                                  [--attempts 3] [--wait 150]
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from src.multi_qrc import HPT_QRC_Multi
from src.data_loader import load_narma10
from phase_export import export_reservoir
from hw_backend import get_processor
from hw_features import sample_batch
from compare_sim_local import encoded_windows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", default="qpu:ascella")
    ap.add_argument("--shots", type=int, default=1)
    ap.add_argument("--attempts", type=int, default=3,
                    help="full submit cycles (each already has internal retries)")
    ap.add_argument("--wait", type=int, default=150,
                    help="seconds between cycles")
    args = ap.parse_args()

    model = HPT_QRC_Multi(window=5, photon_list=[3], n_virtual_nodes=1, seed=42)
    spec = export_reservoir(model, 0)
    X_train, _, _, _ = load_narma10()
    X_win = encoded_windows(model, X_train, 1)

    last_err = None
    for cycle in range(1, args.attempts + 1):
        try:
            proc = get_processor(args.platform)
            print(f"[smoke] cycle {cycle}/{args.attempts}: submitting "
                  f"{args.shots} shot(s) on {args.platform}", flush=True)
            t0 = time.time()
            res = sample_batch(proc, spec, X_win, shots=args.shots,
                               platform=args.platform, use_cache=False)
            step = res["steps"][0]
            print(f"[smoke] SUCCESS in {time.time() - t0:.1f}s", flush=True)
            print(f"  raw counts: {step['raw_counts']}, "
                  f"unbunched: {step['unbunched_counts']}", flush=True)
            nz = [(spec["output_keys"][i], p)
                  for i, p in enumerate(step["p_hat"]) if p > 0]
            print(f"  detected state(s): {nz[:5]}", flush=True)
            return 0
        except Exception as e:
            last_err = e
            print(f"[smoke] cycle {cycle} failed: {type(e).__name__}: {e}", flush=True)
            if cycle < args.attempts:
                print(f"[smoke] waiting {args.wait}s before next cycle", flush=True)
                time.sleep(args.wait)

    print(f"[smoke] all {args.attempts} cycles failed; cloud likely still down. "
          f"Last error: {last_err}", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
