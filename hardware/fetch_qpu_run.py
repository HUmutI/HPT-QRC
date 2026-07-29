"""Recover results from QPU jobs submitted earlier by ``submit_qpu_run.py``.

Run this any time after submission, from any machine with the token. It checks each pending
job, and for any that have completed, histograms the returned samples into the unbunched Fock
subspace, replays the leaky integration over the measured probabilities, fits the three
readouts (hardware, simulation, classical-only) on identical timesteps, and writes a result
file.

Everything needed to do that was stored at submission time, so nothing depends on the
submitting session still being alive.

Usage::

    python hardware/fetch_qpu_run.py            # check and harvest anything finished
    python hardware/fetch_qpu_run.py --watch    # poll until every pending job resolves
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "hardware"))

from src.rc_protocol import Split, evaluate_features  # noqa: E402

PENDING = ROOT / "hardware" / "pending_qpu_runs.json"
RESULTS = ROOT / "hardware" / "results"


def counts_to_probabilities(results, output_keys) -> np.ndarray:
    """Histogram one BSCount into the unbunched subspace, in ``output_keys`` order."""
    index = {tuple(k): i for i, k in enumerate(output_keys)}
    counts = np.zeros(len(output_keys))
    total = kept = 0
    for state, count in results.items():
        total += count
        position = index.get(tuple(int(x) for x in state))
        if position is not None:
            counts[position] += count
            kept += count
    if kept == 0:
        return np.full(len(output_keys), 1.0 / len(output_keys)), total, 1.0
    return counts / kept, total, 1.0 - kept / total


def harvest(record: dict) -> dict | None:
    import perceval as pcvl

    from hw_backend import get_token

    handler = pcvl.RemoteProcessor(record["platform"], token=get_token())._rpc_handler
    status = handler.get_job_status(record["job_id"])
    state = status.get("status")
    print(f"  {record['job_id'][:8]} on {record['platform']}: {state} "
          f"(progress {str(status.get('progress'))[:6]})")
    if state != "completed":
        return None

    job = pcvl.RemoteJob.from_id(record["job_id"], handler) \
        if hasattr(pcvl, "RemoteJob") else None
    payload = job.get_results() if job is not None else handler.get_job_results(
        record["job_id"])
    results_list = payload["results_list"] if "results_list" in payload else payload["results"]

    keys = record["output_keys"]
    probs, raw, drops = [], [], []
    for entry in results_list:
        counts = entry["results"] if isinstance(entry, dict) and "results" in entry else entry
        p, total, drop = counts_to_probabilities(counts, keys)
        probs.append(p)
        raw.append(total)
        drops.append(drop)
    probs = np.array(probs)

    # Replay the same leaky integration the simulation used, over measured probabilities.
    leak = float(record["leak"])
    state_vec = np.full(probs.shape[1], 1.0 / probs.shape[1])
    hw_states = np.empty_like(probs)
    for t in range(len(probs)):
        state_vec = (1.0 - leak) * state_vec + leak * probs[t]
        hw_states[t] = state_vec

    sim_states = np.asarray(record["sim_states"])[: len(probs)]
    targets = np.asarray(record["targets"])[: len(probs)]
    scaled = np.asarray(record["input_scaled"])[: len(probs)]
    window = 20
    padded = np.vstack([np.zeros((window - 1, scaled.shape[1])), scaled])
    classical = np.stack([padded[i : i + window].ravel() for i in range(len(probs))])

    steps = len(probs)
    n_train = int(0.7 * steps)
    split = Split(steps, washout=min(20, n_train // 3), n_train=n_train,
                  n_val=max(5, n_train // 5))
    scores = {
        name: evaluate_features(features, targets, split)["nrmse"]
        for name, features in {
            "hardware": np.hstack([hw_states, classical]),
            "simulation": np.hstack([sim_states, classical]),
            "classical_only": classical,
        }.items()
    }
    correlations = [float(np.corrcoef(hw_states[t], sim_states[t])[0, 1])
                    for t in range(steps)]

    report = {
        **{k: record[k] for k in ("job_id", "platform", "dataset", "steps", "shots",
                                  "modes", "photons", "seed", "submitted_utc")},
        "nrmse": scores,
        "quantum_lift_vs_classical": scores["classical_only"] / scores["hardware"],
        "hardware_vs_simulation_ratio": scores["hardware"] / scores["simulation"],
        "feature_correlation_mean": float(np.mean(correlations)),
        "feature_correlation_min": float(np.min(correlations)),
        "raw_counts_per_step": raw,
        "drop_rate_mean": float(np.mean(drops)),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"qpu_{record['platform'].replace(':', '_')}_{record['job_id'][:8]}.json"
    out.write_text(json.dumps(report, indent=2))

    print(f"\n=== {record['platform']} — REAL QPU RESULT ===")
    for name, value in sorted(scores.items(), key=lambda kv: kv[1]):
        print(f"  {name:<16} NRMSE {value:.4f}")
    print(f"  quantum lift (classical/hardware): "
          f"{report['quantum_lift_vs_classical']:.3f}  (>1 means the device helped)")
    print(f"  hardware/simulation feature correlation: mean "
          f"{report['feature_correlation_mean']:.3f}, min "
          f"{report['feature_correlation_min']:.3f}")
    print(f"  raw counts per step: {min(raw)}–{max(raw)}, mean drop rate "
          f"{report['drop_rate_mean']:.3f}")
    print(f"\nwrote {out}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true", help="poll until all jobs resolve")
    ap.add_argument("--interval", type=int, default=300)
    args = ap.parse_args()

    if not PENDING.exists():
        print(f"no pending jobs recorded in {PENDING}")
        return

    while True:
        pending = json.loads(PENDING.read_text())
        remaining = []
        print(f"checking {len(pending)} pending job(s)...")
        for record in pending:
            try:
                if harvest(record) is None:
                    remaining.append(record)
            except Exception as exc:
                print(f"  {record['job_id'][:8]}: {type(exc).__name__}: {exc}")
                remaining.append(record)
        PENDING.write_text(json.dumps(remaining, indent=2))
        if not remaining or not args.watch:
            if remaining:
                print(f"\n{len(remaining)} job(s) still pending. Re-run this script later.")
            else:
                print("\nall pending jobs harvested.")
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
