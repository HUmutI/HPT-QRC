"""Stitch harvested QPU slices into one trajectory and fit the readouts on all of it.

Each slice was measured as its own cloud job because the tier kills anything past five
minutes, but they share one phase trajectory by construction, so concatenating them in
timestep order reconstructs a single continuous run.

Only with enough concatenated timesteps does an accuracy number mean anything: a 20-step
slice leaves ~10 training rows against ~85 features, which is why the per-slice NRMSE values
are not interpretable on their own.

Usage::

    python hardware/combine_qpu_slices.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rc_protocol import Split, evaluate_features  # noqa: E402

SLICES = ROOT / "hardware" / "results" / "slices"
RESULTS = ROOT / "hardware" / "results"


def main() -> None:
    files = sorted(SLICES.glob("slice_*.json")) if SLICES.exists() else []
    if not files:
        print(f"no slices in {SLICES}; run hardware/run_qpu_campaign.sh first")
        return

    records = [json.loads(f.read_text()) for f in files]

    # total_steps sets n_train, which sets the input scaler, which sets the phases. Slices
    # submitted under different total_steps therefore sample *different* trajectories and
    # must not be concatenated: doing so produced a "simulation" reference scoring 1.06,
    # worse than the noiseless bound can be. Keep only the largest consistent group.
    groups: dict = {}
    for r in records:
        groups.setdefault(r.get("total_steps"), []).append(r)
    if len(groups) > 1:
        best = max(groups.values(), key=lambda g: sum(len(r["hw_probs"]) for r in g))
        for key, g in groups.items():
            if g is not best:
                print(f"  excluding {len(g)} slice(s) from trajectory total_steps={key}: "
                      f"{[r['job_id'][:8] for r in g]} -- different input scaler, so "
                      f"different phases")
        records = best
    records.sort(key=lambda r: (r.get("slice_start", 0), -len(r["hw_probs"])))

    # Slices can overlap when a run was retried: keep, for each starting timestep, the
    # longest measurement, and drop anything the retained slices already cover. Stacking
    # overlapping slices would double-count timesteps and corrupt the state replay.
    kept, covered = [], set()
    for r in records:
        start, n = r["slice_start"], len(r["hw_probs"])
        # Trim the already-covered prefix rather than discarding the whole slice: a retry
        # that overlaps by two timesteps still contributes every timestep beyond them.
        offset = 0
        while offset < n and (start + offset) in covered:
            offset += 1
        if offset >= n:
            print(f"  skipping {r['job_id'][:8]} ({start}-{start + n - 1} fully covered)")
            continue
        if offset:
            print(f"  trimming {r['job_id'][:8]}: dropping {offset} overlapping timestep(s), "
                  f"keeping {start + offset}-{start + n - 1}")
            for key in ("hw_probs", "sim_states", "targets", "input_scaled"):
                r[key] = r[key][offset:]
            r["slice_start"] = start + offset
        kept.append(r)
        covered |= set(range(r["slice_start"], r["slice_start"] + len(r["hw_probs"])))
    records = kept
    gaps = sorted(set(range(min(covered), max(covered) + 1)) - covered) if covered else []
    if gaps:
        print(f"  WARNING: {len(gaps)} missing timesteps in {min(covered)}-{max(covered)}; "
              f"the state replay assumes contiguity")
    shots = {r["shots"] for r in records}
    if len(shots) > 1:
        print(f"  NOTE: slices were measured at different shot counts {sorted(shots)}, so "
              f"their\n        features carry different noise levels")

    hw = np.vstack([np.asarray(r["hw_probs"]) for r in records])
    sim = np.vstack([np.asarray(r["sim_states"]) for r in records])
    targets = np.concatenate([np.asarray(r["targets"]) for r in records])
    scaled = np.vstack([np.asarray(r["input_scaled"]) for r in records])
    leak = float(records[0]["leak"])

    # Replay the leaky integration across the *whole* concatenated trajectory, not per slice,
    # so the state carries between slices exactly as it would in one continuous run.
    state = np.full(hw.shape[1], 1.0 / hw.shape[1])
    hw_states = np.empty_like(hw)
    for t in range(len(hw)):
        state = (1.0 - leak) * state + leak * hw[t]
        hw_states[t] = state

    window = 20
    padded = np.vstack([np.zeros((window - 1, scaled.shape[1])), scaled])
    classical = np.stack([padded[i : i + window].ravel() for i in range(len(hw))])

    steps = len(hw)
    n_train = int(0.7 * steps)
    split = Split(steps, washout=min(30, n_train // 4), n_train=n_train,
                  n_val=max(8, n_train // 5))
    scores = {
        name: evaluate_features(features, targets, split)["nrmse"]
        for name, features in {
            "hardware": np.hstack([hw_states, classical]),
            "simulation": np.hstack([sim, classical]),
            "classical_only": classical,
        }.items()
    }
    correlations = [float(np.corrcoef(hw_states[t], sim[t])[0, 1]) for t in range(steps)]

    report = {
        "platform": records[0]["platform"],
        "slices": len(records),
        "total_timesteps": steps,
        "shots_per_step": records[0]["shots"],
        "photons": records[0]["photons"],
        "modes": records[0]["modes"],
        "nrmse": scores,
        "quantum_lift_vs_classical": scores["classical_only"] / scores["hardware"],
        "hardware_vs_simulation_ratio": scores["hardware"] / scores["simulation"],
        "feature_correlation_mean": float(np.mean(correlations)),
        "feature_correlation_min": float(np.min(correlations)),
        "train_rows": n_train - split.washout,
        "feature_dim": hw_states.shape[1] + classical.shape[1],
    }
    out = RESULTS / "qpu_combined.json"
    out.write_text(json.dumps(report, indent=2))

    print(f"=== {report['platform']}: {steps} timesteps from {len(records)} slices ===")
    for name, value in sorted(scores.items(), key=lambda kv: kv[1]):
        print(f"  {name:<16} NRMSE {value:.4f}")
    print(f"  quantum lift (classical/hardware): {report['quantum_lift_vs_classical']:.3f}")
    print(f"  feature correlation vs simulation: mean "
          f"{report['feature_correlation_mean']:.3f}, min {report['feature_correlation_min']:.3f}")
    print(f"  {report['train_rows']} training rows against {report['feature_dim']} features")
    if report["train_rows"] < report["feature_dim"]:
        print("  NOTE: fewer training rows than features -- the readout is underdetermined "
              "and\n        the NRMSE values above should not be read as an accuracy result.")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
