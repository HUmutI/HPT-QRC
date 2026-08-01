"""Collect every hardware / emulator run into one comparable table.

Reports, per platform and protocol: the three readouts fitted on identical timesteps
(device, exact simulation, classical-only), the device-vs-simulation feature correlation,
and the quantum lift.

The lift is the number that decides whether the device contributed anything:
``classical_only / device`` must exceed 1.

Three schemas are read, because the QPU path was split into submit/harvest partway through
and writes different files. Globbing only ``reservoir_hw_*.json`` -- the emulator schema --
made this script print "No QPU runs present" while eight completed Belenos runs sat in the
same directory, which is exactly the wrong thing for the script whose job is to say whether
there is hardware evidence.

Usage::

    python scripts/summarize_hardware.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "hardware" / "results"


def main() -> None:
    rows = []
    for path in sorted(RESULTS.glob("reservoir_hw_*.json")):
        data = json.loads(path.read_text())
        scores = data.get("nrmse", {})
        rows.append(
            dict(
                platform=data.get("platform"),
                protocol=data.get("protocol"),
                steps=data.get("steps"),
                shots=data.get("shots"),
                modes=data.get("modes"),
                photons=data.get("photons"),
                reservoirs=data.get("reservoirs"),
                device=scores.get("hardware"),
                simulation=scores.get("simulation"),
                classical=scores.get("classical_only"),
                lift=data.get("quantum_lift_vs_classical"),
                corr_mean=data.get("feature_correlation_mean"),
                corr_min=data.get("feature_correlation_min"),
                wall_s=data.get("wall_time_s"),
            )
        )
    # Per-job QPU results, one file per submitted slice.
    for path in sorted(RESULTS.glob("qpu_qpu_*.json")):
        data = json.loads(path.read_text())
        scores = data.get("nrmse", {})
        raw = data.get("raw_counts_per_step")
        rows.append(
            dict(
                platform=data.get("platform"),
                protocol=f"slice[{data.get('job_id', '')[:8]}]",
                steps=data.get("steps"),
                shots=data.get("shots"),
                modes=data.get("modes"),
                photons=data.get("photons"),
                device=scores.get("hardware"),
                simulation=scores.get("simulation"),
                classical=scores.get("classical_only"),
                lift=data.get("quantum_lift_vs_classical"),
                corr_mean=data.get("feature_correlation_mean"),
                corr_min=data.get("feature_correlation_min"),
                drop_rate=data.get("drop_rate_mean"),
                raw_counts_per_step=raw if not isinstance(raw, list) else float(pd.Series(raw).mean()),
                job_state=data.get("job_state"),
            )
        )

    # The stitched run across all trajectory-consistent slices -- the headline QPU number.
    combined = RESULTS / "qpu_combined.json"
    if combined.exists():
        data = json.loads(combined.read_text())
        scores = data.get("nrmse", {})
        rows.append(
            dict(
                platform=data.get("platform"),
                protocol=f"COMBINED[{data.get('slices')} slices]",
                steps=data.get("total_timesteps"),
                shots=data.get("shots_per_step"),
                modes=data.get("modes"),
                photons=data.get("photons"),
                device=scores.get("hardware"),
                simulation=scores.get("simulation"),
                classical=scores.get("classical_only"),
                lift=data.get("quantum_lift_vs_classical"),
                corr_mean=data.get("feature_correlation_mean"),
                corr_min=data.get("feature_correlation_min"),
                train_rows=data.get("train_rows"),
                feature_dim=data.get("feature_dim"),
            )
        )

    if not rows:
        print("no hardware runs found")
        return

    frame = pd.DataFrame(rows).sort_values(["platform", "protocol"])
    out = RESULTS / "hardware_summary.csv"
    frame.to_csv(out, index=False)

    display = frame[
        ["platform", "protocol", "steps", "shots", "device", "simulation", "classical",
         "lift", "corr_mean", "wall_s"]
    ]
    print(display.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nlift = classical / device; > 1 means the reservoir helped")

    real = frame[frame.platform.str.startswith("qpu:")]
    if real.empty:
        print("\nNo QPU runs present -- every row above is a simulator or device emulator.")
    else:
        slices = real[real.protocol.str.startswith("slice")]
        print(f"\n{len(slices)} QPU jobs on {', '.join(sorted(real.platform.unique()))}; "
              f"feature correlation vs simulation "
              f"{slices.corr_mean.min():.3f}-{slices.corr_mean.max():.3f} across shot counts "
              f"{sorted(int(s) for s in slices.shots.unique())}.")
        print("  Per-slice NRMSE is not interpretable -- a 17-22 timestep slice has fewer "
              "rows than features. Only the stitched run is fitted on enough data to score.")
        head = real[real.protocol.str.startswith("COMBINED")]
        if not head.empty:
            row = head.iloc[0]
            print(f"Stitched run: {int(row.steps)} timesteps, hardware {row.device:.4f} vs "
                  f"simulation {row.simulation:.4f} vs classical {row.classical:.4f}.")
            print(f"  {int(row.train_rows)} training rows against {int(row.feature_dim)} "
                  f"features -- data-starved, so this is not an accuracy claim. The reportable "
                  f"result is the feature-level agreement above.")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
