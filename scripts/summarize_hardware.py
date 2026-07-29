"""Collect every hardware / emulator run into one comparable table.

Reads the JSON files ``hardware/run_reservoir_hw.py`` writes and reports, per platform and
protocol: the three readouts fitted on identical timesteps (device, exact simulation,
classical-only), the device-vs-simulation feature correlation, and the quantum lift.

The lift is the number that decides whether the device contributed anything:
``classical_only / device`` must exceed 1.

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
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
