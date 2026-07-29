"""Do Quandela's device emulators actually apply a device model?

``sim:ascella`` and ``sim:belenos`` are described as emulators of the corresponding
processors, but the platform metadata does not say what noise, if any, they apply --- and
``sim:ascella`` self-describes as "Arcturus simulator". Their returned counts are not
informative on their own, because the server applies the coincidence filter and returns
exactly the requested number of accepted samples, so the client-side drop rate reads zero
regardless.

The runs in ``hardware/`` drive every platform with the *same* circuit and the *same* phase
trajectory, and each caches its per-timestep distribution. Comparing those cached
distributions answers the question directly: if a device emulator differs from ``sim:slos``
by no more than two independent ``sim:slos`` runs differ from each other, it is applying no
device model that this experiment can detect.

Usage::

    python scripts/compare_platforms.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parents[1] / "hardware" / "cache"


def load_jobs() -> dict[str, list[dict]]:
    by_platform = defaultdict(list)
    for path in sorted(CACHE.glob("job_*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if data.get("steps"):
            by_platform[data.get("platform", "?")].append(data)
    return by_platform


def distributions(jobs: list[dict]) -> np.ndarray:
    rows = [step["p_hat"] for job in jobs for step in job["steps"]]
    return np.array(rows) if rows else np.empty((0, 0))


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    n = min(len(a), len(b))
    if n == 0 or a.shape[1] != b.shape[1]:
        return {}
    x, y = a[:n], b[:n]
    tvd = 0.5 * np.abs(x - y).sum(axis=1)
    corr = [float(np.corrcoef(x[i], y[i])[0, 1]) for i in range(n)]
    return {
        "steps": n,
        "total_variation_mean": float(tvd.mean()),
        "total_variation_max": float(tvd.max()),
        "correlation_mean": float(np.mean(corr)),
        "correlation_min": float(np.min(corr)),
    }


def main() -> None:
    by_platform = load_jobs()
    if not by_platform:
        print("no cached jobs found")
        return

    print("cached distributions per platform:")
    dists = {}
    for platform, jobs in sorted(by_platform.items()):
        d = distributions(jobs)
        dists[platform] = d
        print(f"  {platform:<14} {len(d):>5} timesteps, {d.shape[1] if len(d) else 0} bins")

    reference = "sim:slos"
    if reference not in dists:
        print(f"\nno {reference} run cached; run the control first")
        return

    print(f"\nagainst {reference} (same circuit, same phase trajectory):")
    for platform, d in sorted(dists.items()):
        if platform == reference or d.shape[1] != dists[reference].shape[1]:
            continue
        stats = compare(d, dists[reference])
        if not stats:
            continue
        print(f"  {platform:<14} TVD mean {stats['total_variation_mean']:.4f} "
              f"max {stats['total_variation_max']:.4f}  |  corr mean "
              f"{stats['correlation_mean']:.4f} min {stats['correlation_min']:.4f} "
              f"({stats['steps']} steps)")

    print(
        "\nA device emulator applying real noise should sit clearly further from sim:slos\n"
        "than sampling scatter alone would put it. Both were run at the same shot count,\n"
        "so any excess total variation is the device model."
    )


if __name__ == "__main__":
    main()
