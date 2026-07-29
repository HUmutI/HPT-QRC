"""Do Quandela's device emulators apply a device model, or only sampling?

``sim:ascella`` and ``sim:belenos`` are presented as emulators of the corresponding
processors, but the platform metadata does not state what noise they apply, and
``sim:ascella`` self-describes as "Arcturus simulator". Their returned counts cannot answer
the question: the server applies the coincidence filter and returns exactly the number of
accepted samples requested, so the client-side drop rate reads zero either way.

This settles it by submitting the *same* circuit at the *same* phase settings to each
platform and comparing the returned distributions against an exact local calculation. A
platform applying a real device model must sit measurably further from the exact
distribution than one that only samples.

Comparing previously cached runs does not work, because the cache stacks jobs in file order
with no guarantee that row *i* of one platform corresponds to row *i* of another. Every
distribution compared here comes from one shared phase batch.

Usage::

    python scripts/compare_platforms.py --steps 30 --shots 20000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "hardware"))

from src.photonic_core import LayeredInterferometer  # noqa: E402


def sample_platform(layer, phases, platform, shots, chunk):
    """Return (steps, bins) sampled distributions from one cloud platform."""
    import perceval as pcvl

    from hw_backend import get_processor
    from hw_features import sample_batch

    circuit, input_state = layer.to_perceval()
    spec = {
        "n_modes": layer.n_modes,
        "photons": layer.n_photons,
        "input_state": input_state,
        "output_keys": [
            tuple(1 if m in pat else 0 for m in range(layer.n_modes)) for pat in layer.patterns
        ],
        "input_names": [p.name for p in circuit.get_parameters()],
        "circuit": circuit,
    }
    if platform == "local":
        processor = pcvl.Processor("SLOS", circuit)
        name = "local:slos"
    else:
        processor = get_processor(platform, m=layer.n_modes)
        name = platform
    out = sample_batch(processor, spec, phases, shots, name,
                       max_shots_per_call=10_000_000,
                       chunk_size=None if platform == "local" else chunk)
    return np.array([s["p_hat"] for s in out["steps"]])


def stats(sampled: np.ndarray, exact: np.ndarray) -> dict:
    tvd = 0.5 * np.abs(sampled - exact).sum(axis=1)
    corr = [float(np.corrcoef(sampled[i], exact[i])[0, 1]) for i in range(len(sampled))]
    return {
        "tvd_mean": float(tvd.mean()),
        "tvd_max": float(tvd.max()),
        "corr_mean": float(np.mean(corr)),
        "corr_min": float(np.min(corr)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--platforms", nargs="+",
                    default=["sim:slos", "sim:ascella", "sim:belenos"])
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--shots", type=int, default=20000)
    ap.add_argument("--modes", type=int, default=12)
    ap.add_argument("--photons", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=30)
    args = ap.parse_args()

    layer = LayeredInterferometer(args.modes, args.photons, depth=1, seed=7)
    rng = np.random.default_rng(11)
    phases = rng.uniform(0, 2 * np.pi, size=(args.steps, layer.n_encoding))
    exact = layer.probabilities(phases)

    print(f"{args.steps} identical phase settings, {args.shots} shots, "
          f"{args.modes} modes / {args.photons} photons, {exact.shape[1]} Fock bins\n")

    report = {}
    for platform in args.platforms:
        try:
            sampled = sample_platform(layer, phases, platform, args.shots, args.chunk)
        except Exception as exc:
            print(f"  {platform:<14} FAILED: {type(exc).__name__}: {exc}")
            continue
        s = stats(sampled, exact)
        report[platform] = s
        print(f"  {platform:<14} TVD vs exact: mean {s['tvd_mean']:.4f} max {s['tvd_max']:.4f}"
              f"   corr: mean {s['corr_mean']:.4f} min {s['corr_min']:.4f}")

    if "sim:slos" in report:
        baseline = report["sim:slos"]["tvd_mean"]
        print(f"\nsim:slos is the sampling-only reference (TVD {baseline:.4f} at this shot "
              f"count).")
        for platform, s in report.items():
            if platform == "sim:slos":
                continue
            excess = s["tvd_mean"] - baseline
            verdict = ("applies a device model beyond sampling" if excess > 0.5 * baseline
                       else "indistinguishable from sampling alone")
            print(f"  {platform:<14} excess TVD {excess:+.4f}  ->  {verdict}")

    out = ROOT / "hardware" / "results" / "platform_comparison.json"
    out.write_text(json.dumps(
        {"steps": args.steps, "shots": args.shots, "modes": args.modes,
         "photons": args.photons, "results": report}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
