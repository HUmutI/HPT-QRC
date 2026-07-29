"""When, if ever, is running this on photonics cheaper than simulating it?

The uncomfortable question. Classical simulation of the feature map costs arithmetic that
grows combinatorially with photon number; the photonic device costs *time*, because it must
accumulate enough coincidences, and its rate falls as ``transmittance ** n``. Both blow up
with ``n``. Which blows up faster decides whether the hardware is ever the cheaper option.

This computes both sides explicitly and solves for the transmittance at which a crossover
exists at all. The answer is a concrete hardware requirement rather than a hope, and it is
reported whichever way it comes out.

Classical side, per timestep, for the layered circuits used here:
    depth * m^3        complex multiply-adds to compose the unitary
    C(m, n) * n! * n   complex multiply-adds for the batch of permanents
Measured throughput is used rather than an assumed FLOP rate, so the comparison reflects
what this code actually achieves.

Photonic side, per timestep:
    shots / (clock * transmittance ** n)   seconds to collect the required coincidences

Usage::

    python experiments/crossover.py --shots 30000
"""

from __future__ import annotations

import argparse
import sys
import time
from math import comb, factorial
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.noise import ASCELLA, BELENOS, HardwareSpec  # noqa: E402
from src.photonic_core import LayeredInterferometer  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results" / "crossover"


def measured_throughput(n_modes: int = 8, n_photons: int = 3, repeats: int = 300) -> float:
    """Effective complex multiply-adds per second achieved by this implementation."""
    layer = LayeredInterferometer(n_modes, n_photons, depth=1, seed=0)
    phases = np.random.default_rng(0).uniform(0, 2 * np.pi, size=layer.n_encoding)
    layer.probabilities(phases)
    start = time.time()
    for _ in range(repeats):
        layer.probabilities(phases)
    seconds = (time.time() - start) / repeats
    work = classical_work(n_modes, n_photons, depth=1)
    return work / seconds


def classical_work(n_modes: int, n_photons: int, depth: int = 1) -> float:
    """Complex multiply-adds to evaluate the feature map for one timestep."""
    unitary = (depth + 1) * n_modes**3
    permanents = comb(n_modes, n_photons) * factorial(n_photons) * n_photons
    return float(unitary + permanents)


def classical_seconds(n_modes: int, n_photons: int, throughput: float, depth: int = 1) -> float:
    return classical_work(n_modes, n_photons, depth) / throughput


def photonic_seconds(spec: HardwareSpec, n_photons: int, shots: int) -> float:
    rate = spec.clock_hz * spec.transmittance**n_photons
    return float("inf") if rate <= 0 else shots / rate


def required_transmittance(n_modes: int, n_photons: int, clock_hz: float, shots: int,
                           throughput: float) -> float:
    """Transmittance at which the device matches classical simulation for this ``n``.

    Solves ``shots / (clock * t^n) = classical_seconds`` for ``t``.
    """
    target = classical_seconds(n_modes, n_photons, throughput)
    if target <= 0:
        return 1.0
    return float((shots / (clock_hz * target)) ** (1.0 / n_photons))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=30_000,
                    help="coincidences per timestep (30000 is the measured threshold at "
                         "which the reservoir beats its classical control)")
    ap.add_argument("--modes", type=int, default=24, help="Belenos mode count")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    throughput = measured_throughput()
    print(f"measured classical throughput: {throughput:.3e} complex MACs/s "
          f"(this implementation, single core)\n")

    rows = []
    for spec in (ASCELLA, BELENOS):
        for n in range(2, 13):
            classical = classical_seconds(args.modes, n, throughput)
            device = photonic_seconds(spec, n, args.shots)
            rows.append(dict(
                platform=spec.name, n_photons=n, fock_dim=comb(args.modes, n),
                classical_s=classical, device_s=device,
                device_over_classical=device / classical if classical > 0 else np.inf,
                required_transmittance=required_transmittance(
                    args.modes, n, spec.clock_hz, args.shots, throughput),
            ))

    frame = pd.DataFrame(rows)
    out = RESULTS / "crossover.csv"
    frame.to_csv(out, index=False)

    for spec in (ASCELLA, BELENOS):
        sub = frame[frame.platform == spec.name]
        print(f"=== {spec.name}  (clock {spec.clock_hz:.2e} Hz, transmittance "
              f"{spec.transmittance:.4f}) ===")
        print(f"{'n':>3} {'Fock dim':>10} {'classical':>12} {'device':>14} "
              f"{'device/classical':>18} {'t needed':>10}")
        for _, r in sub.iterrows():
            print(f"{int(r.n_photons):>3} {int(r.fock_dim):>10} "
                  f"{r.classical_s * 1e3:>10.4f}ms {r.device_s:>12.3e}s "
                  f"{r.device_over_classical:>18.2e} {r.required_transmittance:>10.4f}")
        crossing = sub[sub.device_over_classical < 1.0]
        if crossing.empty:
            best = sub.loc[sub.device_over_classical.idxmin()]
            print(f"\n  No crossover at any photon number. The device is closest at n="
                  f"{int(best.n_photons)}, still {best.device_over_classical:.1e}x slower.")
            feasible = sub[sub.required_transmittance <= 1.0]
            if not feasible.empty:
                row = feasible.loc[feasible.n_photons.idxmax()]
                print(f"  A crossover would need transmittance >= "
                      f"{row.required_transmittance:.3f} at n={int(row.n_photons)} "
                      f"(currently {spec.transmittance:.4f}, i.e. "
                      f"{row.required_transmittance / spec.transmittance:.0f}x better).")
        else:
            row = crossing.loc[crossing.n_photons.idxmin()]
            print(f"\n  Crossover at n={int(row.n_photons)}.")
        print()

    print(f"wrote {out}")


if __name__ == "__main__":
    main()
