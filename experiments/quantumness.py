"""Does quantum interference contribute, or is this a classical random feature map?

This is the experiment that decides whether "photonic" is doing work in this architecture,
and it is the one a reviewer will ask for first. Random Fourier features come close to the
photonic model on several tasks, so the interesting question is not "does a random nonlinear
map help" --- it plainly does --- but "does the *interference* help".

The test is a clean one, because partial distinguishability interpolates exactly between the
two hypotheses at otherwise identical everything:

* ``indistinguishability = 0`` --- photons behave as classical distinguishable particles. The
  output distribution is the permanent of ``|U|^2``, a positive matrix. This is a classical
  stochastic feature map, computable without any quantum interference.
* ``indistinguishability = 1`` --- fully indistinguishable bosons. The distribution is
  ``|perm(U)|^2``, with interference between the ``n!`` photon assignments.

Same circuit, same phases, same shot count, same readout, same everything else. Any accuracy
difference is attributable to interference alone. If there is none, the honest conclusion is
that the architecture is a classical random feature map implemented in optics, and the paper
should say so.

Evaluated in the infinite-shot limit so sampling noise cannot mask the effect, through
Perceval's device model rather than an approximation of ours.

Usage::

    python experiments/quantumness.py --datasets narma10 parity_d3 channel_eq --seeds 3
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.noise import HardwareSpec  # noqa: E402
from src.rc_protocol import evaluate_features  # noqa: E402
from src.tasks import load_task  # noqa: E402
from src.temporal_qrc import TemporalPhotonicQRC  # noqa: E402

warnings.filterwarnings("ignore")
RESULTS = Path(__file__).resolve().parents[1] / "results" / "quantumness"

VISIBILITIES = [0.0, 0.25, 0.5, 0.75, 1.0]
# Interference structure grows with photon number: two photons have only the Hong-Ou-Mandel
# term, three have a genuinely richer permanent. Both are tested because two is the
# hardware-viable choice and three is where the effect should be larger if it exists.
PHOTON_NUMBERS = [2, 3]

# Encoding window matched to each task's own memory order where that is known; NARMA-N needs
# both u_t and u_{t-N+1} in one encoding, and the parity task needs delay + order.
ENCODE_WINDOW = {
    "narma5": 8, "narma10": 12, "narma20": 25, "parity_d3": 8,
    "channel_eq": 12, "santa_fe": 12, "henon": 6, "mackey_glass_h17": 12,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["narma10", "parity_d3", "channel_eq", "santa_fe", "henon"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--modes", type=int, default=10)
    ap.add_argument("--reservoirs", type=int, default=2)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []

    for dataset in args.datasets:
        u, y, split = load_task(dataset)
        for n_photons in PHOTON_NUMBERS:
            for visibility in VISIBILITIES:
                spec = HardwareSpec(name=f"V={visibility:.2f}",
                                    indistinguishability=visibility)
                for seed in range(42, 42 + args.seeds):
                    model = TemporalPhotonicQRC(
                        n_modes=args.modes,
                        photon_list=(n_photons,),
                        reservoirs_per_photon=args.reservoirs,
                        depth=1, leak=0.1, g_in=0.1, g_fb=0.3,
                        encode_window=ENCODE_WINDOW.get(dataset, 10),
                        window=20, washout=split.washout, seed=seed,
                        noise_spec=spec, n_samples=None,       # infinite-shot limit
                        threshold_detectors=True,
                    )
                    score = evaluate_features(
                        model.build_features(u, split.n_train), y, split
                    )["nrmse"]
                    rows.append(dict(dataset=dataset, n_photons=n_photons,
                                     visibility=visibility, seed=seed, nrmse=score))
                    print(f"  {dataset:<11} n={n_photons} V={visibility:.2f} seed={seed}"
                          f"  NRMSE {score:.4f}", flush=True)

    frame = pd.DataFrame(rows)
    out = RESULTS / "indistinguishability.csv"
    frame.to_csv(out, index=False)

    print("\n=== NRMSE vs indistinguishability (median over seeds) ===")
    pivot = frame.pivot_table(index=["dataset", "n_photons"], columns="visibility",
                              values="nrmse", aggfunc="median")
    print(pivot.to_string(float_format=lambda v: f"{v:.4f}"))

    print("\n=== interference effect: V=0 (classical particles) vs V=1 (interference) ===")
    print("    Paired across seeds, since a seed fixes both the circuit and the data.")
    print("    An effect smaller than the seed spread is not an effect.\n")
    print(f"    {'task':<12}{'n':>2}  {'V=0':>8} {'V=1':>8} {'gain':>7} {'paired d':>9} "
          f"{'seed sd':>8} {'p':>7}  verdict")

    from scipy import stats as sstats

    verdicts = []
    for (dataset, n_photons), _ in pivot.iterrows():
        sub = frame[(frame.dataset == dataset) & (frame.n_photons == n_photons)]
        a = sub[sub.visibility == 0.0].sort_values("seed")["nrmse"].to_numpy()
        b = sub[sub.visibility == 1.0].sort_values("seed")["nrmse"].to_numpy()
        if len(a) != len(b) or len(a) < 2:
            continue
        diff = a - b                                   # positive means V=1 is better
        gain = float(np.median(a) / np.median(b))
        spread = float(np.std(a, ddof=1))
        _, p_value = sstats.ttest_rel(a, b)
        # An effect must be both statistically detectable and larger than seed-to-seed noise
        # to be worth claiming.
        detectable = p_value < 0.05 and abs(np.mean(diff)) > 0.5 * spread
        verdict = ("interference helps" if detectable and np.mean(diff) > 0 else
                   "interference hurts" if detectable else "no detectable effect")
        verdicts.append(verdict)
        print(f"    {dataset:<12}{n_photons:>2}  {np.median(a):>8.4f} {np.median(b):>8.4f} "
              f"{gain:>7.3f} {np.mean(diff):>+9.4f} {spread:>8.4f} {p_value:>7.3f}  {verdict}")

    if verdicts and all(v == "no detectable effect" for v in verdicts):
        print("\n    No task shows a detectable interference effect. The honest reading is that"
              "\n    this architecture is a classical random feature map realised in optics:"
              "\n    the interferometer and Fock measurement supply a high-dimensional"
              "\n    nonlinear map, but the indistinguishability of the photons is not what"
              "\n    makes it work. This is the same fact as the insensitivity to"
              "\n    Hong-Ou-Mandel visibility in the noise study, seen from the other side,"
              "\n    and it lowers the hardware bar: a source need not be indistinguishable.")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
