"""Noise-robustness study for the recurrent photonic reservoir.

Answers the question a hardware collaborator actually cares about: does the model still
work at the noise levels of a real device, and if not, which noise source breaks it?

Every noisy evaluation runs through Perceval's ``NoiseModel`` with threshold detectors, so
the numbers are produced by Quandela's own device model rather than an approximation.
Sweeps:

* ``shots``      -- coincidences collected per timestep, the finite-sampling limit
* ``indist``     -- Hong-Ou-Mandel visibility
* ``g2``         -- second-order correlation (multi-photon emission)
* ``hardware``   -- the measured Ascella and Belenos operating points, all sources at once
* ``joint``      -- shots x visibility, to see whether the two interact
* ``rate``       -- coincidence rate and achievable shots per timestep vs photon number

Each configuration is repeated over several seeds because shot noise is stochastic; the
tuned classical control is evaluated alongside so "still good" is measured against
something rather than asserted.

Usage::

    python experiments/noise_study.py --sweep all --dataset narma10 --seeds 3
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines_rc import lag_features  # noqa: E402
from src.noise import ASCELLA, BELENOS, IDEAL, HardwareSpec, coincidence_rate  # noqa: E402
from src.rc_protocol import evaluate_features  # noqa: E402
from src.tasks import load_task  # noqa: E402
from src.temporal_qrc import TemporalPhotonicQRC  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "noise"

# Kept small enough that a full sweep with Perceval in the loop finishes in reasonable time.
# Two photons is also the hardware-viable choice, so the noise study is run on the same
# configuration a QPU would actually execute.
FALLBACK_CONFIG = dict(
    n_modes=12,
    photon_list=(2,),
    reservoirs_per_photon=2,
    depth=1,
    leak=0.1,
    g_in=0.1,
    g_fb=0.3,
    encode_window=10,
    window=20,
)


def load_config(dataset: str, use_tuned: bool = False) -> dict:
    """Configuration for the noise sweep.

    Defaults to :data:`FALLBACK_CONFIG` on purpose. The point of this study is what a real
    device would do, and the accuracy-tuned configuration is not runnable on one: it uses a
    large mode count and a wide ensemble, and at three or more photons the coincidence rate
    falls as ``transmittance ** n`` to the point where the shots are unaffordable. Two
    photons in twelve modes is the regime a QPU can actually deliver, so that is what gets
    stressed here.

    ``use_tuned`` loads the accuracy-tuned configuration instead, for the ablation that
    shows the two regimes respond to noise the same way.
    """
    if not use_tuned:
        return dict(FALLBACK_CONFIG)
    path = ROOT / "results" / "tuning" / f"{dataset}_photonic.json"
    if not path.exists():
        return dict(FALLBACK_CONFIG)
    params = json.loads(path.read_text())["params"]
    params.pop("feedback", None)
    params["photon_list"] = tuple(params["photon_list"])
    # Perceval in a recurrent loop costs ~5-40 ms per step per reservoir, so a wide
    # ensemble would make the grid take days.
    params["reservoirs_per_photon"] = min(int(params.get("reservoirs_per_photon", 2)), 2)
    return params


def run_point(u, y, split, config, spec, shots, seed):
    model = TemporalPhotonicQRC(
        washout=split.washout,
        seed=seed,
        noise_spec=spec,
        n_samples=shots,
        threshold_detectors=True,
        **config,
    )
    result = evaluate_features(model.build_features(u, split.n_train), y, split)
    return result["nrmse"]


def _rows(u, y, split, config, seeds, spec, shots, label, **extra):
    out = []
    for seed in seeds:
        score = run_point(u, y, split, config, spec, shots, seed)
        out.append(dict(sweep=label, seed=seed, nrmse=score, shots=shots or 0,
                        spec=spec.name, **extra))
        print(f"  [{label}] {extra} shots={shots or 'inf'} seed={seed} -> {score:.4f}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="narma10")
    ap.add_argument("--sweep", default="all",
                    choices=["all", "shots", "indist", "g2", "hardware", "joint", "rate"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--use-tuned", action="store_true",
                    help="use the accuracy-tuned config instead of the hardware-viable one")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    u, y, split = load_task(args.dataset)
    config = load_config(args.dataset, use_tuned=args.use_tuned)
    seeds = list(range(42, 42 + args.seeds))
    print(f"dataset={args.dataset}  config={config}", flush=True)

    # Reference points: noiseless quantum, and the tuned classical control.
    rows = []
    baseline = run_point(u, y, split, config, None, None, 42)
    control = evaluate_features(lag_features(u, split.n_train, window=40), y, split)["nrmse"]
    print(f"noiseless photonic {baseline:.4f} | classical control {control:.4f}\n", flush=True)
    rows.append(dict(sweep="reference", seed=42, nrmse=baseline, shots=0, spec="noiseless"))
    rows.append(dict(sweep="reference", seed=42, nrmse=control, shots=0, spec="classical_control"))

    want = args.sweep

    if want in ("all", "shots"):
        for shots in [30, 100, 300, 1000, 3000, 10000, 30000, None]:
            rows += _rows(u, y, split, config, seeds, ASCELLA, shots, "shots")

    if want in ("all", "indist"):
        for v in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]:
            spec = HardwareSpec(name=f"V={v:.2f}", indistinguishability=v)
            rows += _rows(u, y, split, config, seeds[:1], spec, None, "indist", indist=v)

    if want in ("all", "g2"):
        for g2 in [0.0, 0.02, 0.05, 0.10, 0.20, 0.30]:
            spec = HardwareSpec(name=f"g2={g2:.2f}", g2=g2)
            rows += _rows(u, y, split, config, seeds[:1], spec, None, "g2", g2=g2)

    if want in ("all", "hardware"):
        for spec in [IDEAL, ASCELLA, BELENOS]:
            for shots in [1000, 10000, None]:
                rows += _rows(u, y, split, config, seeds, spec, shots, "hardware")

    if want in ("all", "joint"):
        for v in [0.7, 0.85, 1.0]:
            spec = HardwareSpec(name=f"V={v:.2f}", indistinguishability=v)
            for shots in [300, 3000, 30000]:
                rows += _rows(u, y, split, config, seeds[:2], spec, shots, "joint", indist=v)

    frame = pd.DataFrame(rows)
    out = RESULTS / f"noise_{args.dataset}_{args.sweep}.csv"
    frame.to_csv(out, index=False)
    print(f"\nwrote {out}", flush=True)

    if want in ("all", "rate"):
        rate_rows = []
        for spec in (ASCELLA, BELENOS):
            for n in (2, 3, 4):
                rate = coincidence_rate(spec, n)
                # The free tier caps a job at 5 minutes; that budget is shared across the
                # timesteps of one sequence.
                per_step = rate * 300.0 / 1000.0
                rate_rows.append(dict(platform=spec.name, n_photons=n,
                                      coincidences_per_s=rate,
                                      shots_per_step_5min_1000steps=per_step))
        rate_frame = pd.DataFrame(rate_rows)
        rate_out = RESULTS / "coincidence_rates.csv"
        rate_frame.to_csv(rate_out, index=False)
        print(rate_frame.to_string(index=False))
        print(f"wrote {rate_out}", flush=True)


if __name__ == "__main__":
    main()
