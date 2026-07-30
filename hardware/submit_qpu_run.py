"""Submit one self-contained QPU job and return immediately.

Motivated by a hard constraint: Quandela's QPU queue is long (the 2026-07-07 smoke test waited
2 h 16 m) and a laptop cannot stay awake for it. The normal driver
(``run_reservoir_hw.py``) submits a chunk, polls until it returns, then submits the next --
which loses everything if the machine sleeps.

This instead submits **one** job, sized to fit inside the tier's five-minute execution cap,
writes the job id to disk, and exits. ``fetch_qpu_run.py`` recovers the result later, on any
machine, whenever the job finishes.

Note that ``fetch_platform_details`` reports ``maintenance`` for both QPUs while the API still
accepts submissions, so platform status is not a reliable availability signal --- attempting a
submission is.

Sizing. At two photons the coincidence rate is ``clock * transmittance^2``: about
1.2e4/s on Belenos. A five-minute execution window therefore yields roughly
``300 * 1.2e4 = 3.6e6`` coincidences, which at ``shots`` per timestep supports
``3.6e6 / shots`` timesteps. Defaults leave headroom because the published transmittance is
optimistic relative to the rate we actually measured.

Usage::

    python hardware/submit_qpu_run.py --platform qpu:belenos --steps 120 --shots 20000
    # ... later, from anywhere ...
    python hardware/fetch_qpu_run.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "hardware"))

from src.temporal_qrc import TemporalPhotonicQRC  # noqa: E402

PENDING = ROOT / "hardware" / "pending_qpu_runs.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", default="qpu:belenos")
    ap.add_argument("--dataset", default="narma10")
    ap.add_argument("--steps", type=int, default=120,
                    help="total timesteps in the trajectory")
    ap.add_argument("--slice-start", type=int, default=0,
                    help="first timestep this job covers")
    ap.add_argument("--slice-len", type=int, default=None,
                    help="timesteps in this job; must fit the 5-minute execution cap. "
                         "Measured at ~16 s/timestep at 20000 shots on Belenos, so ~18.")
    ap.add_argument("--shots", type=int, default=20_000)
    ap.add_argument("--max-shots-per-call", type=int, default=10_000_000)
    ap.add_argument("--modes", type=int, default=10)
    ap.add_argument("--photons", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import perceval as pcvl
    from perceval.algorithm import Sampler

    from hw_backend import get_processor
    from src.tasks import load_task

    u, y, split = load_task(args.dataset)
    u, y = u[: args.steps], y[: args.steps]

    # One reservoir only: a second would need a second job, defeating the point.
    model = TemporalPhotonicQRC(
        n_modes=args.modes, photon_list=(args.photons,), reservoirs_per_photon=1,
        depth=1, leak=0.1, g_in=0.1, g_fb=0.3, encode_window=10, window=20,
        washout=min(20, args.steps // 5), seed=args.seed,
    )
    from run_reservoir_hw import phase_trajectory

    trajectories, sim_states = phase_trajectory(model, u, int(0.7 * args.steps))
    # The trajectory is computed over the whole series so slices concatenate into one
    # continuous run; each job then submits only its own window.
    lo = args.slice_start
    hi = min(args.steps, lo + (args.slice_len or args.steps))
    phases = trajectories[0][lo:hi]
    reservoir = model.reservoirs[0]
    circuit, input_state = reservoir.optics.to_perceval()

    processor = get_processor(args.platform, m=args.modes)
    processor.set_circuit(circuit)
    processor.with_input(pcvl.BasicState(input_state))
    processor.min_detected_photons_filter(args.photons)

    sampler = Sampler(processor, max_shots_per_call=args.max_shots_per_call)
    sampler.add_iteration_list([
        {"circuit_params": {p.name: float(v)
                            for p, v in zip(circuit.get_parameters(), row)}}
        for row in phases
    ])
    job = sampler.sample_count.execute_async(args.shots)

    record = {
        "job_id": job.id,
        "platform": args.platform,
        "dataset": args.dataset,
        "steps": int(hi - lo),
        "slice_start": int(lo),
        "slice_end": int(hi),
        "total_steps": int(args.steps),
        "shots": int(args.shots),
        "modes": int(args.modes),
        "photons": int(args.photons),
        "seed": int(args.seed),
        "submitted_utc": datetime.now(timezone.utc).isoformat(),
        # Everything needed to rebuild the readout without re-deriving the trajectory.
        "output_keys": [[int(b) for b in
                         (1 if m in pat else 0 for m in range(args.modes))]
                        for pat in reservoir.optics.patterns],
        "sim_states": np.asarray(sim_states[0])[lo:hi].tolist(),
        "targets": np.asarray(y).ravel()[lo:hi].tolist(),
        "input_scaled": model.input_scaler_.transform(u)[lo:hi].tolist(),
        "leak": reservoir.leak,
    }
    pending = json.loads(PENDING.read_text()) if PENDING.exists() else []
    pending.append(record)
    PENDING.write_text(json.dumps(pending, indent=2))

    print(f"submitted to {args.platform}")
    print(f"  job id     : {job.id}")
    print(f"  timesteps  : {hi - lo}  (slice {lo}:{hi} of {args.steps})")
    print(f"  shots/step : {args.shots}")
    print(f"  config     : {args.photons} photons, {args.modes} modes, 1 reservoir")
    print(f"\nrecorded in {PENDING.name}. Safe to close the machine.")
    print("Recover with:  python hardware/fetch_qpu_run.py")


if __name__ == "__main__":
    main()
