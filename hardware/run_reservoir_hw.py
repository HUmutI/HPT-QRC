"""Run the recurrent photonic reservoir on a Quandela QPU.

There is a structural obstacle to putting a *closed-loop* reservoir on a cloud QPU: the
phases at step t depend on the measurement at step t-1, so the timesteps cannot be batched
into one job. A thousand sequential cloud submissions is not viable on a queue measured in
hours, so two protocols are provided and both are reported:

``replay`` (default)
    The recurrence is run in simulation to produce the phase trajectory, then the whole
    trajectory is replayed on hardware as one batched job. The chip evaluates the same
    circuit settings the closed-loop system would have visited, and the readout is trained
    on the hardware-measured features. This is an open-loop replay of a closed-loop
    trajectory -- it measures the device's feature map faithfully, but the feedback path
    itself was simulated, and the paper must say so.

``openloop``
    ``feedback=False``. Every timestep is independent, so this is genuinely end-to-end on
    hardware with no simulation in the loop. It is the weaker model but the stronger claim.

Three readouts are fitted on identical timesteps -- hardware features, simulated features,
and a classical-only control -- so the comparison isolates what the device contributes.

Design choices forced by the hardware, and why:

* **Two photons, not three.** The n-fold coincidence rate falls as ``transmittance ** n``.
  At Ascella's 2.44% that is 1.5e-5 for three photons against 6.0e-4 for two: a factor of
  41. The earlier three-photon probe returned ~48 counts from 1000 requested shots spread
  over 56 Fock bins, which is why its features correlated only 0.18-0.48 with simulation.
* **Shallow circuits.** Loss compounds with depth, and depth bought no accuracy in
  simulation.
* **A real shot budget.** ``hw_features.sample_batch`` defaulted ``max_shots_per_call`` to
  ``10 * shots``, which starves the very post-selection it then performs.
* **Chunking.** The free tier cancels any job at 5 minutes. Job
  96baa2b6-da73-41c8-aa12-beb9cd39650a was killed at 307 s having completed 4% of its
  iterations, which is what motivated the chunked submission path.

Usage::

    python hardware/run_reservoir_hw.py --local --steps 200          # free, local check
    python hardware/run_reservoir_hw.py --platform sim:slos --steps 200   # cloud rehearsal
    python hardware/run_reservoir_hw.py --platform qpu:belenos --steps 200
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

from src.rc_protocol import Split, evaluate_features, nrmse  # noqa: E402
from src.tasks import load_task  # noqa: E402
from src.temporal_qrc import TemporalPhotonicQRC  # noqa: E402

RESULTS = ROOT / "hardware" / "results"


def phase_trajectory(model: TemporalPhotonicQRC, u: np.ndarray, n_train: int):
    """Simulated phase trajectory per reservoir, plus the simulated states.

    Returns ``(list_of_(T, n_enc) arrays, list_of_(T, P) state arrays)``.
    """
    model._build(u.shape[1])
    from sklearn.preprocessing import StandardScaler

    model.input_scaler_ = StandardScaler().fit(u[:n_train])
    scaled = model.input_scaler_.transform(u)
    drive = model._lags(scaled, model.encode_window) if model.encode_window > 1 else scaled

    trajectories, states = [], []
    for reservoir in model.reservoirs:
        base = (reservoir.w_in @ drive.T).T * reservoir.g_in + reservoir.bias
        state = np.full(reservoir.state_size, 1.0 / reservoir.state_size)
        size = reservoir.state_size
        phases = np.empty((len(drive), reservoir.optics.n_encoding))
        seq = np.empty((len(drive), size))
        for t in range(len(drive)):
            phases[t] = base[t] + reservoir.g_fb * (reservoir.w_fb @ (size * state - 1.0))
            p = reservoir.optics.probabilities(phases[t])
            state = (1.0 - reservoir.leak) * state + reservoir.leak * p
            seq[t] = state
        trajectories.append(phases)
        states.append(seq)
    return trajectories, states


def run_on_platform(reservoir, phases, platform, shots, chunk, max_shots_per_call, local):
    """Sample the reservoir's outputs at the given phase settings.

    Returns ``(probabilities (T, P), metadata)``. Uses the cached, chunked submission path
    in ``hw_features`` so an interrupted run resumes without re-billing shots.
    """
    import perceval as pcvl

    from hw_backend import get_processor
    from hw_features import sample_batch

    optics = reservoir.optics
    circuit, input_state = optics.to_perceval()
    spec = {
        "n_modes": optics.n_modes,
        "photons": optics.n_photons,
        "input_state": input_state,
        "output_keys": [
            tuple(1 if m in pat else 0 for m in range(optics.n_modes)) for pat in optics.patterns
        ],
        "input_names": [p.name for p in circuit.get_parameters()],
        "circuit": circuit,
    }

    if local:
        processor = pcvl.Processor("SLOS", circuit)
        platform_name = "local:slos"
    else:
        processor = get_processor(platform, m=optics.n_modes)
        platform_name = platform

    out = sample_batch(
        processor,
        spec,
        phases,
        shots,
        platform_name,
        max_shots_per_call=max_shots_per_call,
        chunk_size=None if local else chunk,
    )
    probs = np.array([step["p_hat"] for step in out["steps"]])
    meta = {
        "wall_time_s": out.get("wall_time_s"),
        "drop_rate": [step.get("drop_rate") for step in out["steps"]],
        "raw_counts": [step.get("raw_counts") for step in out["steps"]],
    }
    return probs, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", default="qpu:belenos")
    ap.add_argument("--local", action="store_true", help="run locally, no cloud, no cost")
    ap.add_argument("--dataset", default="narma10")
    ap.add_argument("--protocol", default="replay", choices=["replay", "openloop"])
    ap.add_argument("--steps", type=int, default=600, help="timesteps to run on hardware")
    ap.add_argument("--shots", type=int, default=5000, help="coincidences requested per step")
    ap.add_argument("--max-shots-per-call", type=int, default=10_000_000)
    ap.add_argument("--chunk", type=int, default=25, help="steps per job (5-minute tier cap)")
    ap.add_argument("--modes", type=int, default=12)
    ap.add_argument("--photons", type=int, default=2)
    ap.add_argument("--reservoirs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    u, y, split = load_task(args.dataset)

    steps = min(args.steps, len(u))
    u, y = u[:steps], y[:steps]
    n_train = int(0.7 * steps)
    split = Split(steps, washout=min(50, n_train // 4), n_train=n_train,
                  n_val=max(10, n_train // 5))

    model = TemporalPhotonicQRC(
        n_modes=args.modes,
        photon_list=(args.photons,),
        reservoirs_per_photon=args.reservoirs,
        depth=1,
        leak=0.1,
        g_in=0.1,
        g_fb=0.3 if args.protocol == "replay" else 0.0,
        feedback=args.protocol == "replay",
        encode_window=10,
        window=20,
        washout=split.washout,
        seed=args.seed,
    )

    print(f"protocol={args.protocol} platform={'local:slos' if args.local else args.platform} "
          f"modes={args.modes} photons={args.photons} steps={steps} shots={args.shots}",
          flush=True)

    trajectories, sim_states = phase_trajectory(model, u, n_train)

    started = time.time()
    hw_blocks, metas = [], []
    for idx, (reservoir, phases) in enumerate(zip(model.reservoirs, trajectories)):
        print(f"[reservoir {idx + 1}/{len(model.reservoirs)}]", flush=True)
        probs, meta = run_on_platform(reservoir, phases, args.platform, args.shots,
                                      args.chunk, args.max_shots_per_call, args.local)
        hw_blocks.append(probs)
        metas.append(meta)

    # Hardware states: replay the same leaky integration over measured probabilities, so
    # the only difference from the simulated states is where the probabilities came from.
    hw_states = []
    for reservoir, probs in zip(model.reservoirs, hw_blocks):
        state = np.full(reservoir.state_size, 1.0 / reservoir.state_size)
        seq = np.empty_like(probs)
        for t in range(len(probs)):
            state = (1.0 - reservoir.leak) * state + reservoir.leak * probs[t]
            seq[t] = state
        hw_states.append(seq)

    classical = model._lag_block(model.input_scaler_.transform(u))
    features = {
        "hardware": np.hstack(hw_states + [classical]),
        "simulation": np.hstack(sim_states + [classical]),
        "classical_only": classical,
    }
    scores = {k: evaluate_features(v, y, split)["nrmse"] for k, v in features.items()}

    correlations = [
        float(np.corrcoef(a[t], b[t])[0, 1])
        for a, b in zip(hw_states, sim_states)
        for t in range(len(a))
    ]

    report = {
        "platform": "local:slos" if args.local else args.platform,
        "protocol": args.protocol,
        "dataset": args.dataset,
        "steps": steps,
        "shots": args.shots,
        "modes": args.modes,
        "photons": args.photons,
        "reservoirs": len(model.reservoirs),
        "nrmse": scores,
        "quantum_lift_vs_classical": scores["classical_only"] / scores["hardware"],
        "hardware_vs_simulation_ratio": scores["hardware"] / scores["simulation"],
        "feature_correlation_mean": float(np.mean(correlations)),
        "feature_correlation_min": float(np.min(correlations)),
        "wall_time_s": time.time() - started,
        "meta": metas,
    }

    print("\n=== NRMSE ===")
    for name, value in sorted(scores.items(), key=lambda kv: kv[1]):
        print(f"  {name:<16} {value:.4f}")
    print(f"\nhardware/simulation feature correlation: mean "
          f"{report['feature_correlation_mean']:.3f}, min {report['feature_correlation_min']:.3f}")
    print(f"quantum lift (classical_only / hardware): "
          f"{report['quantum_lift_vs_classical']:.3f}  (>1 means the reservoir helps)")

    tag = report["platform"].replace(":", "_")
    out = RESULTS / f"reservoir_hw_{tag}_{args.protocol}_{args.dataset}.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
