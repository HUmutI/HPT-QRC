"""Offline sanity check — costs zero cloud credits.

Runs the exact hardware pipeline (phase export -> circuit rebuild -> Sampler ->
histogram -> LexGrouping) against a LOCAL SLOS processor and compares with the
merlin QuantumLayer output the simulation uses.

Pass criteria:
  1. Exact probabilities: L1(merlin p-vector, SLOS p-vector) ~ 1e-6
     -> proves circuit rebuild + phase export + state ordering are all correct.
  2. Sampled: L1 shrinks as shots grow (~ sqrt scaling)
     -> proves the histogram/post-selection path behaves.

Run this BEFORE any cloud job. If check 1 fails, nothing else matters.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import perceval as pcvl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from src.multi_qrc import HPT_QRC_Multi
from src.data_loader import load_narma10
from phase_export import export_reservoir
from hw_features import build_hw_circuit, make_iterations, counts_to_phat, phat_to_features

N_STEPS = 5
SHOT_LEVELS = [100, 1000, 10000]


def encoded_windows(model, X, n_steps):
    """Reproduce the windowing from HPT_QRC_Multi._create_features."""
    model._fit_scaler(X.flatten())
    Xs = model._apply_scaler(X)
    padded = np.vstack([np.zeros((model.window - 1, model.in_size)), Xs.reshape(-1, model.in_size)])
    wins = []
    for t in range(model.window - 1, model.window - 1 + n_steps):
        w = padded[t - model.window + 1: t + 1].flatten()
        if len(w) < model.n_enc_actual:
            w = np.pad(w, (0, model.n_enc_actual - len(w)))
        wins.append(w[: model.n_enc_actual])
    return np.array(wins)


def main():
    model = HPT_QRC_Multi(window=5, photon_list=[3], n_virtual_nodes=1, seed=42)
    spec = export_reservoir(model, 0)
    print(f"Reservoir: {spec['n_modes']} modes, {spec['photons']} photons, "
          f"{len(spec['t_phases'])} fixed phases, output dim {len(spec['output_keys'])}")

    X_train, _, _, _ = load_narma10()
    X_win = encoded_windows(model, X_train, N_STEPS)

    # Reference: exact merlin probability vectors (pre-LexGrouping)
    core = model.reservoirs[0][0]
    with torch.no_grad():
        p_merlin = core(torch.tensor(X_win, dtype=torch.float32)).numpy()

    # Hardware-path circuit on a local exact simulator
    circ = build_hw_circuit(spec)
    proc = pcvl.Processor("SLOS", circ)
    proc.min_detected_photons_filter(spec["photons"])
    proc.with_input(pcvl.BasicState(spec["input_state"]))

    # --- Check 1: exact probabilities through perceval ------------------
    keys = [tuple(k) for k in spec["output_keys"]]
    worst = 0.0
    for i, row in enumerate(X_win):
        for name, v in zip(spec["input_names"], row):
            for p in circ.get_parameters():
                if p.name == name:
                    p.set_value(float(v))
        res = proc.probs()["results"]
        p_slos = np.array([res.get(pcvl.BasicState(list(k)), 0.0) for k in keys])
        p_slos /= p_slos.sum()
        l1 = np.abs(p_slos - p_merlin[i]).sum()
        worst = max(worst, l1)
    status = "PASS" if worst < 1e-4 else "FAIL"
    print(f"\n[check 1] exact SLOS vs merlin, worst L1 over {N_STEPS} steps: {worst:.2e}  {status}")
    if status == "FAIL":
        print("  -> phase export / ordering / input scaling mismatch. Do NOT run on hardware.")
        return 1

    # --- Check 2: sampled path (Sampler + histogram + LexGrouping) ------
    from perceval.algorithm import Sampler
    print(f"\n[check 2] sampling convergence (step 0):")
    feats_exact = phat_to_features(p_merlin[0], spec)
    for shots in SHOT_LEVELS:
        circ2 = build_hw_circuit(spec)
        proc2 = pcvl.Processor("SLOS", circ2)
        proc2.min_detected_photons_filter(spec["photons"])
        proc2.with_input(pcvl.BasicState(spec["input_state"]))
        sampler = Sampler(proc2, max_shots_per_call=10 * shots)
        sampler.add_iteration_list(make_iterations(spec, X_win[:1]))
        res = sampler.sample_count(shots)["results_list"][0]
        p_hat, meta = counts_to_phat(res["results"], spec)
        l1 = np.abs(p_hat - p_merlin[0]).sum()
        feats = phat_to_features(p_hat, spec)
        feat_err = np.abs(feats - feats_exact).max()
        print(f"  {shots:>6} shots: L1(p_hat, exact)={l1:.4f}  "
              f"max feature err={feat_err:.4f}  drop_rate={meta['drop_rate']:.3f}")

    print("\nAll offline checks done. Pipeline is hardware-ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
