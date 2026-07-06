"""Export the fixed reservoir phases from a simulated HPT_QRC_Multi model.

The `t_*` phases are seeded-random and never trained (`layer.eval()`), so the
hardware circuit must be programmed with the *exact same values* the simulation
used — they are extracted from the merlin QuantumLayer, not re-sampled.

Verified conventions (see compare_sim_local.py):
  - merlin stores all `t` phases as one flat tensor whose order matches
    `circuit.get_parameters()` filtered to names starting with "t".
  - merlin applies input values directly as phases in radians (scale 1.0).
  - `QuantumLayer.output_keys` is the unbunched Fock-state ordering of the
    output probability vector.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.multi_qrc import HPT_QRC_Multi  # noqa: E402


def rebuild_circuit(spec: dict):
    """Rebuild the parametric perceval circuit described by a reservoir spec.

    Reuses the exact builder from the simulation so the topology is identical.
    Returns (circuit, n_encoding_params).
    """
    return HPT_QRC_Multi._build_temporal_circuit(
        None,
        spec["n_modes"],
        spec["n_input_modes"],
        spec["window"],
        spec["virtual_depth"],
    )


def export_reservoir(model: HPT_QRC_Multi, k: int) -> dict:
    """Export reservoir k of a built model as a self-contained hardware spec.

    Reservoir ordering in model.reservoirs is:
        for r_idx, n_ph in enumerate(photon_list):
            for vd in 1..n_virtual_nodes
    """
    layer = model.reservoirs[k]
    core = layer[0]  # nn.Sequential(QuantumLayer, LexGrouping)

    r_idx = k // model.n_virtual_nodes
    vd = k % model.n_virtual_nodes + 1
    n_ph = model.photon_list[r_idx]

    spec = {
        "reservoir_index": k,
        "photons": n_ph,
        "virtual_depth": vd,
        "n_modes": model.n_modes,
        "n_input_modes": model.n_input_modes,
        "window": model.window,
        "input_state": [1] * n_ph + [0] * (model.n_modes - n_ph),
        "lex_out": model.lex_out,
    }

    circ, n_enc = rebuild_circuit(spec)
    param_names = [p.name for p in circ.get_parameters()]
    t_names = [n for n in param_names if n.startswith("t")]
    input_names = [n for n in param_names if n.startswith("input")]

    t_vals = core.state_dict()["t"].detach().cpu().numpy().tolist()
    if len(t_names) != len(t_vals):
        raise RuntimeError(
            f"Phase count mismatch for reservoir {k}: circuit has {len(t_names)} "
            f"t-params but layer stores {len(t_vals)} values. Model args must "
            "match the ones used to build the exported model."
        )

    spec["n_enc"] = n_enc
    spec["t_phases"] = dict(zip(t_names, t_vals))
    spec["input_names"] = input_names
    spec["output_keys"] = [list(s) for s in core.output_keys]
    return spec


if __name__ == "__main__":
    import json

    # Probe configuration: single [3]-photon reservoir, virtual depth 1.
    model = HPT_QRC_Multi(window=5, photon_list=[3], n_virtual_nodes=1, seed=42)
    spec = export_reservoir(model, 0)
    out = Path(__file__).parent / "reservoir_spec_probe.json"
    out.write_text(json.dumps(spec, indent=2))
    print(f"Wrote {out}")
    print(f"  modes={spec['n_modes']} photons={spec['photons']} "
          f"t-params={len(spec['t_phases'])} inputs={len(spec['input_names'])} "
          f"output dim={len(spec['output_keys'])}")
