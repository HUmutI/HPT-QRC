# Hardware — Quandela Ascella QPU

Code for running HPT-QRC on real photonic hardware via Quandela Cloud.
Strategy and budget math live in [PLAN.md](PLAN.md). Simulation code in `src/` is untouched;
this folder only replaces the feature-extraction step (exact p-vector → sampled histogram).

## 1. Account & token

1. Create an account at <https://cloud.quandela.com> (apply for the **Academic** tier — see PLAN.md §5).
2. Generate an access token in the web console (account → tokens).
3. Make it available to perceval (pick one):

```bash
export PCVL_CLOUD_TOKEN='<your-token>'          # per shell / .zshrc
# or save once on this machine:
python -c "import perceval as pcvl; pcvl.save_token('<your-token>')"
```

Never commit the token. `QUANDELA_TOKEN` also works (our fallback in `hw_backend.py`).

## 2. Environment

Use the existing conda env (has perceval 1.1.0 + merlin 0.3.1):

```bash
conda activate quandela
```

Test the connection (no shots consumed — just fetches platform specs):

```bash
python hardware/hw_backend.py
```

## 3. Run order

| Step | Command | Cost |
|---|---|---|
| 1. Offline sanity | `python hardware/compare_sim_local.py` | free (local) |
| 2. Cloud rehearsal | `python hardware/run_probe.py --platform sim:slos` | ~free (cloud simulator) |
| 3. Cost probe | `python hardware/run_probe.py` | ~1e4 shots on Ascella |
| 4. Validation subset | `run_hw_subset.py` (build after probe) | decided from probe numbers |

Do not skip step 1: it verifies phase export, state ordering, and input-encoding
conventions against the merlin simulation with zero cloud cost. If it fails,
a hardware run would produce garbage silently.

Record your dashboard credit balance before/after step 3 and note it in
`run_log.csv` (`credits_note` column).

## 4. Files

- `hw_backend.py` — cloud connection + token handling (`get_processor("qpu:ascella")`)
- `phase_export.py` — extracts the fixed random `t_*` phases from the simulated
  `QuantumLayer` so the identical circuit runs on hardware
- `hw_features.py` — sampling → unbunched histogram → `LexGrouping` features;
  every job cached in `cache/` so re-runs never re-bill shots
- `compare_sim_local.py` — offline equivalence check (merlin vs perceval SLOS)
- `run_probe.py` — the §4 cost probe; appends to `run_log.csv`
- `run_hw_subset.py` — (to build after probe) validation-subset driver
- `compare_sim_hw.py` — (to build after probe) sim-vs-HW paper figure

## 5. Verified conventions (don't rediscover these)

Established against perceval 1.1.0 / merlin 0.3.1, checked by `compare_sim_local.py`:

- merlin's flat `t` parameter tensor is ordered exactly like
  `circuit.get_parameters()` filtered to `t*` names.
- merlin applies input values **directly as phases in radians** (scale 1.0 —
  no π or 2π factor). Encoded windows are in [0,1] from the robust scaler,
  so phases span only [0,1] rad by design.
- `QuantumLayer.output_keys` is the unbunched Fock-state order of the output
  probability vector; the hardware histogram uses the same order.
- A trailing phase shifter on an output mode has no effect on probabilities —
  input PS placement mid-circuit is what makes the encoding act.

## 6. Gotchas

- **Platform names**: `qpu:ascella` for hardware, `sim:slos` for the cloud
  simulator. Check availability in the dashboard first — Ascella has maintenance
  windows, and 2026 tier/quota numbers must be verified (PLAN.md §5).
- **Post-selection**: `min_detected_photons_filter(n_photons)` keeps only
  coincidence events; loss shows up as a low `unbunched_counts / shots_requested`
  ratio in the log, which is exactly what the probe measures.
- **All timesteps in one job**: `hw_features.sample_batch` uses Sampler
  iterations (`circuit_params` per step), so a 10-step probe is 1 queued job,
  not 10.
- **Caching**: job results are keyed by (platform, phases, inputs, shots). Delete
  `hardware/cache/` only if you intentionally want to re-spend shots.
