# Hardware — Quandela QPU execution

**Status: no result in this repository is a QPU measurement.** Both `qpu:ascella` and
`qpu:belenos` have reported `status: maintenance` throughout the current work. The code here
is implemented and rehearsed locally and against the cloud simulator.

Note that every earlier run in this directory used **Belenos**, not Ascella, despite what
older versions of these docs said. Ascella is the preferred target when it returns: 80 MHz
against 4.94 MHz, and g² of 1.95 % against 18.2 %.

## 1. Account & token

Resolution order in `hw_backend.get_token()`:

```bash
export PCVL_CLOUD_TOKEN=...        # perceval's native variable, preferred
export QUANDELA_TOKEN=...          # fallback
# or save once on this machine:
python -c "import perceval as pcvl; pcvl.save_token('...')"
```

Free tier: 200 credits/month, **5 minutes per job**, one queued job at a time, low priority,
bookable free QPU slots. Shots are the billing unit — a shot is any detected event containing
at least one photon.

## 2. Environment

```bash
conda activate quandela      # perceval-quandela 1.1.0, merlinquantum 0.3.1
```

The conventions in section 5 were verified against those exact versions.

## 3. Run order

| Step | Command | Cost |
|---|---|---|
| Core correctness | `python -m pytest tests/ -q` | free |
| Zero-cost gate | `python hardware/compare_sim_local.py` | free |
| Local rehearsal | `python hardware/run_reservoir_hw.py --local --steps 600` | free |
| Cloud rehearsal | `python hardware/run_reservoir_hw.py --platform sim:slos --steps 100` | simulator |
| Rate probe | `python hardware/run_reservoir_hw.py --platform qpu:ascella --steps 10 --shots 5000` | small |
| Full run | `python hardware/run_reservoir_hw.py --platform qpu:ascella --steps 600 --shots 30000` | see PLAN.md §3 |

Always run the probe before the full run: the budget in `PLAN.md` assumes the published
transmittance, and the probe measures the real coincidence rate.

## 4. Files

- `run_reservoir_hw.py` — **the current runner.** Executes the recurrent model from
  `src/temporal_qrc.py` and reports hardware, simulation and classical-only readouts on
  identical timesteps.
- `hw_backend.py` — token resolution and `RemoteProcessor` construction with a raised RPC
  read timeout (perceval's 10 s default is exceeded by the cloud API under load, both while
  polling and inside the constructor).
- `hw_features.py` — chunked, cached job submission; histogramming into the unbunched
  subspace.
- `compare_sim_local.py` — exact-probability and sampling-convergence gate.
- `run_hw_subset.py`, `run_hw_native.py`, `run_probe.py`, `phase_export.py`,
  `smoke_test.py`, `compare_sim_hw.py` — the earlier pipeline, built around
  `src/multi_qrc.py`. Kept so the Belenos probe reproduces; not the current path.
- `cache/`, `results/`, `run_log.csv` — job cache and measurements.

## 5. Verified conventions (don't rediscover these)

Established against perceval 1.1.0 / merlin 0.3.1, checked by `compare_sim_local.py` and
`tests/test_photonic_core.py`:

- merlin's flat `t` parameter tensor is ordered exactly like `circuit.get_parameters()`
  filtered to `t*` names.
- merlin applies input values **directly as phases in radians** (scale 1.0 — no π or 2π
  factor).
- `QuantumLayer.output_keys` is the unbunched Fock-state order of the output probability
  vector; the hardware histogram uses the same order.
- A trailing phase shifter on an output mode has no effect on probabilities — mid-circuit
  placement is what makes the encoding act.
- The amplitude for output pattern `S` is `perm(U[S, T])` with `T` the input modes: rows
  indexed by occupied **output** modes, columns by input modes. `U[S, T]` and `U[T, S]` are
  not transposes of each other unless `U` is symmetric. Getting this backwards produces a
  plausible-looking but wrong distribution; it cost a debugging session.

## 6. Gotchas

- **Platform names**: `qpu:ascella`, `qpu:belenos`, `sim:slos`. Check status
  programmatically — `RemoteProcessor(name, token)._rpc_handler.fetch_platform_details()`
  returns `status` and `bookable`.
- **The 5-minute cap is real and silent.** Job `96baa2b6-…` was cancelled at 307 s having
  completed 4 % of its iterations. `sample_batch(chunk_size=...)` splits the batch into
  independently-cached jobs, and a job returning fewer iterations than requested now raises
  instead of caching a truncated result.
- **`max_shots_per_call` must be large.** It defaulted to `10 × shots`, which starves the
  post-selection that follows. Use ~1e7.
- **Use two photons.** n-fold coincidence rate falls as `transmittance^n`. On Ascella that is
  4.8e4/s at n=2 against 1.2e3/s at n=3. The earlier three-photon probe returned ~48 counts
  from 1000 requested shots across 56 Fock bins, and its features correlated only 0.18–0.48
  with simulation. That was sampling noise, not a calibration fault.
- **Post-selection**: `min_detected_photons_filter(n_photons)` keeps only coincidence events;
  loss shows up as a low `unbunched_counts / shots_requested` ratio.
- **Caching**: results are keyed by (platform, circuit unitary, inputs, shots). Delete
  `hardware/cache/` only if you intentionally want to re-spend shots.
- **Closed-loop feedback cannot be batched.** Step *t*'s phases depend on step *t−1*'s
  measurement, so the two protocols in `run_reservoir_hw.py` exist: `replay` (feedback
  simulated, trajectory replayed on chip) and `openloop` (fully on-device, weaker model).
  Report both and describe the `replay` caveat explicitly.
