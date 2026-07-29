# Hardware — Quandela QPU execution

**Status: one real QPU job is queued.** `qpu:belenos`, job
`38587a22-d3a8-4d61-b28e-54611ec5e492`, 120 timesteps at 20 000 shots, 2 photons in 10 modes.
Harvest with `python hardware/fetch_qpu_run.py`.

**Do not trust the platform status field.** Both QPUs report `status: maintenance` while the API
accepts submissions and queues them. Attempting a submission is the only reliable test. Jobs
then sit in `waiting` for a long time (2 h 16 m for the 2026-07-07 smoke test), so the real
constraint is queue latency.

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
| QPU submit | `python hardware/submit_qpu_run.py --platform qpu:belenos --steps 120 --shots 20000` | one job |
| QPU harvest | `python hardware/fetch_qpu_run.py` | free |

Submit and harvest are separate because the queue runs to hours and a laptop will not stay
awake for it. Size the run from the *measured* coincidence rate, not the published
transmittance — the two differ substantially.

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

- **Platform names**: `qpu:ascella`, `qpu:belenos`, `sim:slos`, plus the emulators
  `sim:ascella` and `sim:belenos`. Check status programmatically —
  `RemoteProcessor(name, token)._rpc_handler.fetch_platform_details()` returns `status` and
  `bookable`. `sim:clifford` now reports `decommissioned`, so the field is maintained.
- **The emulators do apply a small device model.** Judged against the distribution a perfect
  sampler would give at the same shot count, `sim:slos` sits at +5 sigma and `sim:ascella` at
  +26 sigma, so the emulator adds noise beyond sampling -- 0.008 of total variation on top of
  0.019 of sampling error. Do **not** infer this from downstream NRMSE: `sim:ascella` scored
  better than the noiseless `sim:slos` on a 600-step run, which is seed scatter in a
  regression, not a statement about the distributions. `scripts/compare_platforms.py` does it
  properly, against a calibrated null.
- **The free tier allows one waiting job at a time.** A second submission returns HTTP 400 with
  no useful message. Queue Ascella only after the Belenos job clears.
- **Use `submit_qpu_run.py` + `fetch_qpu_run.py` for QPU work, not `run_reservoir_hw.py`.** The
  latter submits a chunk, polls, then submits the next, which loses everything if the machine
  sleeps during a multi-hour queue. The former submits one job sized to the five-minute cap,
  stores what is needed to rebuild the readout, and exits.
- **`sim:belenos` is roughly 20× slower than `sim:ascella`.** A 50-step chunk reached only
  20 % after 206 s, extrapolating to hours per run. Budget for it or avoid it; a job that
  overruns blocks the queue, since the free tier allows one at a time. Cancel with
  `POST /api/job/cancel/<id>`.
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
