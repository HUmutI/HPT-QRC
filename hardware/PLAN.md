# Hardware execution plan — Quandela QPU

**Status: implemented and rehearsed; blocked on platform availability.**
Both `qpu:ascella` and `qpu:belenos` have reported `status: maintenance` throughout this
work. Nothing in `results/` is a QPU measurement.

This file replaces the original Ascella-only plan, which was written before any hardware
code existed and before the coincidence-rate problem was measured.

## 1. What the earlier attempt taught us

A three-photon, eight-mode configuration was run on **Belenos** (not Ascella, despite what
the older docs said):

| Observation | Value |
|---|---|
| Requested shots per timestep | 1 000 |
| Raw counts returned | 39–63 |
| Counts surviving unbunched post-selection | 21–36, spread over 56 Fock bins |
| `physical_perf` | 3.9e-5 – 6.2e-5 |
| Correlation of hardware features vs simulation | **0.18 – 0.48** |

The features were dominated by sampling noise. This is not a calibration problem — it is the
`transmittance^n` scaling of n-fold coincidences, and the fix is fewer photons, not more
shots.

A second job (`96baa2b6-da73-41c8-aa12-beb9cd39650a`) was **cancelled by the platform at
307 s having completed 4 %** of its iterations: the free tier's 5-minute per-job cap.

## 2. Design consequences

| Choice | Reason |
|---|---|
| **2 photons**, not 3 | Coincidence rate rises by `1/transmittance` ≈ 40×. Ascella: 4.8e4/s at n=2 vs 1.2e3/s at n=3. |
| **Ascella preferred** over Belenos | 80 MHz vs 4.94 MHz clock; g² 1.95 % vs 18.2 %. |
| 10–12 modes | Enough Fock bins for a useful state, few enough for good per-bin statistics. |
| depth 1 | Loss compounds with depth, and depth bought no accuracy in simulation. |
| `max_shots_per_call` ≈ 1e7 | The old default of `10 × shots` starved the post-selection it then performed. |
| Chunked submission | The 5-minute cap. Each chunk caches independently so an interrupted run resumes. |

The noise study (`experiments/noise_study.py`) shows indistinguishability and g² barely
matter for this model, while shots dominate — so the budget should be spent entirely on
coincidence rate.

## 3. Shot budget

Target 3×10⁴ coincidences per timestep, where simulation shows the reservoir still beats its
classical control.

| Platform | n | Rate | Time per timestep | 600 timesteps |
|---|---|---|---|---|
| Ascella | 2 | 4.8e4/s | 0.63 s | ~6 min |
| Belenos | 2 | 1.2e4/s | 2.6 s | ~26 min |
| Ascella | 3 | 1.2e3/s | 26 s | ~4.3 h |

At two photons this fits comfortably inside a booked free slot. At three it does not.

## 4. Protocols

Closed-loop feedback cannot be batched into one cloud job, since step *t*'s phases depend on
step *t−1*'s measurement. Both of these are run and both are reported:

- **`replay`** — the recurrence runs in simulation to produce the phase trajectory, which is
  replayed on hardware as one batched job; the readout is trained on hardware-measured
  features. The chip's feature map is measured faithfully, but **the feedback path is
  simulated** and must be described that way.
- **`openloop`** — feedback disabled; every timestep independent, so the run is genuinely
  end-to-end on hardware. Weaker model, stronger claim.

## 5. Run order

```bash
python -m pytest tests/ -q                                   # core must match MerLin
python hardware/compare_sim_local.py                         # zero-cost gate
python hardware/run_reservoir_hw.py --local --steps 600       # free rehearsal
python hardware/run_reservoir_hw.py --platform sim:slos --steps 100   # cloud rehearsal
python hardware/run_reservoir_hw.py --platform qpu:ascella --steps 600 --shots 30000
```

Start with a 10-step probe on the QPU to measure the actual coincidence rate before
committing the full run; the extrapolation in section 3 assumes the published transmittance.

## 6. Account limits

Free tier: 200 credits/month, **5 minutes per job**, one queued job at a time, bookable free
QPU slots (`bookable: true` on both platforms). Shots are the billing unit — a shot is any
detected event containing at least one photon.
