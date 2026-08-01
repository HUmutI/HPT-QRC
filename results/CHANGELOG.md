# Results changelog

## v3 (current) — recurrent model, standard protocols

**All v1/v2 numbers are superseded and should not be quoted.** They were produced under a
protocol that is not comparable to the reservoir-computing literature, against an echo state
network with no input-scaling parameter, and with a single ridge penalty shared across
models of very different feature counts.

What changed:

- **Recurrence.** The model has a state that persists across timesteps
  (`src/temporal_qrc.py`). How much that is worth is measured per task below, and on most
  tasks it is worth little — the claim that it doubles the error on both NARMA tasks is
  withdrawn.
- **Standard protocol.** NARMA is driven by its exogenous input, as in the literature. The
  previous autoregressive variant is kept as `narma10_autoregressive` and clearly labelled.
- **NRMSE** (RMSE ÷ std) as the headline metric, so numbers can be placed against published
  results directly.
- **Per-model ridge penalty selection** on a validation slice (`src/rc_protocol.py`).
- **Fixed ESN baseline.** With input scaling it reaches 0.183 on NARMA-10, matching the
  literature's ≈0.185.
- **Classical control in every table** — ridge on a window of the raw drive.
- **Matched-capacity study** (`results/capacity/`), because the tuned photonic configuration
  uses more features than the baselines.
- **Noise study** at measured Ascella/Belenos operating points via Perceval's `NoiseModel`
  with threshold detectors (`results/noise/`).

### 2026-08-01 — deeper search, hardware, and two claims withdrawn

**Hardware.** 126 timesteps measured on `qpu:belenos` at 2 photons in 10 modes, stitched from
six trajectory-consistent slices. Device features correlate 0.805–0.844 with simulation
across a 5× range of shot counts, against 0.18–0.48 for the earlier three-photon probe — the
two-photon redesign is confirmed by measurement. The accuracy row (hardware 0.8360,
simulation 0.5487, classical 0.4971) is **not** an accuracy result: 66 training rows against
65 features, and coincidences six to eight times below what the noise study says is needed.

**Search.** Multi-timescale integration and a multi-estimate objective. NARMA-20 went from a
regressed 0.4735 to 0.1771 at 300 trials × 3 realisations, its best figure to date.

#### Retracted: "recurrence is what carries the result"

Withdrawn. It rested on a binary ablation, and the current searches choose `feedback=False`
outright on 8 of 11 datasets — so on those the ablation and the model are the same object and
the gap is zero by construction. Where feedback is kept, the gain sits at the floor of its
range (0.010–0.016 against a ceiling of 3.0).

The gain sweep (`experiments/feedback_strength.py`, 5 seeds) measures what the flag could
not: **3.46× on NARMA-5, 3.06× on Mackey-Glass (both at the saturation floor), 1.07× on
NARMA-10, 1.00× on NARMA-20** — where the best gain is zero and the no-feedback ablation is
marginally better than the tuned model (0.1766 vs 0.1876, p = 0.19). The feature map, not the
recurrence, does most of the work on most tasks. What the sweep does establish is a failure
boundary: past `g_fb ≈ 0.6` every task sits at NRMSE ≈ 1, the echo state property breaking.

#### Re-tuning made three of four datasets worse on test

Recorded because it is a methodological finding, not a footnote. Re-running the searches under
the expanded space at higher trial counts:

| dataset | before (trials) | after, single split (300) | val moved |
|---|---|---|---|
| santa_fe | 0.0601 (100) | 0.0705 | 0.0318 → 0.0294 |
| sp500_rv | 0.6993 (150) | 0.7192 | 0.7709 → 0.7819 |
| henon | 2e-5 (100) | 1e-4 | both at the floor |
| **narma20** | 0.2232 (250) | **0.1771** | 0.1631 → 0.2157 |

The only one that improved is the only one whose objective averaged over several data
realisations, and its validation score got *worse* while test improved — the signature of a
search that has stopped fitting its selection split. The other three improved validation while
degrading test.

So the expanded search space is not harmful; single-split selection is, and more trials make
it worse. `--val-blocks` gives recorded series that cannot be resampled the same multi-estimate
treatment via rolling-origin validation.

Sequence stated for the record: the weakness was demonstrated on NARMA-20 *before* Santa Fe was
re-run, but the decision to add rolling-origin validation came *after* seeing Santa Fe's test
number degrade. Both sets of numbers are reported.

### Note on the abandoned re-searches (2026-08-01)

`santa_fe`, `henon` and `parity_d3` are reported under their original 100-trial single-split
searches. Their re-searches under the expanded space degraded test while improving validation
(table above), and the rolling-origin replacement was stopped before completing: the photonic
search kept sampling ~4e5-feature configurations, which on Santa Fe's 4000-step series is a
12 GB feature matrix, and the process spent five hours at 7 % CPU thrashing rather than
computing. Rather than report a configuration chosen by a procedure documented here as
inadequate, or a half-finished one, those three revert to the last complete and internally
consistent search. `experiments/tune_temporal.py --max-dim` now prunes oversized trials before
building the matrix, so a future re-run is affordable.

`sp500_rv` is the exception: its rolling-origin re-tune completed for all five models at 120
trials × 3 windows, and is reported. It moved the photonic model from 0.6993 to 0.6894 on the
tuned seed and the ESN from 0.6823 to 0.7515 — the ESN's single-split configuration was the
more overfit of the two. Across five seeds the photonic median is 0.6993 and the ESN's 0.7482.

Trial budgets therefore differ *between* datasets (100–300). Within each dataset every model
gets the identical budget, which is the comparison that has to be fair.

### Retracted (earlier)

- The "50× linear memory capacity vs ESN" claim. Measured linear memory capacity is ~11 for
  the photonic reservoir against 27–30 for an ESN — it has *less* linear memory, and wins
  through nonlinear capacity per feature instead (0.85 vs 0.43–0.62).
- Any claim that the previous windowed model beat classical baselines. On NARMA-10 under its
  own protocol, a plain ridge on the raw window scored 5.24e-3 against its 5.38e-3.
- The mixed-configuration recommendation (v2 for NARMA/MG, v1 for S&P 500). Reporting a
  different configuration per dataset because it scored better there is not defensible; one
  tuning procedure now runs per dataset for every model alike.

## v2 — window=10, photon_list=[2,3,4] (superseded)
## v1 — window=5, homogeneous n=3 (superseded)

Archived under `results/v1_window5_homo/` and `results/v2_window10_hetero/` for provenance.
