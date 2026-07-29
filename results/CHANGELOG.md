# Results changelog

## v3 (current) — recurrent model, standard protocols

**All v1/v2 numbers are superseded and should not be quoted.** They were produced under a
protocol that is not comparable to the reservoir-computing literature, against an echo state
network with no input-scaling parameter, and with a single ridge penalty shared across
models of very different feature counts.

What changed:

- **Recurrence.** The model has a state that persists across timesteps
  (`src/temporal_qrc.py`). Removing it doubles the error on both NARMA tasks.
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

Headline (5 seeds, NRMSE): photonic 0.0912 on NARMA-10 and 0.2262 on NARMA-20, both with
DM-HAC p < 0.001 against every baseline; a tie with the ESN on Mackey-Glass (p = 0.41); and
third of six on S&P 500 RV where nothing is statistically distinguishable.

### Retracted

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
