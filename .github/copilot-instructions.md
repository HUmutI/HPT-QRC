# Working in this repository

A recurrent linear-optical reservoir computer for time-series forecasting, benchmarked
against classical feature maps. Simulation only unless a file says otherwise.

## Layout

- `src/photonic_core.py` — exact boson-sampling probabilities. Matches MerLin to 1e-16 and
  is ~310× faster by exploiting the layered circuit structure. **Do not change the physics
  here without running `tests/test_photonic_core.py`** — every result depends on it.
- `src/temporal_qrc.py` — the model. State feeds back into the phase encoding; only the
  ridge readout is trained.
- `src/rc_protocol.py` — the shared split, ridge-penalty selection and metrics. Every model
  goes through this. Do not fit a readout anywhere else.
- `src/tasks.py` — benchmark registry. `narma*` are input-driven (standard protocol);
  `narma10_autoregressive` is the older, easier variant kept only for reproducibility.
- `src/baselines_rc.py` — baselines, all with the same interface.
- `src/noise.py` — hardware specs and the Perceval noise backend.
- `src/multi_qrc.py` — the previous windowed model. Kept so old numbers reproduce. Not the
  current model; do not extend it.
- `hardware/` — QPU execution path.

## Rules that exist for a reason

- **Never fix the ridge penalty across models.** It is selected per model on a validation
  slice. A shared penalty silently favours whichever feature count it happens to suit; the
  previous version of this code did that and the comparison was not meaningful.
- **Never tune one model against untuned baselines.** Every model gets the same Optuna trial
  budget in `experiments/tune_temporal.py`.
- **The classical control belongs in every table.** Ridge on a window of the raw drive is
  the model the previous architecture never actually beat. If a change makes the quantum
  features look good, check the control first.
- **Report feature dimension alongside NRMSE**, and cite `experiments/matched_capacity.py`
  for any claim that the photonic map wins. It uses more features than the baselines.
- **Scalers fit on training rows only.** `tests/test_protocol_and_model.py` enforces this
  and causality; if you add a feature map, add it to those tests.
- **Noise goes through Perceval, not our own approximation.** `src/photonic_core.py` is
  exact only in the noiseless case.

## Claims not to make

No quantum advantage. No hardware claim while the QPUs are in maintenance. No "beats X"
without the matched-dimension column and the DM-HAC p-value. The S&P 500 result is a
non-result and is reported that way.

Retracted earlier claims — do not reintroduce them: the "50× memory capacity vs ESN"
figure (measured memory capacity is ~11 vs the ESN's 27–30, i.e. the reservoir has *less*
linear memory), and any statement that the previous windowed model beat classical baselines.

## Environment

`conda activate quandela` (Python 3.11, perceval-quandela 1.1.0, merlinquantum 0.3.1).
The hardware conventions were verified against those versions specifically.
Run `python -m pytest tests/ -q` before committing anything that touches `src/`.
