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
- **Sweeps must vary one axis.** `experiments/matched_capacity.py` loads the *tuned*
  parameters and changes only the width. It once also dropped `feedback`, which swept a
  different model than was tuned and produced a monotonic collapse that read as a real
  finding. If a swept curve gets worse as capacity grows, suspect the sweep, not the model.
- **Search objective averages over data realisations.** A single validation split is
  overfit past ~150 trials (NARMA-20: val 0.191→0.163 while test went 0.180→0.223).

## Claims not to make

No quantum advantage. No "beats X" without the matched-dimension column and the DM-HAC
p-value. The S&P 500 and Santa Fe results are losses to the ESN and are reported that way.

**No hardware accuracy claim.** 126 timesteps were collected on `qpu:belenos`; the run is
data-starved (66 training rows against 65 features) and shot-limited by the free-tier credit
ceiling. What the hardware supports is feature-level agreement with simulation (correlation
0.82) and the confirmation of a shot-limited prediction — not a win.

**Interference is not the mechanism.** `experiments/quantumness.py` interpolates between
`perm(|U|²)` and `|perm(U)|²` at fixed everything else and in the infinite-shot limit; the
difference is not measurable. The feature map's combinatorial structure carries the
performance. Do not write the paper as if two-photon interference does the work.

Retracted earlier claims — do not reintroduce them: the "50× memory capacity vs ESN"
figure (measured memory capacity is ~11 vs the ESN's 27–30, i.e. the reservoir has *less*
linear memory), and any statement that the previous windowed model beat classical baselines.

## Environment

`conda activate quandela` (Python 3.11, perceval-quandela 1.1.0, merlinquantum 0.3.1).
The hardware conventions were verified against those versions specifically.
Run `python -m pytest tests/ -q` before committing anything that touches `src/`.
