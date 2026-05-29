FOLDER GUIDE — EPFL_ANTI
=========================

Project: HPT-QRC (Hybrid Photonic Temporal Quantum Reservoir Computing)
Goal:    Benchmark QRC against classical models on NARMA10, Mackey-Glass,
         and S&P 500 realized volatility — for an academic paper.


SETUP (do this first)
---------------------

1. Create and activate the conda environment:

     conda create -n quandela python=3.11
     conda activate quandela

2. Install standard dependencies:

     pip install -r requirements.txt

3. Verify install:

     python -c "import perceval; import merlin; print('OK')"


HOW TO RUN
----------

Always run from the project root (EPFL_ANTI/), not from inside a subfolder.

   python experiments/multi_seed_benchmark.py    # main benchmark (start here)
   python experiments/train_narma.py             # quick single-seed test
   python experiments/ablation_study.py          # ablation experiments
   python scripts/plot_dm_heatmaps.py            # generate result plots

Tuning (run before multi_seed_benchmark if JSON configs are missing in results/):
   python experiments/tune_qrc.py
   python experiments/tune_baselines.py

All outputs go to results/ automatically.


FOLDER STRUCTURE
----------------

src/                        Core library (do not run directly — imported by experiments)
  multi_qrc.py              Main QRC model (HPT_QRC_Multi class) — needs merlin
  data_loader.py            Data loaders: NARMA10, Mackey-Glass, SP500, VIX
  classical_baselines.py    AR, HAR, HARX, LSTM, RC, ClassicalContextRidge models
  esn_baseline.py           Echo State Network
  rff_baseline.py           Random Fourier Features + Ridge (classical comparator)
  dm_mcs.py                 Diebold-Mariano test + Hansen MCS (statistical tests)
  noise_models.py           Shot noise and indistinguishability models
  walk_forward.py           Walk-forward CV split logic

experiments/                Runnable scripts — always run from project root
  multi_seed_benchmark.py   5-seed benchmark, mean+-std, publication-ready  <- MAIN
  train_narma.py            Single-seed run + DM tables + overlay plots
  walk_forward_runner.py    Walk-forward cross-validation driver
  memory_capacity.py        Memory capacity analysis (Jaeger 2001)
  esp_check.py              Echo-state-property / fading-memory check
  efficiency_benchmark.py   Training time and model size table
  pca_independence.py       PCA / linear independence analysis of features
  online_rls.py             Online recursive least-squares adaptation
  flops_energy_calc.py      Static FLOP / energy estimate
  ablation_study.py         Ablation: photon count, reservoir count, window size
  ablation_fock_scaling.py  Joint (n_photons, n_modes) Fock-dim scaling sweep
  ablation_matched_dim.py   Photon-list ablation at matched feature dimension
  tune_qrc.py               Optuna hyperparameter search for QRC
  tune_baselines.py         Optuna hyperparameter search for LSTM / ESN / RFF
  tune_sp500_window.py      Window size tuning for S&P 500

scripts/                    Post-processing utilities
  plot_dm_heatmaps.py       Generate DM test heatmap plots
  format_dm.py              Print DM tables as markdown
  render_report.py          Render full walkthrough report

results/                    All experiment outputs: CSVs, PNGs, JSON configs
                            (JSON files like qrc_best_configs.json are tuning results
                             needed by multi_seed_benchmark — run tune_*.py first if missing)

docs/                       Documentation
  PROTOCOL.md               Pre-registered experiment protocol (what we test and how)
  walkthrough.md            Full research log and result commentary

literature/                 Reference papers and benchmark data
  qrc_repo/Data.CSV         S&P 500 realized volatility dataset (the actual data file)
  esn_repo/                 ESN reference implementation

paper/workshop_draft/       LaTeX manuscript files


QUICK REFERENCE
---------------
  Model code      ->  src/multi_qrc.py
  Data loading    ->  src/data_loader.py
  Run benchmark   ->  experiments/multi_seed_benchmark.py
  Results         ->  results/
  Paper files     ->  paper/workshop_draft/
