#!/usr/bin/env bash
# Reproduce every result in the repository, in dependency order.
#
# Nothing here touches the Quandela cloud. Cloud and QPU runs are separate and deliberate;
# see hardware/README.md.
#
# Wall-clock on a 12-core laptop is several hours, dominated by the Optuna searches and by
# the noise sweeps (which run Perceval inside a sequential recurrent loop). Set FAST=1 for a
# reduced-budget smoke run that exercises every path in a few minutes.
#
# Usage:
#   bash scripts/reproduce_all.sh
#   FAST=1 bash scripts/reproduce_all.sh

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
if [ "${FAST:-0}" = "1" ]; then
  TRIALS=8; SEEDS=2; NOISE_SEEDS=1
  echo "FAST mode: reduced budgets, results will NOT match the paper"
else
  TRIALS=150; SEEDS=5; NOISE_SEEDS=3
fi

DATASETS="narma5 narma10 narma20 mackey_glass_h17 lorenz63 sp500_rv"

echo "=== 1/7 tests (the core-vs-MerLin check gates everything else) ==="
$PYTHON -m pytest tests/ -q

echo "=== 2/7 hyperparameter search: every model, equal budget ==="
for ds in $DATASETS; do
  echo "--- $ds"
  $PYTHON -W ignore experiments/tune_temporal.py --dataset "$ds" --model all --trials "$TRIALS"
done

echo "=== 3/7 headline benchmark with DM-HAC and Hansen MCS ==="
$PYTHON -W ignore experiments/run_benchmarks.py --datasets $DATASETS --seeds "$SEEDS"

echo "=== 4/7 matched capacity: is it the optics or just more features? ==="
$PYTHON -W ignore experiments/matched_capacity.py --dataset narma10 --seeds 3

echo "=== 5/7 diagnostics: memory/IPC, echo state property, encoding window ==="
$PYTHON -W ignore experiments/memory_ipc.py --n 2000 --seeds 3
$PYTHON -W ignore experiments/esp_check.py --seeds 3
$PYTHON -W ignore experiments/ablation_encoding.py --seeds 3

echo "=== 6/7 noise robustness at measured device parameters ==="
$PYTHON -W ignore experiments/noise_study.py --dataset narma10 --sweep all --seeds "$NOISE_SEEDS"
# NARMA-20 needs an encoding window matching its own order, or the model is crippled before
# any noise is applied.
$PYTHON -W ignore experiments/noise_study.py --dataset narma20 --sweep all \
        --seeds "$NOISE_SEEDS" --encode-window 25

echo "=== 7/7 figures, report and paper tables ==="
$PYTHON -W ignore experiments/make_figures.py
$PYTHON -W ignore scripts/render_report.py
$PYTHON -W ignore scripts/make_paper_tables.py

echo
echo "Done. Results in results/, figures in results/figures/,"
echo "paper tables in paper/workshop_draft/results_tables.tex."
echo "Rebuild the paper with: cd paper/workshop_draft && pdflatex main && bibtex main && pdflatex main && pdflatex main"
