#!/usr/bin/env bash
# Re-tune the datasets whose searches predate the current search space.
#
# santa_fe, sp500_rv, henon and parity_d3 were last tuned on 2026-07-29, before the
# multi-timescale integration knob (`n_scales`/`scale_ratio`) and the multi-realisation
# objective were added. Two of them are the tasks the model loses, so they are exactly the
# ones that should not be carrying a stale search.
#
# The multi-realisation objective needs resampleable data, so the two real-data tasks get
# one realisation and the synthetic ones get three; `tune_temporal.py` enforces this anyway.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/Users/umut/miniconda3/envs/quandela/bin/python

for ds in santa_fe sp500_rv; do
  echo "=== retune $ds (single realisation: real data) ==="
  $PY -W ignore experiments/tune_temporal.py --dataset "$ds" --model all --trials 300
done

# henon and parity_d3 are already solved to 1.7e-5 and exactly 0. No search can improve a
# saturated task, so these two run only for methodological uniformity -- every dataset
# searched under the same space -- and get half the budget. They also run last, so the two
# tasks that can actually change the paper's conclusions land first.
for ds in henon parity_d3; do
  echo "=== retune $ds (3 realisations, half budget: task already saturated) ==="
  $PY -W ignore experiments/tune_temporal.py --dataset "$ds" --model all --trials 150 \
      --objective-seeds 3
done

echo "=== re-benchmark the four under their new configs ==="
$PY -W ignore experiments/run_benchmarks.py --datasets santa_fe sp500_rv henon parity_d3 --seeds 5

# --- rolling-origin pass for the two recorded series -------------------------------------
# santa_fe and sp500_rv cannot be resampled, so their searches ran against a single
# validation split and overfit it: santa_fe going 100 -> 300 trials improved validation from
# 0.0318 to 0.0294 while test degraded from 0.0601 to 0.0705. Re-run both with the objective
# averaged over three rolling-origin windows, which is the same "average over several
# estimates" principle the synthetic tasks already get from resampling.
for ds in santa_fe sp500_rv; do
  echo "=== retune $ds (3 rolling-origin validation windows) ==="
  $PY -W ignore experiments/tune_temporal.py --dataset "$ds" --model all --trials 300 \
      --val-blocks 3
done

echo "=== final re-benchmark of the recorded series ==="
$PY -W ignore experiments/run_benchmarks.py --datasets santa_fe sp500_rv --seeds 5
