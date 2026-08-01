#!/usr/bin/env bash
# The last compute: rolling-origin re-tune of the two recorded series, then the benchmarks
# that are still stale.
#
# Trial budget is 120 rather than 300. Rolling-origin validation evaluates every trial on
# three windows, so a 300-trial search costs 3.6x a single-split one; on Santa Fe (the
# longest series, 4000 steps) that ran 5 CPU-hours without finishing. 120 trials x 3 windows
# is still 360 feature builds per model, more than the 300 the single-split runs did, and the
# budget is identical across all five models on these datasets -- which is the comparison
# that has to be fair. The differing budget between *datasets* is stated in the changelog.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/Users/umut/miniconda3/envs/quandela/bin/python
say() { echo "[$(date '+%H:%M:%S')] $*"; }

for ds in sp500_rv santa_fe; do
  say "retune $ds -- 120 trials x 3 rolling-origin windows"
  $PY -W ignore experiments/tune_temporal.py --dataset "$ds" --model all --trials 120 \
      --val-blocks 3
done

say "re-benchmarking every dataset whose configuration changed"
$PY -W ignore experiments/run_benchmarks.py \
    --datasets santa_fe sp500_rv henon parity_d3 --seeds 5

say "FINAL PASS COMPLETE"
