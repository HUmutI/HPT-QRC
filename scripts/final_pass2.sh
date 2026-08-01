#!/usr/bin/env bash
# Last compute. Santa Fe under a feature-dimension cap, then every stale benchmark.
#
# sp500_rv already completed its rolling-origin re-tune uncapped, and its winning
# configuration came out at 5444 features -- well under the cap -- so re-running it capped
# would explore a space it never used. Santa Fe is 5x longer and its search kept wandering
# into ~4e5-feature configurations, where a 12 GB feature matrix put the process at 7% CPU
# for five hours. `--max-dim 30000` prunes those before the matrix is built.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/Users/umut/miniconda3/envs/quandela/bin/python
say() { echo "[$(date '+%H:%M:%S')] $*"; }

say "retune santa_fe -- 120 trials x 3 rolling-origin windows, dim capped at 30k"
$PY -W ignore experiments/tune_temporal.py --dataset santa_fe --model all --trials 120 \
    --val-blocks 3 --max-dim 30000

say "re-benchmarking every dataset whose configuration changed"
$PY -W ignore experiments/run_benchmarks.py \
    --datasets santa_fe sp500_rv henon parity_d3 --seeds 5

say "FINAL PASS COMPLETE"
