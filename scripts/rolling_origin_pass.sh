#!/usr/bin/env bash
# Re-tune the two recorded series with a rolling-origin objective.
#
# santa_fe and sp500_rv cannot be resampled, so their searches scored every trial against a
# single validation window and overfit it. Measured on santa_fe: going from 100 to 300 trials
# improved validation from 0.0318 to 0.0294 while test degraded from 0.0601 to 0.0705. The
# synthetic tasks already avoid this by averaging the objective over fresh realisations;
# `--val-blocks 3` is the same principle for data that cannot be regenerated.
#
# Idempotent, because it is also appended to retune_stale.sh and bash's behaviour when a
# script is extended mid-execution is not something to rely on for an unattended run. If that
# append did execute, this exits without spending the compute again.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/Users/umut/miniconda3/envs/quandela/bin/python
say() { echo "[$(date '+%H:%M:%S')] $*"; }

running() {
  local pid
  for pid in $(pgrep -f "experiments/$1" 2>/dev/null); do
    case "$(ps -o comm= -p "$pid" 2>/dev/null)" in *python*) return 0 ;; esac
  done
  return 1
}

say "waiting for the retune chain to finish"
while running tune_temporal.py || running run_benchmarks.py; do sleep 30; done

if grep -q "rolling-origin validation windows" logs/retune_stale.log 2>/dev/null; then
  say "rolling-origin pass already ran inside retune_stale.sh -- nothing to do"
  exit 0
fi

for ds in santa_fe sp500_rv; do
  say "retune $ds with 3 rolling-origin validation windows"
  $PY -W ignore experiments/tune_temporal.py --dataset "$ds" --model all --trials 300 \
      --val-blocks 3
done

say "final re-benchmark of the recorded series"
$PY -W ignore experiments/run_benchmarks.py --datasets santa_fe sp500_rv --seeds 5
say "ROLLING ORIGIN PASS COMPLETE"
