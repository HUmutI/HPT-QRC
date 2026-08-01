#!/usr/bin/env bash
# Re-benchmark henon and parity_d3 under their new tuned configurations.
#
# retune_stale.sh was supposed to do this, but it was appended to while bash was executing
# it: bash tracks a byte offset into the script file, and extending the file mid-run shifted
# what it read next, so it skipped the re-benchmark line and exited. Both datasets therefore
# have new tuned configurations and stale 5-seed numbers. Never edit a running shell script,
# including at the end.
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

say "waiting for the rolling-origin pass"
while running tune_temporal.py || running run_benchmarks.py; do sleep 30; done

say "re-benchmarking henon and parity_d3"
$PY -W ignore experiments/run_benchmarks.py --datasets henon parity_d3 --seeds 5
say "ALL BENCHMARKS COMPLETE"
