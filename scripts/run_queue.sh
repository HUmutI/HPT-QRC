#!/usr/bin/env bash
# The remaining compute queue, detached from any Claude session.
#
# Written so the session can be handed to another client (`claude --continue
# --remote-control`) without killing the work. Session-bound background tasks die with the
# session; this is nohup'd and does not. Progress lands in logs/, so any session can read it.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/Users/umut/miniconda3/envs/quandela/bin/python
say() { echo "[$(date '+%H:%M:%S')] $*"; }

# `pgrep -f PATTERN` matches the pattern anywhere in a full command line, including the
# command lines of *shells that merely mention it*. A wrapper shell running
#   until ! pgrep -f "...run_benchmarks.py"; do sleep 30; done
# therefore matches itself and waits forever. This has deadlocked three separate waiters in
# this project, once silently for 50 minutes. Confirm the executable really is python.
running() {
  local pid
  for pid in $(pgrep -f "experiments/$1" 2>/dev/null); do
    case "$(ps -o comm= -p "$pid" 2>/dev/null)" in *python*) return 0 ;; esac
  done
  return 1
}

# The 5-seed benchmark over channel_eq / santa_fe / sp500_rv / henon / parity_d3 is already
# running under its own nohup. Wait it out rather than racing it -- two processes writing
# results/benchmarks/ would interleave rows.
say "waiting for any in-flight 5-seed benchmark"
while running run_benchmarks.py; do sleep 30; done
say "no benchmark in flight"

# NARMA-20's winning configuration changed after the re-tune, so its 5-seed numbers were
# produced by a model that no longer exists.
say "re-benchmarking narma20 under its new configuration"
$PY -W ignore experiments/run_benchmarks.py --datasets narma20 --seeds 5 \
    > logs/bench_narma20.log 2>&1
say "narma20 re-benchmark done"

say "re-tuning the four datasets whose searches predate the current search space"
bash scripts/retune_stale.sh > logs/retune_stale.log 2>&1
say "retune chain done"

say "QUEUE COMPLETE"
