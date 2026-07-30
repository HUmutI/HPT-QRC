#!/usr/bin/env bash
# Collect a full QPU trajectory as a sequence of jobs that each fit the execution cap.
#
# The free tier allows one waiting job at a time and kills any job at five minutes. At 20000
# shots per timestep on Belenos we measured ~16 s per timestep, so ~18 timesteps fit. A
# 120-step trajectory therefore needs 7 sequential jobs.
#
# Each slice is submitted, polled to completion, and harvested before the next is sent.
# Completed slices are written to disk as they land, so interrupting this loses only the
# slice in flight.
#
# Usage:
#   bash hardware/run_qpu_campaign.sh [total_steps] [slice_len] [shots] [platform]

set -uo pipefail
cd "$(dirname "$0")/.."

TOTAL="${1:-120}"
SLICE="${2:-18}"
SHOTS="${3:-20000}"
PLATFORM="${4:-qpu:belenos}"
PY="${PY:-/Users/umut/miniconda3/envs/quandela/bin/python}"

echo "QPU campaign: $TOTAL timesteps in slices of $SLICE at $SHOTS shots on $PLATFORM"
echo

for (( start=0; start<TOTAL; start+=SLICE )); do
  echo "=== slice ${start}:$(( start + SLICE < TOTAL ? start + SLICE : TOTAL )) ==="
  if ! $PY -W ignore hardware/submit_qpu_run.py \
        --platform "$PLATFORM" --steps "$TOTAL" \
        --slice-start "$start" --slice-len "$SLICE" --shots "$SHOTS" 2>&1 | grep -v -i warning
  then
    echo "submission failed (likely out of credit or a job already queued); stopping."
    break
  fi

  # Poll until this job resolves. A job cancelled by the cap still returns finished
  # iterations, so 'canceled' counts as resolved and is harvested like any other.
  for (( i=0; i<240; i++ )); do
    sleep 15
    state=$($PY -W ignore -c "
import sys,json; sys.path.insert(0,'hardware')
from hw_backend import get_token
import perceval as pcvl
p=json.load(open('hardware/pending_qpu_runs.json'))
if not p: print('none'); raise SystemExit
r=p[-1]
print(pcvl.RemoteProcessor(r['platform'],token=get_token())._rpc_handler.get_job_status(r['job_id']).get('status'))
" 2>/dev/null | tail -1)
    echo "    [$((i*15))s] $state"
    case "$state" in
      completed|canceled|failed|none) break ;;
    esac
  done

  $PY -W ignore hardware/fetch_qpu_run.py 2>&1 | grep -v -i warning | grep -E "REAL QPU|NRMSE|lift|correlation|counts|harvest|pending"
  echo
done

echo "campaign finished. Combine slices with:  python hardware/combine_qpu_slices.py"
