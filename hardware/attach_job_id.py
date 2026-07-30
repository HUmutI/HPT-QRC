"""Attach a job id recovered from the dashboard to the most recent pending record.

Quandela returns a non-2xx status when it auto-adjusts the shot budget for insufficient
credit, and perceval raises on it -- but the job is created regardless. When that happens the
submitter saves everything except the id, and this attaches it so the job can be harvested
instead of orphaned.

Usage::

    python hardware/attach_job_id.py 1234abcd-...
"""

import json
import sys
from pathlib import Path

PENDING = Path(__file__).resolve().parents[1] / "hardware" / "pending_qpu_runs.json"


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(1)
    job_id = sys.argv[1].strip()
    pending = json.loads(PENDING.read_text()) if PENDING.exists() else []
    orphans = [r for r in pending if r.get("needs_manual_id")]
    if not orphans:
        print("no record is waiting for a job id")
        raise SystemExit(1)
    record = orphans[-1]
    record["job_id"] = job_id
    record["needs_manual_id"] = False
    PENDING.write_text(json.dumps(pending, indent=2))
    print(f"attached {job_id} to slice "
          f"{record.get('slice_start')}:{record.get('slice_end')} "
          f"@{record.get('shots')} shots")
    print("harvest with:  python hardware/fetch_qpu_run.py")


if __name__ == "__main__":
    main()
