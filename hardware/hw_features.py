"""Hardware feature extraction: sample the reservoir circuit, histogram the
shots into the unbunched Fock subspace, and apply the same LexGrouping the
simulation uses so feature dimensions match exactly.

Pipeline per timestep:
    encoded window -> input* phases -> Sampler iteration -> BSCount
    -> histogram over spec["output_keys"] -> p_hat -> LexGrouping -> features
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import perceval as pcvl
from perceval.algorithm import Sampler
from merlin import LexGrouping

from phase_export import rebuild_circuit

CACHE_DIR = Path(__file__).parent / "cache"


def build_hw_circuit(spec: dict) -> pcvl.Circuit:
    """Rebuild the circuit with t phases fixed to the exported simulation values.

    input* parameters stay free — they are set per timestep through Sampler
    iterations (circuit_params), so many timesteps share one cloud job.
    """
    circ, _ = rebuild_circuit(spec)
    t_phases = spec["t_phases"]
    for p in circ.get_parameters():
        if p.name in t_phases:
            p.set_value(t_phases[p.name])
    return circ


def make_iterations(spec: dict, X_win: np.ndarray) -> list:
    """One Sampler iteration per timestep: encoded window -> input phases (radians)."""
    return [
        {"circuit_params": {name: float(v) for name, v in zip(spec["input_names"], row)}}
        for row in np.atleast_2d(X_win)
    ]


def counts_to_phat(results, spec: dict) -> tuple[np.ndarray, dict]:
    """Histogram a BSCount into the unbunched subspace, in output_keys order.

    Bunched states (>1 photon in a mode) and wrong photon numbers are dropped;
    the drop rate is reported so the probe can quantify it.
    """
    keys = [tuple(k) for k in spec["output_keys"]]
    index = {k: i for i, k in enumerate(keys)}
    counts = np.zeros(len(keys))
    total = kept = 0
    for state, c in results.items():
        total += c
        i = index.get(tuple(state))
        if i is not None:
            counts[i] += c
            kept += c
    if kept == 0:
        p_hat = np.full(len(keys), 1.0 / len(keys))
    else:
        p_hat = counts / kept
    meta = {"raw_counts": int(total), "unbunched_counts": int(kept),
            "drop_rate": 1.0 - kept / total if total else 1.0}
    return p_hat, meta


def phat_to_features(p_hat: np.ndarray, spec: dict) -> np.ndarray:
    """Apply the identical LexGrouping mapping the simulation uses."""
    lg = LexGrouping(len(spec["output_keys"]), spec["lex_out"])
    with torch.no_grad():
        return lg(torch.tensor(p_hat, dtype=torch.float32).unsqueeze(0)).numpy().ravel()


def _job_key(platform: str, spec: dict, X_win: np.ndarray, shots: int) -> str:
    payload = json.dumps({
        "platform": platform,
        "t": spec["t_phases"],
        "x": np.asarray(X_win).round(12).tolist(),
        "shots": shots,
        "input_state": spec["input_state"],
    }, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def _run_job(make_sampler, shots: int, key: str, poll_s: float = 5.0):
    """Submit asynchronously and poll defensively.

    A transient HTTP timeout while polling must NOT kill the run — the job
    keeps existing (and billing) cloud-side. The job id is persisted to
    cache/jobid_<key>.txt right after submission so a crashed script never
    orphans a job untraceably.
    """
    import requests

    sampler = make_sampler()
    if not hasattr(sampler.sample_count, "execute_async"):
        return sampler.sample_count(shots)  # local processor: sync path

    # Job creation can hit transient gateway/internal errors (4xx/5xx) when the
    # cloud API is degraded — retry with backoff, a FRESH Sampler each time
    # (a perceval Job object cannot be resubmitted).
    job = None
    for attempt in range(1, 6):
        try:
            job = make_sampler().sample_count.execute_async(shots)
            break
        except (requests.exceptions.RequestException, ConnectionError) as e:
            if attempt == 5:
                raise
            wait = 30 * attempt
            print(f"[hw] submit attempt {attempt}/5 failed ({e}); retrying in {wait}s")
            time.sleep(wait)
    job_id = getattr(job, "id", None)
    if job_id:
        CACHE_DIR.mkdir(exist_ok=True)
        (CACHE_DIR / f"jobid_{key}.txt").write_text(str(job_id))
        print(f"[hw] job submitted, id={job_id}")

    consecutive_errors = 0
    while True:
        try:
            if job.is_complete:
                break
            if getattr(job, "is_failed", False):
                raise RuntimeError(f"cloud job {job_id} failed: {job.status}")
            consecutive_errors = 0
        except (requests.exceptions.RequestException, ConnectionError) as e:
            consecutive_errors += 1
            print(f"[hw] poll error ({consecutive_errors}/10), retrying: {type(e).__name__}")
            if consecutive_errors >= 10:
                raise RuntimeError(
                    f"cloud API unreachable after 10 poll attempts; job {job_id} "
                    "is still alive server-side — check the dashboard."
                ) from e
        time.sleep(poll_s)
    return job.get_results()


def sample_batch(processor, spec: dict, X_win: np.ndarray, shots: int,
                 platform: str, max_shots_per_call: int = None,
                 use_cache: bool = True) -> dict:
    """Run all timesteps in X_win as one multi-iteration job. Returns a dict with
    per-timestep p_hat, features, and job metadata. Results are cached to disk so
    a re-run never re-bills shots.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    key = _job_key(platform, spec, X_win, shots)
    cache_file = CACHE_DIR / f"job_{key}.json"

    if use_cache and cache_file.exists():
        print(f"[hw] cache hit: {cache_file.name}")
        return json.loads(cache_file.read_text())

    circ = build_hw_circuit(spec)
    processor.set_circuit(circ)
    processor.min_detected_photons_filter(spec["photons"])
    processor.with_input(pcvl.BasicState(spec["input_state"]))

    def make_sampler():
        # A perceval Job object cannot be resubmitted ("job has already been
        # executed"), so every submit attempt needs a fresh Sampler.
        s = Sampler(processor, max_shots_per_call=max_shots_per_call or 10 * shots)
        s.add_iteration_list(make_iterations(spec, X_win))
        return s

    t0 = time.time()
    job_result = _run_job(make_sampler, shots, key)
    wall = time.time() - t0

    out = {"platform": platform, "shots_requested": shots, "wall_time_s": wall,
           "n_timesteps": len(np.atleast_2d(X_win)), "steps": []}
    for res in job_result["results_list"]:
        p_hat, meta = counts_to_phat(res["results"], spec)
        feats = phat_to_features(p_hat, spec)
        step = {"p_hat": p_hat.tolist(), "features": feats.tolist(), **meta}
        for perf_key in ("physical_perf", "logical_perf"):
            if perf_key in res:
                step[perf_key] = float(res[perf_key])
        out["steps"].append(step)

    cache_file.write_text(json.dumps(out))
    print(f"[hw] job done in {wall:.1f}s, cached to {cache_file.name}")
    return out
