# Writing the paper from this repository

For whoever writes the submission. Every number below is regenerable, and this file says
which file it comes from. If a claim is not on this page, check before writing it — several
plausible-sounding claims about this project have been measured and found false, and they are
listed at the bottom.

**Source of truth is always a CSV or JSON under `results/`, never prose.** README and paper
text are downstream of those files. If they disagree, the CSV wins and the prose is a bug.

---

## 1. What the paper can claim

### 1.1 Accuracy — best median on 9 of 10 tasks

Source: `results/benchmarks/<task>_raw.csv` (per-seed), `<task>_stats.json` (DM, MCS).
Table macro: `\PaperTableBenchmarks` in `paper/workshop_draft/results_tables.tex`.

5 seeds, median NRMSE, standard input-driven protocol, per-model ridge penalty selected on
validation, equal Optuna budget across models within each dataset.

| Task | Photonic | Best baseline | DM *p* | Status |
|---|---|---|---|---|
| NARMA-20 | 0.1876 | 0.4395 (RFF) | < 0.0001 | **decisive** |
| NARMA-10 | 0.0937 | 0.1763 (poly) | 0.0002 | **decisive** |
| NARMA-5 | 0.0088 | 0.0160 (ESN) | 0.0002 | **decisive** |
| Parity (d=3) | 6.0e-15 | 3.0e-14 (RFF) | < 0.0001 | decisive but saturated |
| Mackey-Glass (h=17) | 1.1e-5 | 2.2e-4 (ESN) | 0.0001 | decisive but saturated |
| Hénon (h=4) | 1.7e-5 | 8.3e-5 (RFF) | 0.016 | saturated |
| Lorenz-63 (h=20) | 0.1634 | 0.2124 (ESN) | 0.095 | **lead, not a win** |
| Channel eq. | 0.0773 | 0.0883 (ESN) | 0.48 | **lead, not a win** |
| Santa Fe laser | 0.0596 | 0.0607 (ESN) | 0.90 | **lead, not a win** |
| S&P 500 RV | 0.6993 | **0.6957 (linear)** | 0.77 | **loss** |

Three qualifications belong next to the headline, not in a footnote:

- **Three of the nine leads do not separate statistically** (p = 0.095, 0.48, 0.90). Write
  them as leads. On Lorenz-63 the per-seed NRMSE spans 0.06–11.0 across models; five seeds
  cannot resolve that.
- **Three tasks are saturated** below 1e-5, where several models tie and ratios are
  meaningless.
- **S&P 500 is a loss to the *linear control*** — a ridge on the raw window. Keep it in.
  Dropping the one dataset where the architecture fails is the thing reviewers look for.

Medians, not means: on chaotic tasks one unlucky seed moves a mean to 3.5× its median.

### 1.2 It is the feature map, not the interference

Source: `results/quantumness/indistinguishability.csv`. Script: `experiments/quantumness.py`.

Perceval's partial-distinguishability parameter interpolates between `perm(|U|²)` (visibility
0, classical particles, no interference) and `|perm(U)|²` (visibility 1, full interference),
holding circuit, encoding, state and readout fixed, in the infinite-shot limit so sampling
noise cannot mask the effect. Paired across seeds.

NARMA-10, 2 photons: 0.2465 at visibility 0 against 0.2373 at visibility 1 — within seed
scatter. Same on channel equalisation (0.1726 vs 0.1728) and Hénon (0.8811 vs 0.8830).

**Two-photon interference contributes nothing measurable.** What carries the performance is
the combinatorial expansion into `C(m, n)` Fock bins under a phase-encoded unitary. This is a
null result and should be reported as one — it pre-empts the first objection a reviewer at a
quantum workshop will raise, rather than inviting it.

### 1.3 What the recurrence is worth

Source: `results/feedback/feedback_strength.csv`. Figure:
`results/figures/feedback_strength.png`. Script: `experiments/feedback_strength.py`.

| Task | No feedback | Best gain | NRMSE there | Recurrence buys |
|---|---|---|---|---|
| NARMA-5 | 0.02439 | 0.05 | 0.00705 | **3.46×** |
| Mackey-Glass | 3.0e-5 | 0.005 | 1.0e-5 | 3.06× (at the floor) |
| NARMA-10 | 0.09369 | 0.02 | 0.08739 | 1.07× |
| NARMA-20 | 0.16663 | **0** | 0.16663 | 1.00× |

Why the sweep and not the on/off ablation: the search selects `feedback=False` outright on
8 of 11 datasets, so there the ablation and the model are the same object and the gap is zero
by construction. Where feedback is kept, the gain sits at the floor of its range (0.010–0.016
against a ceiling of 3.0). A boolean cannot separate "recurrence matters" from "the search
wanted it nearly off".

**The positive result here is the failure boundary.** Above `g_fb ≈ 0.6` every task sits at
NRMSE ≈ 1 — no better than predicting the target's mean — and stays there. That is the echo
state property breaking: the state stops contracting and the reservoir loses its fading
memory. Onset is task-dependent, endpoint is not.

### 1.4 Noise robustness — the result Quandela asked for

Source: `results/noise/noise_narma10_all.csv`. Figure: `results/figures/noise_narma10.png`.
Perceval `NoiseModel` with threshold detectors, at operating points read from the cloud API.

Device imperfections barely register. Indistinguishability 0.5 → 1.0 moves NRMSE between
0.2468 and 0.2486; g²(0) from 0 to 0.30 moves it 0.2479 → 0.2498. Full Ascella and Belenos
models — every source at once — give 0.2483 and 0.2465 against a noiseless 0.2479.

**Shot count is the whole story:**

| Coincidences/timestep | 1 000 | 3 000 | 10 000 | 30 000 | ∞ |
|---|---|---|---|---|---|
| NRMSE | 0.4113 | 0.4038 | 0.3798 | 0.3433 | 0.2479 |

Classical control sits at 0.3481, so roughly 3×10⁴ coincidences per timestep are needed to
beat it. The mechanism is not surprising: a reservoir is a random feature map and the readout
is refitted on whatever the device produces, so a *perturbed* map is still a good map;
sampling noise is different in kind because it corrupts every feature independently at every
timestep and no readout can absorb it.

Design rule, which inverts the usual instinct: **optimise for coincidence rate, not photon
quality.** Rate falls as `transmittance^n`, which is what forces two photons.

### 1.5 Hardware — feasibility and fidelity, not accuracy

Source: `hardware/results/qpu_combined.json`, per-job files `qpu_qpu_belenos_*.json`,
`hardware/results/hardware_summary.csv`. Full write-up: `hardware/PLAN.md`.

126 timesteps on `qpu:belenos`, 2 photons in 10 modes, stitched from six
trajectory-consistent slices out of eight submitted jobs.

| Claim | Evidence |
|---|---|
| The device reproduces the simulated feature map | correlation **0.822** (0.805–0.844 per job, across 4e3–2e4 shots) |
| The two-photon design decision was right | 0.33 → 0.82 correlation, 390× more counts, argued from `t^n` *before* measuring |
| A simulation prediction held on silicon | predicted shot-limited below ~3e4 coincidences; measured 4–20e3 and it is |
| Reconfiguration, not photon collection, is the bottleneck | 17–23 timesteps per 5-minute job across a 5× shot range → 13–18 s/timestep |

**Do not claim a hardware accuracy win.** The stitched run scores hardware 0.8360, simulation
0.5487, classical 0.4971 — but on 66 training rows against 65 features. It is data-starved and
shot-starved; the free-tier credit ceiling is roughly 30× short. The honest sentence is that
losing here is the *predicted* outcome, measured rather than simulated.

### 1.6 Capacity — it is not just more features

Source: `results/capacity/capacity_narma10.csv`. Figure:
`results/figures/capacity_narma10.png`. Script: `experiments/matched_capacity.py`.

The tuned photonic configurations use far more features than the baselines, so every family
is swept across a common dimension range. At ~1400 features photonic reaches 0.109 against
0.188 for RFF and 0.268 for an ESN; the ESN saturates beyond 2000 units and the photonic map
does not.

**Report feature dimension next to every NRMSE.** The tuned configurations run to 123k
(NARMA-20) and 361k (channel eq.) features. Hiding that is the fastest way to lose a reviewer;
the capacity sweep is the answer to it and should be prominent.

Memory: `results/capacity/memory_ipc.csv` — linear memory capacity ~11 for the photonic
reservoir against 27–30 for an ESN. It has **less** linear memory and wins through nonlinear
capacity per feature (0.85 vs 0.43–0.62).

### 1.7 Crossover — when would a chip pay?

Source: `results/crossover/crossover.csv`. Script: `experiments/crossover.py`.

At 2 photons on Ascella the device is 1607× slower than simulating it, and would need
transmittance 0.978 to break even. At 3 photons, 29 555× and transmittance 0.754. Current
hardware does not meet either. State it as a concrete target, not a hand-wave.

---

## 2. Claims that are measured and false — do not write these

| Claim | Why it is false |
|---|---|
| "Recurrence carries the result" | Search disables feedback on 8 of 11 datasets; gain sweep gives 1.00× on NARMA-20 |
| "Removing feedback doubles the error on both NARMA tasks" | NARMA-10's tuned model already has feedback off, so the ablation compares a model to itself |
| Any quantum-advantage framing | §1.2 — interference contributes nothing measurable |
| A hardware accuracy win | §1.5 — 66 rows vs 65 features |
| "50× memory capacity vs ESN" | Measured 11 vs 27–30; the reservoir has *less* linear memory |
| Any v1/v2 number | Retracted; see `results/CHANGELOG.md`. Files under `results/v1_*`, `results/v2_*`, `results/dm_tables*.md` carry retraction banners |

Two of these were in the repository's own README and paper draft until they were measured.
Assume anything not on this page needs checking.

---

## 3. Where the numbers live

| Want | File |
|---|---|
| Per-seed benchmark NRMSE | `results/benchmarks/<task>_raw.csv` |
| DM-HAC p-values, MCS membership | `results/benchmarks/<task>_stats.json` |
| Tuned hyperparameters, feature dims | `results/tuning/<task>_<model>.json` |
| Feedback gain sweep | `results/feedback/feedback_strength.csv` |
| Noise sweeps and device operating points | `results/noise/noise_<task>_all.csv` |
| Interference test | `results/quantumness/indistinguishability.csv` |
| Capacity sweep, memory/IPC | `results/capacity/` |
| Crossover | `results/crossover/crossover.csv` |
| Hardware, stitched | `hardware/results/qpu_combined.json` |
| Hardware, per job | `hardware/results/qpu_qpu_belenos_*.json` |

**LaTeX tables are generated, never typed.** `python scripts/make_paper_tables.py` writes
`paper/workshop_draft/results_tables.tex`, which `main.tex` pulls in via `\input`. Editing
that file by hand will be silently overwritten. Add a table by adding a function to the
script.

Figures likewise: `python experiments/make_figures.py`.

---

## 4. Method details a reviewer will ask about

All of this is in `docs/PROTOCOL.md`, which is a pre-registration plus a numbered deviation
log. Deviations 9–15 cover the search procedure and are the ones most likely to be
questioned.

- **Equal tuning budget** across all five models *within* each dataset. Budgets differ
  *between* datasets (100–300 trials) and that is stated — the within-dataset equality is the
  comparison that matters.
- **The search objective averages over several estimates**, because a single validation split
  is overfit past ~150 trials. Measured twice: NARMA-20 improved validation 0.191 → 0.163
  while test degraded 0.180 → 0.223. Synthetic tasks resample the generator
  (`--objective-seeds`); recorded series use rolling-origin windows (`--val-blocks`).
- **`--max-dim` prunes photonic trials above 30k features.** A compute budget: the top of the
  space is ~4e5 features, which on a 4000-step series is a 12 GB matrix.
- **Three datasets report pre-deviation searches.** `santa_fe`, `henon` and `parity_d3` use
  their original 100-trial single-split searches; the replacements were worse or unfinished.
  Deviation 15.
- **Deviation 9 records that rolling-origin validation was adopted *after* seeing a test
  number degrade, not before.** Keep that in the paper. A reviewer would rather read it from
  you than find it.

---

## 5. Practical

```bash
conda activate quandela                       # perceval 1.1.0, merlin 0.3.1 (pinned)
python -m pytest tests/ -q                    # 54 tests; run before trusting anything
python scripts/make_paper_tables.py           # regenerate all LaTeX tables
python experiments/make_figures.py            # regenerate all figures
cd paper/workshop_draft && latexmk -pdf main.tex
```

`paper/workshop_draft/main.tex` currently builds clean at **14 pages** with 0 errors and 0 undefined references.
SaTQuML wants ~9, so the remaining work is cutting, not writing from scratch. The material
most safely cut is breadth of task coverage; the material that should survive is the hardware
section, the noise study and the two null results, because those are what a workshop reviewer
in this area has not seen from everyone else.
