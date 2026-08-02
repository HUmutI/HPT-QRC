**Read [`../../docs/FOR_THE_PAPER.md`](../../docs/FOR_THE_PAPER.md) first.** It is the claim-by-claim guide with sources.

# Workshop draft

`main.tex` compiles clean at **14 pages**, no undefined references, every table and figure
generated rather than typed. Last regenerated against results on **2026-08-01**. See
`results/CHANGELOG.md` for what earlier versions claimed and why those claims were retracted.

**Not yet written for submission.** SaTQuML wants ~9 pages; this is 14, and it is a full
technical report rather than a workshop paper. Cutting it is the remaining work.

**Two claims are retracted and must not reappear in any draft:**

- *"Recurrence carries the result."* The search disables feedback on 8 of 11 datasets, and
  the gain sweep measures 1.00× on NARMA-20. Use `\PaperTableBenchmarks`'s dagger marks and
  `results/figures/feedback_strength.png` instead — the honest version is a stronger section.
- *Any hardware accuracy claim.* 126 timesteps were measured on `qpu:belenos`, but with 66
  training rows against 65 features. What is reportable is feature-level agreement with
  simulation at correlation 0.82, and a shot-limited prediction confirmed on silicon.

Also do not frame this as quantum advantage: `experiments/quantumness.py` finds no detectable
contribution from two-photon interference, and that null result is part of the paper.

## Tables are generated, not typed

```bash
python scripts/make_paper_tables.py     # -> paper/workshop_draft/results_tables.tex
```

Then in `main.tex`:

```latex
\usepackage{siunitx}    % results_tables.tex uses \num
\input{results_tables}
...
\begin{table}[h]\centering
  \PaperTableBenchmarks
  \caption{NRMSE, 5 seeds, mean $\pm$ std, standard input-driven protocol.}
\end{table}
```

Available macros: `\PaperTableBenchmarks`, `\PaperTableDM`, `\PaperTableCapacity`,
`\PaperTableMemory`, `\PaperTableNoise`, `\PaperTableRates`.

Hand-typed numbers are how a paper ends up disagreeing with its own repository. Regenerate
after any results change.

## Figures

```bash
python experiments/make_figures.py
```

Writes to `results/figures/`: `capacity_narma10.png`, `noise_narma10.png`, `benchmark.png`,
`coincidence_rate.png`.

## What the paper can claim

- A recurrent linear-optical reservoir beats tuned classical feature maps on NARMA-10 and
  NARMA-20, **at matched feature dimension**, with DM-HAC $p < 0.001$.
- Recurrence accounts for roughly half the error: removing feedback and changing nothing
  else doubles NRMSE on both NARMA tasks.
- Accuracy is essentially unaffected by device imperfections at measured Ascella and Belenos
  operating points, while being strongly limited by coincidence count — which gives a design
  rule (optimise rate, not photon quality) and a quantitative feasibility threshold.

## What the paper must not claim

- No quantum advantage of any kind.
- Nothing about S&P 500 realised volatility. The model ranks third of six there and no model
  is statistically distinguishable from any other ($p = 0.45$–$0.88$). Report it and move on.
- No hardware claim. Both QPUs have been in maintenance; nothing here is a chip measurement.
  If a hardware run lands, the `replay` protocol simulates the feedback path and that must be
  stated in the caption, not just the appendix.
- No "outperforms" without both the matched-dimension column and the DM-HAC $p$-value. The
  tuned photonic configuration uses more features than the baselines.
- The Model Confidence Set has little power at these sample sizes (200–600 test points) and
  retains almost every model on every task. Do not present it as though it discriminates.

## Standing rules for edits

- Never reintroduce the "50× memory capacity" claim. Measured linear memory capacity is ~11
  for the photonic reservoir against 27–30 for an ESN — it has *less* linear memory and wins
  through nonlinear capacity per feature instead.
- Never type a result number into `main.tex`. Regenerate the tables.
- If a result changes, rerun `scripts/make_paper_tables.py` and
  `experiments/make_figures.py` before recompiling.
