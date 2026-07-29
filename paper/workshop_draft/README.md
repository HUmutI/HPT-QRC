# Workshop draft

`main.tex` was rewritten against the current results on 2026-07-29. It compiles clean (11
pages, no undefined references) and every table and figure is generated, not typed. See
`results/CHANGELOG.md` for what the previous version claimed and why it was retracted.

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
