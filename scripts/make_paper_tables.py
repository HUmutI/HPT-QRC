"""Generate the paper's result tables as LaTeX, straight from the result CSVs.

The draft in ``paper/workshop_draft/main.tex`` currently hard-codes numbers from the
retracted v1/v2 results. Writing them by hand is how a paper ends up disagreeing with its
own repository, so the tables are generated instead and the draft should ``\\input`` them::

    \\input{results_tables}

Then ``\\PaperTableBenchmarks``, ``\\PaperTableCapacity``, ``\\PaperTableNoise``,
``\\PaperTableMemory`` and ``\\PaperTableRates`` are available as macros.

Usage::

    python scripts/make_paper_tables.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "paper" / "workshop_draft" / "results_tables.tex"

LABEL = {
    "photonic": "Photonic (recurrent)",
    "photonic_no_feedback": "Photonic (no feedback)",
    "esn": "Echo state network",
    "rff": "Random Fourier features",
    "poly": "Polynomial window",
    "classical_control": "Linear window (control)",
}
# Column headers for the tables that put models across the page; the full names in LABEL
# are too wide for ten-task tables and were part of what pushed them off the text block.
SHORT_LABEL = {
    "photonic": "Photonic",
    "photonic_no_feedback": "No fb.",
    "esn": "ESN",
    "rff": "RFF",
    "poly": "Poly.",
    "classical_control": "Linear",
}
DATASET_LABEL = {
    "narma5": "NARMA-5",
    "narma10": "NARMA-10",
    "narma20": "NARMA-20",
    "mackey_glass_h17": "Mackey-Glass ($h{=}17$)",
    "lorenz63": "Lorenz-63 ($h{=}20$)",
    "sp500_rv": "S\\&P 500 RV",
    "santa_fe": "Santa Fe laser",
    "channel_eq": "Channel eq.",
    "parity_d3": "Parity ($d{=}3$)",
    "henon": "H\\'enon ($h{=}4$)",
}


def _num(value: float, places: int = 4) -> str:
    if pd.isna(value):
        return "---"
    if value != 0 and abs(value) < 10 ** (-places):
        return f"\\num{{{value:.1e}}}"
    return f"{value:.{places}f}"


def _macro(name: str, body: str) -> str:
    return f"\\newcommand{{\\{name}}}{{%\n{body}\n}}\n\n"


def _feedback_disabled(dataset: str) -> bool:
    """True when the tuned configuration for this task already has feedback off."""
    path = RESULTS / "tuning" / f"{dataset}_photonic.json"
    if not path.exists():
        return False
    return json.loads(path.read_text())["params"].get("feedback") is False


def _all_raw() -> pd.DataFrame | None:
    """Every benchmarked seed on disk, not just the most recent run.

    ``run_benchmarks.py`` writes ``all_raw.csv`` covering the datasets *that invocation*
    was given, so running it on a subset silently shrinks the paper's main table -- it had
    been reduced to four of the ten tasks, dropping every NARMA and Lorenz-63. The
    per-dataset ``*_raw.csv`` files are the durable record, so build the union from those.
    """
    paths = sorted((RESULTS / "benchmarks").glob("*_raw.csv"))
    paths = [p for p in paths if p.name != "all_raw.csv"]
    if not paths:
        return None
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        if "dataset" not in frame.columns:
            frame["dataset"] = path.name[: -len("_raw.csv")]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def table_benchmarks() -> str:
    frame = _all_raw()
    if frame is None:
        return ""
    datasets = [d for d in DATASET_LABEL if d in set(frame.dataset)]
    models = [m for m in LABEL if m in set(frame.model)]

    # Median, not mean. On the chaotic tasks the seed distribution is heavy-tailed -- one
    # unlucky Lorenz-63 trajectory sends a model's mean to 3.5x its median and would decide
    # the ranking on a single draw. The interquartile range is quoted for the same reason.
    best = {
        d: frame[frame.dataset == d].groupby("model")["nrmse"].median().idxmin()
        for d in datasets
    }

    # Datasets down the page, models across it. With ten tasks the other orientation ran
    # 446pt past the text block -- an \hbox overflow wide enough to print off the paper.
    lines = [
        "\\footnotesize",
        "\\begin{tabular}{l" + "r" * len(models) + "}",
        "\\toprule",
        "Task & " + " & ".join(SHORT_LABEL[m] for m in models) + " \\\\",
        "\\midrule",
    ]
    for dataset in datasets:
        cells = []
        for model in models:
            sub = frame[(frame.dataset == dataset) & (frame.model == model)]["nrmse"]
            if not len(sub):
                cells.append("---")
                continue
            iqr = sub.quantile(0.75) - sub.quantile(0.25)
            # Below 1e-4 the task is solved and the spread across seeds is floating-point
            # noise, not a property of the model. Printing two significant figures of it
            # implies a precision that is not there, and the scientific-notation pair is
            # also what pushed this table past the text block.
            cell = (
                _num(sub.median())
                if sub.median() < 1e-4
                else f"{_num(sub.median())} ({_num(iqr)})"
            )
            # Where the search itself chose feedback=False, the no-feedback ablation is the
            # same model. Showing identical numbers without saying so looks like a bug.
            if model == "photonic_no_feedback" and _feedback_disabled(dataset):
                cell = f"{cell}$^{{\dagger}}$"
            if best.get(dataset) == model:
                cell = f"\\textbf{{{cell}}}"
            cells.append(cell)
        lines.append(f"{DATASET_LABEL[dataset]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return _macro("PaperTableBenchmarks", "\n".join(lines))


def table_dm() -> str:
    rows = []
    for dataset in DATASET_LABEL:
        path = RESULTS / "benchmarks" / f"{dataset}_stats.json"
        if not path.exists():
            continue
        dm = json.loads(path.read_text()).get("dm_vs_photonic", {})
        for model, stats in dm.items():
            rows.append((dataset, model, stats["stat"], stats["p_value"]))
    if not rows:
        return ""
    frame = pd.DataFrame(rows, columns=["dataset", "model", "stat", "p"])
    datasets = [d for d in DATASET_LABEL if d in set(frame.dataset)]
    models = [m for m in LABEL if m in set(frame.model)]

    lines = [
        "\\begin{tabular}{l" + "r" * len(models) + "}",
        "\\toprule",
        "Task & " + " & ".join(SHORT_LABEL[m] for m in models) + " \\\\",
        "\\midrule",
    ]
    for dataset in datasets:
        cells = []
        for model in models:
            sub = frame[(frame.dataset == dataset) & (frame.model == model)]
            cells.append(f"{sub['p'].iloc[0]:.4f}" if len(sub) else "---")
        lines.append(f"{DATASET_LABEL[dataset]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return _macro("PaperTableDM", "\n".join(lines))


def table_capacity() -> str:
    path = RESULTS / "capacity" / "capacity_narma10.csv"
    if not path.exists():
        return ""
    grouped = (
        pd.read_csv(path)
        .groupby(["family", "setting"])
        .agg(dim=("feature_dim", "mean"), nrmse=("nrmse", "mean"), std=("nrmse", "std"))
        .reset_index()
    )
    matched = grouped[(grouped.dim >= 500) & (grouped.dim <= 1500)].sort_values("nrmse")
    lines = ["\\begin{tabular}{lrr}", "\\toprule",
             "Model & Feature dim. & NRMSE \\\\", "\\midrule"]
    for _, row in matched.iterrows():
        lines.append(
            f"{LABEL.get(row.family, row.family)} & {int(row.dim)} & "
            f"{_num(row.nrmse)} $\\pm$ {_num(row['std'])} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return _macro("PaperTableCapacity", "\n".join(lines))


def table_quantumness() -> str:
    """Interference effect, with the paired test, generated rather than typed."""
    path = RESULTS / "quantumness" / "indistinguishability.csv"
    if not path.exists():
        return ""
    import numpy as np
    from scipy import stats as sstats

    frame = pd.read_csv(path)
    lines = ["\\begin{tabular}{lrrrrrl}", "\\toprule",
             "Task & $n$ & $\\mathcal{I}{=}0$ & $\\mathcal{I}{=}1$ & ratio & $p$ & verdict \\\\",
             "\\midrule"]
    for (dataset, n_photons), group in frame.groupby(["dataset", "n_photons"]):
        a = group[group.visibility == 0.0].sort_values("seed")["nrmse"].to_numpy()
        b = group[group.visibility == 1.0].sort_values("seed")["nrmse"].to_numpy()
        if len(a) != len(b) or len(a) < 2:
            continue
        diff = a - b
        spread = float(np.std(a, ddof=1))
        _, p_value = sstats.ttest_rel(a, b)
        detectable = p_value < 0.05 and abs(diff.mean()) > 0.5 * spread
        verdict = ("helps" if diff.mean() > 0 else "\\emph{hurts}") if detectable \
            else "no detectable effect"
        lines.append(
            f"{DATASET_LABEL.get(dataset, dataset)} & {int(n_photons)} & "
            f"{np.median(a):.4f} & {np.median(b):.4f} & {np.median(a) / np.median(b):.3f} & "
            f"{p_value:.3f} & {verdict} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return _macro("PaperTableQuantumness", "\n".join(lines))


def table_crossover() -> str:
    path = RESULTS / "crossover" / "crossover.csv"
    if not path.exists():
        return ""
    frame = pd.read_csv(path)
    lines = ["\\begin{tabular}{lrrrrr}", "\\toprule",
             "Platform & $n$ & Fock dim. & classical & device & $t$ needed \\\\",
             "\\midrule"]
    for platform, group in frame.groupby("platform"):
        for _, row in group[group.n_photons <= 6].iterrows():
            lines.append(
                f"{platform.capitalize()} & {int(row.n_photons)} & {int(row.fock_dim)} & "
                f"{row.classical_s * 1e3:.2f}\\,ms & \\num{{{row.device_s:.2e}}}\\,s & "
                f"{row.required_transmittance:.3f} \\\\"
            )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return _macro("PaperTableCrossover", "\n".join(lines))


def table_memory() -> str:
    path = RESULTS / "capacity" / "memory_ipc.csv"
    if not path.exists():
        return ""
    frame = pd.read_csv(path).groupby("system").mean(numeric_only=True)
    lines = ["\\begin{tabular}{lrrrr}", "\\toprule",
             "System & Dim. & Linear MC & IPC & IPC/feature \\\\", "\\midrule"]
    for name, row in frame.sort_values("ipc_per_feature", ascending=False).iterrows():
        lines.append(
            f"{name.replace('_', ' ')} & {int(row.feature_dim)} & {row.linear_mc:.1f} & "
            f"{row.ipc_total:.1f} & {row.ipc_per_feature:.2f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return _macro("PaperTableMemory", "\n".join(lines))


def table_noise() -> str:
    path = RESULTS / "noise" / "noise_narma10_all.csv"
    if not path.exists():
        return ""
    frame = pd.read_csv(path)
    blocks = []

    shots = frame[frame.sweep == "shots"]
    if not shots.empty:
        stats = shots.groupby("shots")["nrmse"].agg(["mean", "std"]).reset_index()
        infinity = "$\\infty$"
        rows = [
            f"{infinity if row.shots == 0 else int(row.shots)} & "
            f"{_num(row['mean'])} $\\pm$ {_num(row['std'])} \\\\"
            for _, row in stats.iterrows()
        ]
        blocks.append(
            "\\begin{tabular}{rr}\n\\toprule\nCoincidences/step & NRMSE \\\\\n\\midrule\n"
            + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}"
        )

    for sweep, column, header in [("indist", "indist", "Indistinguishability"),
                                  ("g2", "g2", "$g^{(2)}(0)$")]:
        sub = frame[frame.sweep == sweep]
        if sub.empty or column not in sub.columns:
            continue
        stats = sub.groupby(column)["nrmse"].mean().reset_index()
        rows = [f"{row[column]:g} & {_num(row['nrmse'])} \\\\" for _, row in stats.iterrows()]
        blocks.append(
            f"\\begin{{tabular}}{{rr}}\n\\toprule\n{header} & NRMSE \\\\\n\\midrule\n"
            + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}"
        )

    return _macro("PaperTableNoise", "\\hfill\n".join(blocks))


def table_rates() -> str:
    path = RESULTS / "noise" / "coincidence_rates.csv"
    if not path.exists():
        return ""
    frame = pd.read_csv(path)
    lines = ["\\begin{tabular}{lrrr}", "\\toprule",
             "Platform & $n$ & Coincidences/s & Time for $3\\times10^4$ \\\\", "\\midrule"]
    for _, row in frame.iterrows():
        seconds = 3e4 / row.coincidences_per_s
        pretty = f"{seconds:.2f}\\,s" if seconds < 90 else f"{seconds / 60:.0f}\\,min"
        lines.append(
            f"{row.platform.capitalize()} & {int(row.n_photons)} & "
            f"\\num{{{row.coincidences_per_s:.2e}}} & {pretty} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return _macro("PaperTableRates", "\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    header = (
        "% Generated by scripts/make_paper_tables.py -- do not edit by hand.\n"
        "% Regenerate after any results change:  python scripts/make_paper_tables.py\n"
        "% Requires siunitx for \\num.\n\n"
    )
    body = "".join(
        [table_benchmarks(), table_dm(), table_capacity(), table_memory(),
         table_noise(), table_rates(), table_quantumness(), table_crossover()]
    )
    Path(args.out).write_text(header + body)
    macros = [line.split("{")[1].split("}")[0] for line in body.splitlines()
              if line.startswith("\\newcommand")]
    print(f"wrote {args.out}")
    print("macros:", ", ".join("\\" + m for m in macros))


if __name__ == "__main__":
    main()
