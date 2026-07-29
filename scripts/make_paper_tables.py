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
DATASET_LABEL = {
    "narma5": "NARMA-5",
    "narma10": "NARMA-10",
    "narma20": "NARMA-20",
    "mackey_glass_h17": "Mackey-Glass ($h{=}17$)",
    "lorenz63": "Lorenz-63 ($h{=}20$)",
    "sp500_rv": "S\\&P 500 RV",
}


def _num(value: float, places: int = 4) -> str:
    if pd.isna(value):
        return "---"
    if value != 0 and abs(value) < 10 ** (-places):
        return f"\\num{{{value:.1e}}}"
    return f"{value:.{places}f}"


def _macro(name: str, body: str) -> str:
    return f"\\newcommand{{\\{name}}}{{%\n{body}\n}}\n\n"


def table_benchmarks() -> str:
    path = RESULTS / "benchmarks" / "all_raw.csv"
    if not path.exists():
        return ""
    frame = pd.read_csv(path)
    datasets = [d for d in DATASET_LABEL if d in set(frame.dataset)]
    models = [m for m in LABEL if m in set(frame.model)]

    # Bold the best mean per dataset.
    best = {
        d: frame[frame.dataset == d].groupby("model")["nrmse"].mean().idxmin() for d in datasets
    }

    lines = [
        "\\begin{tabular}{l" + "r" * len(datasets) + "}",
        "\\toprule",
        "Model & " + " & ".join(DATASET_LABEL[d] for d in datasets) + " \\\\",
        "\\midrule",
    ]
    for model in models:
        cells = []
        for dataset in datasets:
            sub = frame[(frame.dataset == dataset) & (frame.model == model)]["nrmse"]
            if not len(sub):
                cells.append("---")
                continue
            cell = f"{_num(sub.mean())} $\\pm$ {_num(sub.std())}"
            if best.get(dataset) == model:
                cell = f"\\textbf{{{cell}}}"
            cells.append(cell)
        lines.append(f"{LABEL[model]} & " + " & ".join(cells) + " \\\\")
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
        "\\begin{tabular}{l" + "r" * len(datasets) + "}",
        "\\toprule",
        "Baseline vs photonic & " + " & ".join(DATASET_LABEL[d] for d in datasets) + " \\\\",
        "\\midrule",
    ]
    for model in models:
        cells = []
        for dataset in datasets:
            sub = frame[(frame.dataset == dataset) & (frame.model == model)]
            cells.append(f"{sub['p'].iloc[0]:.4f}" if len(sub) else "---")
        lines.append(f"{LABEL[model]} & " + " & ".join(cells) + " \\\\")
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
         table_noise(), table_rates()]
    )
    Path(args.out).write_text(header + body)
    macros = [line.split("{")[1].split("}")[0] for line in body.splitlines()
              if line.startswith("\\newcommand")]
    print(f"wrote {args.out}")
    print("macros:", ", ".join("\\" + m for m in macros))


if __name__ == "__main__":
    main()
