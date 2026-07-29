"""Render the current results into a single markdown report.

Replaces an earlier version that wrote to a hard-coded absolute path under a defunct tool
session directory and embedded literal hackathon-era numbers (including an "ESN NRMSE
29.7064", produced by a broken normalisation) in a template string. Everything here is read
from the result CSVs.

Usage::

    python scripts/render_report.py                 # -> results/REPORT.md
    python scripts/render_report.py --out foo.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

LABEL = {
    "photonic": "Photonic (recurrent)",
    "photonic_no_feedback": "Photonic (no feedback)",
    "esn": "Echo state network",
    "rff": "Random Fourier features",
    "poly": "Polynomial window",
    "classical_control": "Linear window (control)",
}


def _fmt(value: float) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:.4f}" if abs(value) >= 1e-3 else f"{value:.2e}"


def section_benchmarks() -> str:
    path = RESULTS / "benchmarks" / "all_raw.csv"
    if not path.exists():
        return "## Benchmarks\n\n_No benchmark results found._\n"
    frame = pd.read_csv(path)
    out = ["## Benchmarks", "", "NRMSE, mean ± std over seeds. Lower is better.", ""]

    datasets = list(dict.fromkeys(frame.dataset))
    out += ["| Model | " + " | ".join(datasets) + " |", "|" + "---|" * (len(datasets) + 1)]
    for model in LABEL:
        if model not in set(frame.model):
            continue
        cells = []
        for dataset in datasets:
            sub = frame[(frame.dataset == dataset) & (frame.model == model)]["nrmse"]
            cells.append(f"{_fmt(sub.mean())} ± {_fmt(sub.std())}" if len(sub) else "—")
        out.append(f"| {LABEL[model]} | " + " | ".join(cells) + " |")
    out.append("")

    for dataset in datasets:
        stats_path = RESULTS / "benchmarks" / f"{dataset}_stats.json"
        if not stats_path.exists():
            continue
        dm = json.loads(stats_path.read_text()).get("dm_vs_photonic", {})
        if dm:
            pairs = ", ".join(f"{LABEL.get(k, k)} p={v['p_value']:.4f}" for k, v in dm.items())
            out.append(f"- **{dataset}** — Diebold-Mariano vs photonic: {pairs}")
    out.append("")
    return "\n".join(out)


def section_capacity() -> str:
    path = RESULTS / "capacity" / "capacity_narma10.csv"
    if not path.exists():
        return ""
    grouped = (
        pd.read_csv(path)
        .groupby(["family", "setting"])
        .agg(dim=("feature_dim", "mean"), nrmse=("nrmse", "mean"))
        .reset_index()
    )
    matched = grouped[(grouped.dim >= 500) & (grouped.dim <= 1500)].sort_values("nrmse")
    out = ["## Matched capacity (NARMA-10)", "",
           "The tuned photonic configuration uses more features than the baselines, so this",
           "compares every family in a common dimension range.", "",
           "| Model | dim | NRMSE |", "|---|---|---|"]
    for _, row in matched.iterrows():
        out.append(f"| {LABEL.get(row.family, row.family)} | {int(row.dim)} | {_fmt(row.nrmse)} |")
    out.append("")
    return "\n".join(out)


def section_memory() -> str:
    path = RESULTS / "capacity" / "memory_ipc.csv"
    if not path.exists():
        return ""
    frame = pd.read_csv(path).groupby("system").mean(numeric_only=True)
    out = ["## Memory and information processing capacity", "",
           "| System | dim | Linear MC | IPC total | IPC per feature |",
           "|---|---|---|---|---|"]
    for name, row in frame.sort_values("ipc_per_feature", ascending=False).iterrows():
        out.append(
            f"| {name} | {int(row.feature_dim)} | {row.linear_mc:.1f} | "
            f"{row.ipc_total:.1f} | {row.ipc_per_feature:.2f} |"
        )
    out.append("")
    return "\n".join(out)


def section_noise() -> str:
    path = RESULTS / "noise" / "noise_narma10_all.csv"
    if not path.exists():
        return ""
    frame = pd.read_csv(path)
    out = ["## Noise robustness", ""]
    for _, row in frame[frame.sweep == "reference"].iterrows():
        out.append(f"- {row.spec}: NRMSE {_fmt(row.nrmse)}")
    out.append("")

    for sweep, column, title in [("shots", "shots", "Coincidences per timestep"),
                                 ("indist", "indist", "Indistinguishability"),
                                 ("g2", "g2", "g2(0)")]:
        sub = frame[frame.sweep == sweep]
        if sub.empty or column not in sub.columns:
            continue
        stats = sub.groupby(column)["nrmse"].agg(["mean", "std"]).reset_index()
        out += [f"### {title}", "", f"| {title} | NRMSE |", "|---|---|"]
        for _, row in stats.iterrows():
            value = "∞" if sweep == "shots" and row[column] == 0 else f"{row[column]:g}"
            spread = f" ± {row['std']:.4f}" if pd.notna(row["std"]) else ""
            out.append(f"| {value} | {row['mean']:.4f}{spread} |")
        out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS / "REPORT.md"))
    args = ap.parse_args()

    parts = [
        "# Results report",
        "",
        "Generated from the result CSVs by `scripts/render_report.py`. See `README.md` for",
        "interpretation and caveats.",
        "",
        section_benchmarks(),
        section_capacity(),
        section_memory(),
        section_noise(),
    ]
    text = "\n".join(p for p in parts if p)
    Path(args.out).write_text(text)
    print(f"wrote {args.out} ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
