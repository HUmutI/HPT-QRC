"""Publication figures from the result CSVs.

One figure per claim:

``capacity``   NRMSE against feature dimension -- answers "is it the optics or just more
               features?", which is the first thing a reviewer asks.
``noise``      NRMSE against each noise source, with the measured hardware operating points
               marked -- the robustness result.
``benchmark``  Per-dataset model comparison with seed spread.
``rate``       Coincidence rate against photon number -- why the hardware runs use two.

Colour is assigned per model family in a fixed order and never cycled, so a family keeps
its hue across every figure. The palette is the validated categorical default; because
several of its slots fall below 3:1 against the surface, every series carries a direct
label as well as a legend entry, so identity is never colour-alone.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "results" / "figures"

# Fixed categorical order. A family keeps its hue in every figure.
FAMILY_COLOR = {
    "photonic": "#2a78d6",
    "photonic_no_feedback": "#eb6834",
    "esn": "#1baf7a",
    "rff": "#eda100",
    "poly": "#e87ba4",
    "classical_control": "#4a3aa7",
}
LABEL = {
    "photonic": "Photonic (recurrent)",
    "photonic_no_feedback": "Photonic (no feedback)",
    "esn": "Echo state network",
    "rff": "Random Fourier features",
    "poly": "Polynomial window",
    "classical_control": "Linear window (control)",
}

TEXT = "#0b0b0b"
MUTED = "#52514e"


def _style(ax):
    """Recessive axes and grid: the data should be the only assertive thing on the page."""
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d5d4cf")
    ax.grid(True, color="#ececea", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(MUTED)


def fig_capacity(dataset: str = "narma10") -> None:
    path = ROOT / "results" / "capacity" / f"capacity_{dataset}.csv"
    if not path.exists():
        print(f"skip capacity: {path} missing")
        return
    frame = pd.read_csv(path)
    grouped = (
        frame.groupby(["family", "setting"])
        .agg(dim=("feature_dim", "mean"), nrmse=("nrmse", "mean"), std=("nrmse", "std"))
        .reset_index()
        .sort_values("dim")
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    _style(ax)
    for family, sub in grouped.groupby("family"):
        sub = sub.sort_values("dim")
        color = FAMILY_COLOR.get(family, MUTED)
        # The polynomial settings vary window *and* degree, so ordering them by dimension
        # is not a sweep along one axis. Joining them with a line would assert a trend that
        # does not exist, so that family is drawn as points only.
        connected = family != "poly"
        if connected:
            ax.plot(sub["dim"], sub["nrmse"], color=color, linewidth=2, marker="o",
                    markersize=5, markeredgecolor="white", markeredgewidth=1,
                    label=LABEL.get(family, family))
            ax.fill_between(sub["dim"], sub["nrmse"] - sub["std"].fillna(0),
                            sub["nrmse"] + sub["std"].fillna(0), color=color,
                            alpha=0.10, lw=0)
        else:
            ax.scatter(sub["dim"], sub["nrmse"], color=color, s=34, zorder=3,
                       edgecolor="white", linewidth=1, label=LABEL.get(family, family))
        anchor = sub.loc[sub["nrmse"].idxmin()] if not connected else sub.iloc[-1]
        ax.annotate(LABEL.get(family, family), (anchor["dim"], anchor["nrmse"]),
                    textcoords="offset points", xytext=(6, -2 if connected else -12),
                    fontsize=8, color=color, va="center")

    ax.set_xscale("log")
    ax.set_xlabel("Feature dimension", color=TEXT, fontsize=10)
    ax.set_ylabel("NRMSE (lower is better)", color=TEXT, fontsize=10)
    ax.set_title(f"{dataset}: accuracy against feature dimension", color=TEXT,
                 fontsize=11, loc="left", pad=12)
    ax.set_xlim(right=grouped["dim"].max() * 3.2)
    ax.legend(frameon=False, fontsize=8, loc="upper right", labelcolor=MUTED)
    fig.tight_layout()
    out = FIGS / f"capacity_{dataset}.png"
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def fig_noise(dataset: str = "narma10") -> None:
    path = ROOT / "results" / "noise" / f"noise_{dataset}_all.csv"
    if not path.exists():
        print(f"skip noise: {path} missing")
        return
    frame = pd.read_csv(path)
    reference = frame[frame.sweep == "reference"]
    noiseless = float(reference[reference.spec == "noiseless"]["nrmse"].iloc[0])
    control = float(reference[reference.spec == "classical_control"]["nrmse"].iloc[0])

    panels = [
        ("shots", "shots", "Coincidences per timestep", True),
        ("indist", "indist", "Photon indistinguishability (HOM visibility)", False),
        ("g2", "g2", "Second-order correlation $g^{(2)}(0)$", False),
    ]
    available = [p for p in panels if (frame.sweep == p[0]).any()]
    if not available:
        print("skip noise: no sweeps present")
        return

    fig, axes = plt.subplots(1, len(available), figsize=(4.6 * len(available), 4.2))
    axes = np.atleast_1d(axes)

    for ax, (sweep, xcol, xlabel, logx) in zip(axes, available):
        _style(ax)
        sub = frame[frame.sweep == sweep].copy()
        if sweep == "shots":
            sub = sub[sub.shots > 0]
        stats = sub.groupby(xcol)["nrmse"].agg(["mean", "std"]).reset_index()

        ax.axhline(noiseless, color=MUTED, linestyle="--", linewidth=1.2)
        ax.annotate("noiseless", (0.02, noiseless), xycoords=("axes fraction", "data"),
                    fontsize=8, color=MUTED, va="bottom")
        ax.axhline(control, color="#4a3aa7", linestyle=":", linewidth=1.4)
        ax.annotate("classical control", (0.02, control),
                    xycoords=("axes fraction", "data"), fontsize=8, color="#4a3aa7",
                    va="bottom")

        ax.plot(stats[xcol], stats["mean"], color="#2a78d6", linewidth=2, marker="o",
                markersize=5, markeredgecolor="white", markeredgewidth=1)
        ax.fill_between(stats[xcol], stats["mean"] - stats["std"].fillna(0),
                        stats["mean"] + stats["std"].fillna(0), color="#2a78d6",
                        alpha=0.12, lw=0)

        # Mark the measured hardware operating points on the axis they belong to.
        if sweep == "indist":
            for value, name, color in [(0.8636, "Ascella", "#eb6834"),
                                       (0.827, "Belenos", "#1baf7a")]:
                ax.axvline(value, color=color, linewidth=1.2, alpha=0.8)
                # Anchored low: the reference lines carry labels along the top edge.
                ax.annotate(name, (value, 0.04), xycoords=("data", "axes fraction"),
                            rotation=90, fontsize=8, color=color, va="bottom", ha="right")
        if sweep == "g2":
            for value, name, color in [(0.0195, "Ascella", "#eb6834"),
                                       (0.182, "Belenos", "#1baf7a")]:
                ax.axvline(value, color=color, linewidth=1.2, alpha=0.8)
                # Anchored low: the reference lines carry labels along the top edge.
                ax.annotate(name, (value, 0.04), xycoords=("data", "axes fraction"),
                            rotation=90, fontsize=8, color=color, va="bottom", ha="right")

        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel, color=TEXT, fontsize=10)
        ax.set_ylabel("NRMSE", color=TEXT, fontsize=10)

    axes[0].set_title(f"{dataset}: robustness to hardware noise", color=TEXT, fontsize=11,
                      loc="left", pad=12)
    fig.tight_layout()
    out = FIGS / f"noise_{dataset}.png"
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def fig_benchmark() -> None:
    path = ROOT / "results" / "benchmarks" / "all_raw.csv"
    if not path.exists():
        print(f"skip benchmark: {path} missing")
        return
    frame = pd.read_csv(path)
    datasets = list(dict.fromkeys(frame.dataset))
    families = [f for f in FAMILY_COLOR if f in set(frame.model)]

    fig, axes = plt.subplots(1, len(datasets), figsize=(3.5 * len(datasets), 4.2))
    axes = np.atleast_1d(axes)
    for ax, dataset in zip(axes, datasets):
        _style(ax)
        sub = frame[frame.dataset == dataset]
        stats = sub.groupby("model")["nrmse"].agg(["mean", "std"]).reindex(families).dropna(
            subset=["mean"]
        )
        positions = np.arange(len(stats))
        ax.bar(positions, stats["mean"], yerr=stats["std"].fillna(0).values, capsize=3,
               color=[FAMILY_COLOR[m] for m in stats.index], width=0.68,
               error_kw=dict(ecolor=MUTED, lw=1))
        ax.set_xticks(positions)
        ax.set_xticklabels([LABEL[m].replace(" (", "\n(") for m in stats.index],
                           rotation=45, ha="right", fontsize=7.5)
        ax.set_title(dataset, color=TEXT, fontsize=10, loc="left")
        if np.nanmax(stats["mean"]) / max(np.nanmin(stats["mean"]), 1e-9) > 50:
            ax.set_yscale("log")
        ax.set_ylabel("NRMSE", color=TEXT, fontsize=9)
    fig.tight_layout()
    out = FIGS / "benchmark.png"
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def fig_rate() -> None:
    from src.noise import ASCELLA, BELENOS, coincidence_rate

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    _style(ax)
    photons = [1, 2, 3, 4, 5]
    for spec, color in ((ASCELLA, "#eb6834"), (BELENOS, "#1baf7a")):
        rates = [coincidence_rate(spec, n) for n in photons]
        ax.plot(photons, rates, color=color, linewidth=2, marker="o", markersize=6,
                markeredgecolor="white", markeredgewidth=1, label=spec.name.capitalize())
        ax.annotate(spec.name.capitalize(), (photons[-1], rates[-1]),
                    textcoords="offset points", xytext=(6, 0), fontsize=8, color=color,
                    va="center")
    # A 5-minute job spread over 1000 timesteps needs this many coincidences per second
    # to reach 1000 shots per step; below it, the shot budget is the binding constraint.
    ax.axhline(1000 * 1000 / 300, color=MUTED, linestyle="--", linewidth=1.2)
    ax.annotate("1000 shots/step over a 5-min job", (0.02, 1000 * 1000 / 300),
                xycoords=("axes fraction", "data"), fontsize=8, color=MUTED, va="bottom")
    ax.set_yscale("log")
    ax.set_xticks(photons)
    ax.set_xlabel("Photon number $n$", color=TEXT, fontsize=10)
    ax.set_ylabel("Detected $n$-fold coincidences per second", color=TEXT, fontsize=10)
    ax.set_title("Why the hardware runs use two photons", color=TEXT, fontsize=11,
                 loc="left", pad=12)
    ax.set_xlim(right=5.6)
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED)
    fig.tight_layout()
    out = FIGS / "coincidence_rate.png"
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def fig_feedback() -> None:
    """NRMSE against feedback gain, which is what the binary ablation could not show.

    Two things are visible here that a flag comparison hides: the contribution of the
    recurrence is strongly task-dependent, and every task shares a cliff at roughly the same
    gain, where the state stops contracting and the reservoir loses its fading memory.
    """
    path = ROOT / "results" / "feedback" / "feedback_strength.csv"
    if not path.exists():
        print("  (no feedback sweep yet)")
        return
    frame = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    _style(ax)
    colors = {"narma5": "#eb6834", "narma10": "#1baf7a", "narma20": "#3b6ea5",
              "mackey_glass_h17": "#9a6fb0"}
    labels = {"narma5": "NARMA-5", "narma10": "NARMA-10", "narma20": "NARMA-20",
              "mackey_glass_h17": "Mackey-Glass"}
    for dataset, sub in frame.groupby("dataset"):
        med = sub.groupby("g_fb").nrmse.median().sort_index()
        # The ablation sits at gain 0, which a log axis cannot show. Plot it as a separate
        # marker on the left edge rather than dropping it or faking a small positive value.
        ablation = med.get(0.0)
        positive = med[med.index > 0]
        color = colors.get(dataset, MUTED)
        ax.plot(positive.index, positive.values, color=color, linewidth=2, marker="o",
                markersize=5, markeredgecolor="white", markeredgewidth=1,
                label=labels.get(dataset, dataset))
        if ablation is not None:
            ax.plot([positive.index.min() * 0.55], [ablation], marker="s", markersize=6,
                    color=color, markeredgecolor="white", markeredgewidth=1)

    # Shaded from 0.6, where every task has reached NRMSE ~1 -- i.e. is predicting no better
    # than the target's mean. The onset is task-dependent (Mackey-Glass is already two orders
    # of magnitude off by 0.3, NARMA-10 is not), so shading from 0.3 would assert a shared
    # threshold that the curves do not show.
    ax.axvspan(0.6, 3.0, color="#f2dede", alpha=0.45, zorder=0)
    ax.annotate("no fading memory\n(NRMSE $\\approx$ 1 on every task)", (0.62, 0.60),
                xycoords=("data", "axes fraction"), fontsize=8, color="#9c3b34", va="top")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Feedback gain $g_{fb}$  (squares at left: no feedback)", color=TEXT,
                  fontsize=10)
    ax.set_ylabel("Test NRMSE", color=TEXT, fontsize=10)
    ax.set_title("What the recurrence is worth, task by task", color=TEXT, fontsize=11,
                 loc="left", pad=12)
    # Lower left is where Mackey-Glass sits; a legend there covers its ablation marker.
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=MUTED)
    fig.tight_layout()
    out = FIGS / "feedback_strength.png"
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="+",
                    default=["capacity", "noise", "benchmark", "rate", "feedback"])
    ap.add_argument("--dataset", default="narma10")
    args = ap.parse_args()

    FIGS.mkdir(parents=True, exist_ok=True)
    if "capacity" in args.which:
        fig_capacity(args.dataset)
    if "noise" in args.which:
        fig_noise(args.dataset)
    if "benchmark" in args.which:
        fig_benchmark()
    if "rate" in args.which:
        fig_rate()
    if "feedback" in args.which:
        fig_feedback()


if __name__ == "__main__":
    main()
