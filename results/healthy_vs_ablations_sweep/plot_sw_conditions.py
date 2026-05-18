"""
Generate 5-panel figure: grouped bar charts (cue 1 vs cue 2) of mean SW distance
per ablation direction, split by overall + 4 angular separation bins.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

CSV_PATH = Path(__file__).parent / "sw_sweep.csv"
OUT_DIR = Path(__file__).parent

N_DIRS = 14
DIR_COLS = [f"sw_Healthy_vs_Ablated_{d}" for d in range(N_DIRS)]
ANGLE_BINS = [0, 45, 90, 135, 180]
ANGLE_LABELS = ["0–45°", "45–90°", "90–135°", "135–180°"]
CUE_COLORS = {"Cue 1": "#2166ac", "Cue 2": "#d6604d"}
BAR_WIDTH = 0.35
DIR11_COLOR = "lightgrey"

def load_and_prepare(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    diff = np.abs(df["color1_deg"] - df["color2_deg"])
    df["angle_sep_deg"] = np.minimum(diff, 360 - diff)
    df["angle_bin"] = pd.cut(
        df["angle_sep_deg"],
        bins=ANGLE_BINS,
        labels=ANGLE_LABELS,
        include_lowest=True,
        right=True,
    )
    return df


def compute_means(df: pd.DataFrame) -> dict:
    """Returns {condition_label: {cue: (means[14], sems[14])}}"""
    conditions = {"All angles": df}
    for label in ANGLE_LABELS:
        conditions[label] = df[df["angle_bin"] == label]

    result = {}
    for cond_name, subset in conditions.items():
        cue_stats = {}
        for cue_val, cue_label in [(1, "Cue 1"), (2, "Cue 2")]:
            rows = subset[subset["cue"] == cue_val]
            means = np.array([rows[c].mean() for c in DIR_COLS])
            sems = np.array([rows[c].sem() for c in DIR_COLS])
            cue_stats[cue_label] = (means, sems)
        result[cond_name] = cue_stats
    return result


def plot_panel(ax, cond_stats: dict, title: str, show_legend: bool = False):
    x = np.arange(N_DIRS)
    offsets = {"Cue 1": -BAR_WIDTH / 2, "Cue 2": BAR_WIDTH / 2}

    for cue_label, (means, sems) in cond_stats.items():
        offset = offsets[cue_label]
        colors = [
            DIR11_COLOR if d == 11 else CUE_COLORS[cue_label]
            for d in range(N_DIRS)
        ]
        bars = ax.bar(
            x + offset, means, BAR_WIDTH,
            color=colors,
            yerr=sems,
            error_kw=dict(elinewidth=0.8, capsize=2, ecolor="black", alpha=0.6),
            label=cue_label,
            alpha=0.85,
        )
        # Re-colour dir 11 bars to make them greyish but still distinguishable
        bars[11].set_color(DIR11_COLOR)
        bars[11].set_edgecolor("black")
        bars[11].set_linewidth(0.8)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation(labelOnlyBase=False))
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in range(N_DIRS)], fontsize=8)
    ax.set_xlabel("Ablation direction", fontsize=9)
    ax.set_ylabel("Mean SW distance (log)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4, which="both")
    ax.set_xlim(-0.6, N_DIRS - 0.4)

    if show_legend:
        # Custom legend: cue colours + dir 11 marker
        from matplotlib.patches import Patch
        handles = [
            Patch(facecolor=CUE_COLORS["Cue 1"], label="Cue 1"),
            Patch(facecolor=CUE_COLORS["Cue 2"], label="Cue 2"),
            Patch(facecolor=DIR11_COLOR, edgecolor="black", linewidth=0.8, label="Dir 11 (any cue)"),
        ]
        ax.legend(handles=handles, fontsize=8, loc="upper left")

    # Annotate dir 11 raw means above the bars
    c1_mean = cond_stats["Cue 1"][0][11]
    c2_mean = cond_stats["Cue 2"][0][11]
    y_top = ax.get_ylim()[1]
    ax.annotate(
        f"Dir 11\nC1:{c1_mean:.0f}\nC2:{c2_mean:.1f}",
        xy=(11, y_top * 0.6),
        fontsize=6,
        ha="center",
        color="dimgrey",
    )


def main():
    df = load_and_prepare(CSV_PATH)
    all_stats = compute_means(df)

    condition_order = ["All angles"] + ANGLE_LABELS
    condition_titles = [
        "All angle separations (both cues)",
        "Angle sep: 0–45°",
        "Angle sep: 45–90°",
        "Angle sep: 90–135°",
        "Angle sep: 135–180°",
    ]

    fig, axes = plt.subplots(
        5, 1,
        figsize=(14, 18),
        constrained_layout=True,
    )

    for i, (cond, title) in enumerate(zip(condition_order, condition_titles)):
        plot_panel(axes[i], all_stats[cond], title, show_legend=(i == 0))

    fig.suptitle(
        "Sliced Wasserstein distance: Healthy vs Ablated\nper direction, cue, and angular separation",
        fontsize=12, fontweight="bold"
    )

    out_path = OUT_DIR / "sw_by_cue_and_angle_bins.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")

    # Also save a wide version suitable for papers
    fig2, axes2 = plt.subplots(
        1, 5,
        figsize=(28, 5),
        constrained_layout=True,
    )
    for i, (cond, title) in enumerate(zip(condition_order, condition_titles)):
        plot_panel(axes2[i], all_stats[cond], title, show_legend=(i == 0))
    fig2.suptitle(
        "SW distance: Healthy vs Ablated — per direction, cue, and angular separation",
        fontsize=11, fontweight="bold"
    )
    out_wide = OUT_DIR / "sw_by_cue_and_angle_bins_wide.png"
    fig2.savefig(out_wide, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_wide}")


if __name__ == "__main__":
    main()
