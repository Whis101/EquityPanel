"""The headline eGFR chart: share of patients below each clinical line, 2009 vs 2021.

Needs matplotlib: pip install "equitypanel[report]". Builds a Figure directly (no pyplot),
so it works without a display and without touching global matplotlib state.
"""

import pandas as pd

from equitypanel.reclassification.reclassify import LINES, RACE_TERM_COL

SYNTHETIC_LABEL = "Synthetic patients (Synthea). Not real-world rates."
LINE_LABELS = {
    "ckd_lt60": "eGFR < 60\nCKD threshold",
    "referral_lt30": "eGFR < 30\ncommon referral line",
    "waitlist_le20": "eGFR ≤ 20\ntransplant waiting-time line",
}
PANELS = ("Black", "non-Black")
YEAR_COLORS = {"2009": "#2a78d6", "2021": "#eb6834"}  # categorical slots 1 and 2
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
BAR_WIDTH = 0.36


def headline_chart(summary: pd.DataFrame):
    """Two panels (Black, non-Black patients); per line, the % below it under 2009 and 2021.

    summary is a reclassification_summary() frame with race_term rows. Each bar is
    labelled with its count, so the chart reads without color. Returns a Figure.
    """
    from matplotlib.figure import Figure

    rows = summary[summary["group_col"] == RACE_TERM_COL].set_index("group")
    missing = [p for p in PANELS if p not in rows.index]
    if missing:
        raise ValueError(f"summary has no race_term row(s) for {missing}")

    fig = Figure(figsize=(11, 5.2), dpi=150, facecolor="white")
    axes = fig.subplots(1, 2, sharey=True)
    xs = range(len(LINES))
    for ax, panel in zip(axes, PANELS, strict=True):
        row = rows.loc[panel]
        n = int(row["n"])
        for offset, year in ((-BAR_WIDTH / 2, "2009"), (BAR_WIDTH / 2, "2021")):
            counts = [int(row[f"{line}_{year}"]) for line in LINES]
            shares = [100 * c / n for c in counts]
            bars = ax.bar(
                [x + offset for x in xs],
                shares,
                width=BAR_WIDTH - 0.02,  # small gap between neighbouring bars
                color=YEAR_COLORS[year],
                label=f"CKD-EPI {year}" + (" (race-based)" if year == "2009" else " (race-free)"),
                zorder=2,
            )
            for bar, count, share in zip(bars, counts, shares, strict=True):
                ax.annotate(
                    f"{share:.0f}%\n({count:,})",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=TEXT_PRIMARY,
                )
        newly = int(row["waitlist_le20_newly_below"]) - int(row["waitlist_le20_newly_above"])
        sign = "+" if newly >= 0 else "−"
        ax.set_title(
            f"{panel} patients (n = {n:,})\nnet change at eGFR ≤ 20: {sign}{abs(newly):,}",
            fontsize=11,
            color=TEXT_PRIMARY,
            loc="left",
        )
        ax.set_xticks(list(xs), [LINE_LABELS[line] for line in LINES], fontsize=9)
        ax.tick_params(colors=TEXT_SECONDARY, length=0)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(TEXT_SECONDARY)
    axes[0].set_ylabel("% of patients below the line", color=TEXT_SECONDARY)
    top = max(ax.get_ylim()[1] for ax in axes)
    axes[0].set_ylim(0, top * 1.15)  # room for the bar labels (sharey applies it to both)
    axes[1].legend(frameon=False, fontsize=9, loc="upper right")

    fig.suptitle(
        "Removing race from eGFR changes who crosses clinical lines",
        x=0.01,
        ha="left",
        fontsize=13,
        color=TEXT_PRIMARY,
    )
    fig.text(0.01, 0.905, SYNTHETIC_LABEL, ha="left", fontsize=9.5, color=TEXT_SECONDARY)
    fig.subplots_adjust(top=0.78, bottom=0.15, left=0.07, right=0.99, wspace=0.08)
    return fig
