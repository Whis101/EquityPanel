"""Report charts as inline SVG strings. Needs matplotlib ([report] extra).

Figures are built with matplotlib.figure.Figure directly (no pyplot), so nothing
needs a screen. Text is kept as SVG text (not paths) and escaped by matplotlib.
"""

import io
import re

import pandas as pd

SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
METRIC_TITLES = {
    "auc": "AUC by group (higher is better; 0.5 = chance)",
    "fnr": "False negative rate by group (readmitted but not flagged; lower is better)",
    "fpr": "False positive rate by group (flagged but not readmitted)",
    "o_e": "Observed / expected readmissions (1 = calibrated)",
}


def figure_to_svg(fig) -> str:
    """Serialise a Figure to an SVG string without the XML prolog, ready to inline."""
    from matplotlib import rc_context

    buf = io.StringIO()
    with rc_context({"svg.fonttype": "none", "svg.hashsalt": "equitypanel"}):
        fig.savefig(buf, format="svg", bbox_inches="tight")
    svg = buf.getvalue()
    svg = re.sub(r"^<\?xml[^>]*>\s*", "", svg)
    svg = re.sub(r"^<!DOCTYPE[^>]*>\s*", "", svg)
    return re.sub(r"<metadata>.*?</metadata>\s*", "", svg, flags=re.DOTALL)


def _style(ax) -> None:
    ax.tick_params(colors=TEXT_SECONDARY, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(TEXT_SECONDARY)


def group_ci_chart(metrics: pd.DataFrame, group_col: str, metric: str) -> str:
    """Dot + 95% CI bar per group for one metric. Small groups get a hollow dot and '(small)'."""
    from matplotlib.figure import Figure

    rows = metrics[(metrics["group_col"] == group_col) & (metrics["metric"] == metric)]
    if rows.empty:
        raise ValueError(f"no {metric!r} rows for group column {group_col!r}")
    rows = rows.iloc[::-1]  # matplotlib draws bottom-up; keep table order top-down
    labels = [
        f"{g} (small)" if small else str(g)
        for g, small in zip(rows["group"], rows["small_group"], strict=True)
    ]
    ys = list(range(len(rows)))

    fig = Figure(figsize=(7, 0.45 * len(rows) + 1.1), dpi=100, facecolor="white")
    ax = fig.subplots()
    color = SERIES[0]
    for y, (_, row) in zip(ys, rows.iterrows(), strict=True):
        if pd.notna(row["ci_low"]) and pd.notna(row["ci_high"]):
            ax.plot([row["ci_low"], row["ci_high"]], [y, y], color=color, linewidth=2, zorder=2)
        if pd.notna(row["value"]):
            ax.scatter(
                [row["value"]],
                [y],
                s=64,
                zorder=3,
                color="white" if row["small_group"] else color,
                edgecolors=color,
                linewidths=2,
            )
            ax.annotate(
                f"{row['value']:.3f}",
                (row["value"], y),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=TEXT_PRIMARY,
            )
    ax.set_yticks(ys, labels, fontsize=9)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_title(METRIC_TITLES.get(metric, metric), fontsize=10, color=TEXT_PRIMARY, loc="left")
    ax.set_xlabel("point estimate with 95% CI", fontsize=8, color=TEXT_SECONDARY)
    _style(ax)
    return figure_to_svg(fig)


def calibration_chart(calibration: pd.DataFrame, group_col: str) -> str:
    """Mean predicted vs observed readmission rate per bin, one line per group."""
    from matplotlib.figure import Figure

    rows = calibration[calibration["group_col"] == group_col]
    if rows.empty:
        raise ValueError(f"no calibration rows for group column {group_col!r}")
    groups = list(dict.fromkeys(rows["group"]))
    if len(groups) > len(SERIES):
        raise ValueError(f"at most {len(SERIES)} groups can be drawn, got {len(groups)}")

    fig = Figure(figsize=(6, 4.2), dpi=100, facecolor="white")
    ax = fig.subplots()
    top = float(max(rows["mean_pred"].max(), rows["obs_rate"].max()) * 1.05)
    ax.plot([0, top], [0, top], color=TEXT_SECONDARY, linewidth=1, linestyle="--", zorder=1)
    for i, group in enumerate(groups):
        part = rows[rows["group"] == group].sort_values("mean_pred")
        ax.plot(
            part["mean_pred"],
            part["obs_rate"],
            color=SERIES[i],
            linewidth=2,
            marker="o",
            markersize=4,
            label=str(group),
            zorder=2,
        )
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel("mean predicted risk in bin", fontsize=9, color=TEXT_SECONDARY)
    ax.set_ylabel("observed readmission rate", fontsize=9, color=TEXT_SECONDARY)
    ax.set_title("Calibration (dashed line = perfect)", fontsize=10, color=TEXT_PRIMARY, loc="left")
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _style(ax)
    return figure_to_svg(fig)
