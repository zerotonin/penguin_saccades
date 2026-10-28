# ╔══════════════════════════════════════════════════════════════════╗
# ║  penguin_saccades — viz                                          ║
# ║  « every figure as SVG + PNG + CSV, Wong colours only »          ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Triggered averages with bootstrap bands, rainclouds instead of  ║
# ║  bar charts, interval chunk averages, and the save_figure        ║
# ║  helper that writes editable SVG, a PNG preview and the numbers  ║
# ║  behind the plot.                                                ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Figure helpers and the standard figure set."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from penguin_saccades import constants as C
from penguin_saccades.stats import bootstrap_median_ci


# ┌────────────────────────────────────────────────────────────┐
# │ Style and saving  « one look, three files per figure »     │
# └────────────────────────────────────────────────────────────┘
def apply_style() -> None:
    """Minimal lab style: small fonts, no top/right spines."""
    mpl.rcParams.update({
        "font.size": C.FONT_SIZE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 100,
    })


def save_figure(fig: plt.Figure, stem: str, output_dir: Path, data: pd.DataFrame | None = None) -> list[Path]:
    """Export ``fig`` as SVG (editable text) and PNG, plus the data as CSV.

    Args:
        fig:        Figure to save.
        stem:       File name without extension.
        output_dir: Target directory, created if missing.
        data:       Numbers behind the figure, written to ``<stem>.csv``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    with mpl.rc_context(C.SVG_PARAMS):
        svg = output_dir / f"{stem}.svg"
        fig.savefig(svg, format="svg", bbox_inches="tight")
        saved.append(svg)
    png = output_dir / f"{stem}.png"
    fig.savefig(png, format="png", dpi=C.FIGURE_DPI, bbox_inches="tight")
    saved.append(png)
    if data is not None:
        csv = output_dir / f"{stem}.csv"
        data.to_csv(csv, index=False)
        saved.append(csv)
    return saved


# ┌──────────────────────────────────────────────────────────────┐
# │ Triggered average  « mean trace with bootstrap median band » │
# └──────────────────────────────────────────────────────────────┘
def triggered_average_table(matrix: np.ndarray, time_s: np.ndarray, label: str) -> pd.DataFrame:
    """Column-wise median and bootstrap CI of a triggered-window matrix."""
    rows = []
    for j, t in enumerate(time_s):
        med, lo, hi = bootstrap_median_ci(matrix[:, j], n_boot=2000)
        rows.append({"label": label, "time_s": t, "median": med, "ci_lower": lo, "ci_upper": hi,
                     "n": int(np.isfinite(matrix[:, j]).sum())})
    return pd.DataFrame(rows)


def plot_triggered_average(
    tables: list[pd.DataFrame], colours: dict[str, str], ylabel: str, ax: plt.Axes | None = None
) -> plt.Axes:
    """Overlay several triggered-average tables (from :func:`triggered_average_table`)."""
    if ax is None:
        _, ax = plt.subplots(figsize=C.FIGURE_SIZE_SINGLE)
    for table in tables:
        label = table["label"].iloc[0]
        colour = colours.get(label, WONG_FALLBACK)
        ax.plot(table["time_s"], table["median"], color=colour, label=f"{label} (n = {int(table['n'].max())})")
        ax.fill_between(table["time_s"], table["ci_lower"], table["ci_upper"], color=colour, alpha=0.25, lw=0)
    ax.axvline(0, color="k", lw=0.5, ls=":")
    ax.set_xlabel("time relative to saccade peak (s)")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    return ax


WONG_FALLBACK: str = C.WONG["black"]


# ┌────────────────────────────────────────────────────────────┐
# │ Raincloud  « half violin, jittered strip, box »            │
# └────────────────────────────────────────────────────────────┘
def plot_raincloud(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    colours: dict[str, str],
    ylabel: str,
    order: list[str] | None = None,
    log_y: bool = False,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Raincloud plot of ``value_col`` per group with n in the tick labels."""
    if ax is None:
        _, ax = plt.subplots(figsize=C.FIGURE_SIZE_DOUBLE)
    order = order or sorted(df[group_col].dropna().unique())
    rng = np.random.default_rng(C.RANDOM_SEED)
    labels = []
    for i, group in enumerate(order):
        x = df.loc[df[group_col] == group, value_col].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        labels.append(f"{group}\n(n = {x.size})")
        if x.size == 0:
            continue
        colour = colours.get(group, WONG_FALLBACK)
        if x.size > 1 and np.ptp(x) > 0:
            parts = ax.violinplot([x], positions=[i], showextrema=False, widths=0.8, side="high")
            for body in parts["bodies"]:
                body.set_facecolor(colour)
                body.set_edgecolor("none")
                body.set_alpha(0.4)
        ax.scatter(i - 0.25 + rng.uniform(-0.1, 0.1, x.size), x, s=6, color=colour, alpha=0.6, lw=0)
        ax.boxplot([x], positions=[i], widths=0.12, showfliers=False, notch=x.size > 10,
                   medianprops={"color": "k"}, boxprops={"color": "k"}, whiskerprops={"color": "k"})
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
    return ax


# ┌────────────────────────────────────────────────────────────────────────────┐
# │ Interval chunks  « interval-triggered averages from the long chunk table » │
# └────────────────────────────────────────────────────────────────────────────┘
def plot_interval_chunks(chunks: pd.DataFrame, frame_rate_hz: float, variables: dict[str, str],
                         colours: dict[str, str], ax: plt.Axes | None = None) -> tuple[plt.Axes, pd.DataFrame]:
    """Median and CI per frame for each variable of the chunk table."""
    if ax is None:
        _, ax = plt.subplots(figsize=C.FIGURE_SIZE_SINGLE)
    tables = []
    for var, label in variables.items():
        sub = chunks[chunks["variable"] == var]
        if sub.empty:
            continue
        wide = sub.pivot_table(index=["interval_id", "chunk_id", C.COL_TRAJECTORY, C.COL_SEGMENT],
                               columns="frame", values="value")
        time_s = (wide.columns.to_numpy() - wide.shape[1] // 2) / frame_rate_hz
        tables.append(triggered_average_table(wide.to_numpy(), time_s, label))
    plot_triggered_average(tables, colours, "value", ax=ax)
    ax.set_xlabel("time within interval chunk (s)")
    return ax, pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
