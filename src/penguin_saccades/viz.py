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
from matplotlib.lines import Line2D

from penguin_saccades import constants as C
from penguin_saccades.saccades import triggered_windows, zero_offset_rows
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


def triggered_matrices(
    trajectories: pd.DataFrame, saccades: pd.DataFrame, yaw_col: str, vel_col: str, half_window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Velocity and zero-offset angle (deg) around each saccade peak, left turns mirrored.

    ``saccades`` and ``trajectories`` must belong to one data set; rows are
    matched per trajectory and segment.  Returns two ``(n_saccades, 2 *
    half_window + 1)`` matrices, empty when nothing matches.
    """
    mats_v: list[np.ndarray] = []
    mats_a: list[np.ndarray] = []
    for key, seg in trajectories.groupby([C.COL_TRAJECTORY, C.COL_SEGMENT]):
        if not seg[yaw_col].notna().any():
            continue
        s = saccades[(saccades[C.COL_TRAJECTORY] == key[0]) & (saccades[C.COL_SEGMENT] == key[1])]
        if s.empty:
            continue
        frames = seg[C.COL_FRAME].to_numpy()
        peaks = np.searchsorted(frames, s["peak_idx"].to_numpy(dtype=int))
        flip = (s["direction"] == C.DIRECTION_OF_NEGATIVE_VELOCITY).to_numpy()
        mats_v.append(triggered_windows(seg[vel_col].to_numpy(), peaks, half_window, flip))
        mats_a.append(zero_offset_rows(triggered_windows(np.rad2deg(seg[yaw_col].to_numpy()), peaks, half_window, flip)))
    width = 2 * half_window + 1
    if not mats_v:
        return np.empty((0, width)), np.empty((0, width))
    return np.vstack(mats_v), np.vstack(mats_a)


def pooled_triggered_matrices(
    trajectories: pd.DataFrame,
    saccades: pd.DataFrame,
    yaw_col: str,
    vel_col: str,
    window_s: float = C.TRIGGER_WINDOW_S,
    frame_rates: dict[str, float] = C.FRAME_RATES,
    target_fps: float = C.POOLED_FRAME_RATE_HZ,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Triggered matrices across data sets on one time grid.

    Each data set is windowed at its own frame rate and every row is then
    linearly resampled onto the grid of ``target_fps``.  Returns
    ``(mat_v, mat_a, dataset_per_row, time_s)``.
    """
    half_target = round(window_s * target_fps / 2)
    time_s = np.arange(-half_target, half_target + 1) / target_fps
    mats_v, mats_a, labels = [], [], []
    for dataset in sorted(saccades[C.COL_DATASET].unique()):
        fps = frame_rates[dataset]
        half = round(window_s * fps / 2)
        traj = trajectories[trajectories[C.COL_DATASET] == dataset]
        sacc = saccades[saccades[C.COL_DATASET] == dataset]
        mat_v, mat_a = triggered_matrices(traj, sacc, yaw_col, vel_col, half)
        if not mat_v.shape[0]:
            continue
        source_t = np.arange(-half, half + 1) / fps
        mats_v.append(resample_rows(mat_v, source_t, time_s))
        mats_a.append(resample_rows(mat_a, source_t, time_s))
        labels += [dataset] * mat_v.shape[0]
    if not mats_v:
        return np.empty((0, time_s.size)), np.empty((0, time_s.size)), np.array([], dtype=str), time_s
    return np.vstack(mats_v), np.vstack(mats_a), np.array(labels), time_s


def resample_rows(matrix: np.ndarray, source_t: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    """Linearly resample each row from ``source_t`` onto ``target_t``; outside the finite support stays NaN."""
    out = np.full((matrix.shape[0], target_t.size), np.nan)
    for i, row in enumerate(matrix):
        finite = np.isfinite(row)
        if finite.sum() < 2:
            continue
        out[i] = np.interp(target_t, source_t[finite], row[finite], left=np.nan, right=np.nan)
    return out


def plot_triggered_average(
    tables: list[pd.DataFrame],
    colours: dict[str, str],
    ylabel: str,
    ax: plt.Axes | None = None,
    ylim: tuple[float, float] | None = None,
) -> plt.Axes:
    """Overlay several triggered-average tables (from :func:`triggered_average_table`)."""
    if ax is None:
        _, ax = plt.subplots(figsize=C.FIGURE_SIZE_SINGLE)
    if ylim is not None:
        ax.set_ylim(*ylim)
    for table in tables:
        if table.empty:
            continue
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
    marker_col: str | None = None,
    markers: dict[str, str] = C.CAMERA_MARKERS,
) -> plt.Axes:
    """Raincloud plot of ``value_col`` per group with n in the tick labels.

    With ``marker_col`` the strip uses one marker shape per value of that
    column (the camera, by default), so pooled groups stay legible.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=C.FIGURE_SIZE_DOUBLE)
    order = order or sorted(df[group_col].dropna().unique())
    rng = np.random.default_rng(C.RANDOM_SEED)
    labels = []
    for i, group in enumerate(order):
        sub = df.loc[df[group_col] == group]
        values = sub[value_col].to_numpy(dtype=float)
        finite = np.isfinite(values)
        x = values[finite]
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
        jitter = i - 0.25 + rng.uniform(-0.1, 0.1, x.size)
        shapes = sub[marker_col].to_numpy()[finite] if marker_col else np.full(x.size, "o")
        for shape_key in np.unique(shapes):
            sel = shapes == shape_key
            ax.scatter(jitter[sel], x[sel], s=8, color=colour, alpha=0.6, lw=0,
                       marker=markers.get(shape_key, "o") if marker_col else "o")
        ax.boxplot([x], positions=[i], widths=0.12, showfliers=False, notch=x.size > 10,
                   medianprops={"color": "k"}, boxprops={"color": "k"}, whiskerprops={"color": "k"})
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
    return ax


def add_camera_legend(ax: plt.Axes, datasets: list[str], markers: dict[str, str] = C.CAMERA_MARKERS,
                      labels: dict[str, str] = C.CAMERA_LABELS, **kwargs: object) -> None:
    """Legend of marker shapes for the cameras present in a pooled panel."""
    handles = [Line2D([], [], ls="none", marker=markers[d], color="k", ms=4, label=labels.get(d, d)) for d in datasets]
    ax.legend(handles=handles, frameon=False, **kwargs)


# ┌────────────────────────────────────────────────────────────────────────────┐
# │ Interval chunks  « interval-triggered averages from the long chunk table » │
# └────────────────────────────────────────────────────────────────────────────┘
def plot_interval_chunks(chunks: pd.DataFrame, variables: dict[str, str], colours: dict[str, str],
                         ax: plt.Axes | None = None, ylim: tuple[float, float] | None = None,
                         frame_rates: dict[str, float] = C.FRAME_RATES,
                         target_fps: float = C.POOLED_FRAME_RATE_HZ) -> tuple[plt.Axes, pd.DataFrame]:
    """Median and CI per frame for each variable of the chunk table, all data sets pooled on one grid."""
    if ax is None:
        _, ax = plt.subplots(figsize=C.FIGURE_SIZE_SINGLE)
    tables = []
    for var, label in variables.items():
        matrix, time_s = pooled_chunk_matrix(chunks[chunks["variable"] == var], frame_rates, target_fps)
        if matrix.shape[0]:
            tables.append(triggered_average_table(matrix, time_s, label))
    plot_triggered_average(tables, colours, "value", ax=ax, ylim=ylim)
    ax.set_xlabel("time within interval chunk (s)")
    return ax, pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def pooled_chunk_matrix(
    chunks: pd.DataFrame, frame_rates: dict[str, float], target_fps: float, chunk_s: float = C.ISI_CHUNK_S
) -> tuple[np.ndarray, np.ndarray]:
    """One row per interval chunk of one variable, every data set resampled onto the ``target_fps`` grid."""
    size_t = round(chunk_s * target_fps)
    time_s = (np.arange(size_t) - size_t // 2) / target_fps
    rows = []
    for dataset, sub in chunks.groupby(C.COL_DATASET):
        if sub["value"].notna().sum() == 0:
            continue
        wide = sub.pivot_table(index=["interval_id", "chunk_id", C.COL_TRAJECTORY, C.COL_SEGMENT],
                               columns="frame", values="value")
        source_t = (wide.columns.to_numpy(dtype=float) - wide.shape[1] // 2) / frame_rates[dataset]
        rows.append(resample_rows(wide.to_numpy(dtype=float), source_t, time_s))
    if not rows:
        return np.empty((0, size_t)), time_s
    return np.vstack(rows), time_s


# ┌──────────────────────────────────────────────────────────────────────┐
# │ Yaw velocity histograms  « noise floor and threshold per data set »  │
# └──────────────────────────────────────────────────────────────────────┘
def plot_yaw_velocity_histograms(
    trajectories: pd.DataFrame,
    thresholds_degps: dict[str, float] = C.SACCADE_VELOCITY_THRESHOLDS_DEGPS,
    bins: tuple[float, float, int] = C.YAW_VELOCITY_HIST_BINS,
) -> tuple[plt.Figure, pd.DataFrame]:
    """One panel per (data set, species): |yaw velocity| of body and head on log-log axes.

    The saccade threshold of each body part is drawn in its colour.

    Returns the figure and the long table of bin edges and frame counts
    behind it, so the noise floor of each source can be read off.
    """
    edges = np.logspace(np.log10(bins[0]), np.log10(bins[1]), bins[2] + 1)
    groups = list(trajectories.groupby([C.COL_DATASET, C.COL_SPECIES], sort=True))
    fig, axes = plt.subplots(1, len(groups), figsize=(C.FIGURE_SIZE_SINGLE[0] * len(groups), C.FIGURE_SIZE_SINGLE[1]),
                             sharey=True, squeeze=False)
    rows: list[pd.DataFrame] = []
    for ax, ((dataset, species), frames) in zip(axes[0], groups):
        for part, col in (("body", C.COL_BODY_VEL), ("head", C.COL_HEAD_VEL)):
            speed = np.abs(frames[col].to_numpy(dtype=float))
            speed = speed[np.isfinite(speed)]
            if speed.size == 0:
                continue
            counts, _ = np.histogram(speed, bins=edges)
            threshold = thresholds_degps[part]
            above = int((speed > threshold).sum())
            ax.stairs(counts, edges, color=C.BODYPART_COLOURS[part], lw=1.2,
                      label=f"{part} (n = {speed.size}, {above} > {threshold:g})")
            ax.axvline(threshold, color=C.BODYPART_COLOURS[part], ls="--", lw=0.8)
            rows.append(pd.DataFrame({C.COL_DATASET: dataset, C.COL_SPECIES: species, "part": part,
                                      "bin_lo_degps": edges[:-1], "bin_hi_degps": edges[1:], "frames": counts}))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{dataset}: {species}")
        ax.set_xlabel(r"$|$yaw velocity$|$ ($^\circ\,s^{-1}$)")
        ax.legend(frameon=False, fontsize=C.FONT_SIZE - 2)
    axes[0, 0].set_ylabel("frames")
    fig.tight_layout()
    return fig, pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
