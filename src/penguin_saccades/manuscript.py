# ╔══════════════════════════════════════════════════════════════════╗
# ║  penguin_saccades — manuscript                                   ║
# ║  « the paper's figures and Table 1 from the pipeline tables »    ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  One function per manuscript figure, each returning the figure   ║
# ║  and the long table behind it. Every figure is species-based:    ║
# ║  both cameras are pooled, the slower one resampled onto the      ║
# ║  faster grid and the camera kept as marker shape. Triggered      ║
# ║  averages share one pair of y-limits. Rainclouds replace the box ║
# ║  plots of the submitted version; Wong colours replace red/blue.  ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Manuscript figures (Fig. 1E, 2, 3, 4, 5, supplements) and Table 1."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from penguin_saccades import constants as C
from penguin_saccades import viz
from penguin_saccades.stats import summarise, two_sample_fisher

COMBINED_COLOUR: str = C.WONG["black"]
SHADE_COLOUR: str = C.WONG["yellow"]
TYPE_LABELS: dict[str, str] = {"free_head": "free head", "associated": "associated head", "free_body": "body"}
DURATION_BIN_S: float = 0.1
AMPLITUDE_BIN_DEG: float = 5.0
HEAD_LEAD_BIN_S: float = 0.2
POOLED_LABEL: str = "both cameras"
YLABEL_VELOCITY = r"yaw velocity ($^\circ\,s^{-1}$)"
YLABEL_ANGLE = r"yaw ($^\circ$)"


# ┌────────────────────────────────────────────────────────────┐
# │ Table 1  « duration summaries of saccades and intervals »  │
# └────────────────────────────────────────────────────────────┘
def table_1(saccades: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    """Median, mean, max duration (ms) with CI, SEM and n per data set, species, movement type and body part.

    Saccades appear once per type (associated, free) and once pooled per
    body part; intervals once per bounding class.  Censored intervals are
    included, as in the submitted version.
    """
    sacc = saccades.assign(duration_ms=saccades["duration_s"] * 1000.0,
                           movement_type=saccades[C.COL_SACCADE_TYPE].map(_movement_type),
                           bodypart=saccades[C.COL_SOURCE])
    pooled = sacc.assign(movement_type="saccade")
    ints = intervals.assign(duration_ms=intervals["duration_s"] * 1000.0, movement_type="ISI",
                            bodypart=intervals[C.COL_INTERVAL_CLASS].map(_interval_bodypart))
    keys = [C.COL_DATASET, C.COL_SPECIES, "movement_type", "bodypart"]
    table = pd.concat([summarise(_with_pooled_cameras(df), "duration_ms", keys) for df in (sacc, pooled, ints)],
                      ignore_index=True)
    table = table.rename(columns={"median": "median_duration_ms", "mean": "mean_duration_ms",
                                  "max": "max_duration_ms", "ci_lower": "lower_95_ci", "ci_upper": "upper_95_ci"})
    cols = [*keys, "median_duration_ms", "mean_duration_ms", "max_duration_ms", "lower_95_ci", "upper_95_ci", "sem", "n"]
    return table[cols].sort_values(keys).reset_index(drop=True)


def _with_pooled_cameras(df: pd.DataFrame) -> pd.DataFrame:
    """Append a copy of every species seen by more than one camera under the data set label ``POOLED_LABEL``."""
    cameras = df.groupby(C.COL_SPECIES)[C.COL_DATASET].nunique()
    multi = df[df[C.COL_SPECIES].isin(cameras.index[cameras > 1])]
    return pd.concat([df, multi.assign(**{C.COL_DATASET: POOLED_LABEL})], ignore_index=True)


def _movement_type(saccade_type: str) -> str:
    return "associated" if saccade_type == "associated" else "free"


def _interval_bodypart(interval_class: str) -> str:
    return "head+body" if interval_class == "any" else interval_class


# ┌────────────────────────────────────────────────────────────┐
# │ Fig. 1E  « example traces with head saccades shaded »      │
# └────────────────────────────────────────────────────────────┘
def figure_example_traces(
    trajectories: pd.DataFrame, saccades: pd.DataFrame, species: str = "gentoo"
) -> tuple[plt.Figure, pd.DataFrame]:
    """Yaw velocity, yaw and translation of body and head over the densest run of head saccades of a species."""
    head = saccades[(saccades[C.COL_SPECIES] == species) & (saccades[C.COL_SOURCE] == "head")]
    if head.empty:
        raise ValueError(f"No head saccades for species {species!r}.")
    seg_key, t0 = _densest_window(head)
    dataset = seg_key[0]
    fps = C.FRAME_RATES[dataset]
    traj = trajectories[trajectories[C.COL_DATASET] == dataset]
    seg = traj[(traj[C.COL_TRAJECTORY] == seg_key[1]) & (traj[C.COL_SEGMENT] == seg_key[2])]
    window = seg[(seg[C.COL_TIME] >= t0) & (seg[C.COL_TIME] <= t0 + C.EXAMPLE_WINDOW_S)].copy()
    window["time_in_window_s"] = window[C.COL_TIME] - t0
    shown = head[(head[C.COL_DATASET] == dataset) & (head[C.COL_TRAJECTORY] == seg_key[1])
                 & (head[C.COL_SEGMENT] == seg_key[2])]
    fig, axes = plt.subplots(3, 1, figsize=(C.FIGURE_SIZE_DOUBLE[0], 2.4 * C.FIGURE_SIZE_SINGLE[1]), sharex=True)
    t = window["time_in_window_s"].to_numpy()
    for ax, cols, ylabel in (
        (axes[0], {"body": C.COL_BODY_VEL, "head": C.COL_HEAD_VEL}, YLABEL_VELOCITY),
        (axes[1], {"body": C.COL_BODY_YAW, "head": C.COL_HEAD_YAW}, YLABEL_ANGLE),
        (axes[2], {"body": C.COL_TRANS}, r"translational speed (m$\,s^{-1}$)"),
    ):
        for part, col in cols.items():
            y = window[col].to_numpy(dtype=float)
            if col in (C.COL_BODY_YAW, C.COL_HEAD_YAW):
                y = np.rad2deg(y - y[0])
            ax.plot(t, y, color=C.BODYPART_COLOURS[part] if len(cols) > 1 else COMBINED_COLOUR, label=part)
        for _, s in shown.iterrows():
            ax.axvspan(s["start_idx"] / fps - t0, s["stop_idx"] / fps - t0, color=SHADE_COLOUR, alpha=0.4, lw=0)
        ax.set_ylabel(ylabel)
    axes[0].legend(frameon=False)
    axes[0].axhline(0, color="k", lw=0.5, ls=":")
    axes[2].set_xlabel("time (s)")
    axes[2].set_xlim(0, C.EXAMPLE_WINDOW_S)
    fig.suptitle(f"{species}, {C.CAMERA_LABELS.get(dataset, dataset)}: trajectory {seg_key[1]}, segment {seg_key[2]}",
                 fontsize=C.FONT_SIZE)
    fig.tight_layout()
    return fig, window.assign(**{C.COL_DATASET: dataset})


def _densest_window(head: pd.DataFrame) -> tuple[tuple[str, int, int], float]:
    """(data set, trajectory, segment) and window start holding the most head saccade peaks within ``EXAMPLE_WINDOW_S``."""
    best_key, best_t0, best_n = None, 0.0, -1
    for key, grp in head.groupby([C.COL_DATASET, C.COL_TRAJECTORY, C.COL_SEGMENT]):
        peaks = np.sort(grp["peak_time_s"].to_numpy(dtype=float))
        starts = np.maximum(peaks - 0.5, 0.0)
        counts = np.searchsorted(peaks, starts + C.EXAMPLE_WINDOW_S, side="right") - np.arange(peaks.size)
        i = int(np.argmax(counts))
        if counts[i] > best_n:
            best_key, best_t0, best_n = (str(key[0]), int(key[1]), int(key[2])), float(starts[i]), int(counts[i])
    return best_key, best_t0


# ┌────────────────────────────────────────────────────────────┐
# │ Fig. 2  « body saccades per species and combined »         │
# └────────────────────────────────────────────────────────────┘
def figure_body_saccades(
    trajectories: pd.DataFrame, saccades: pd.DataFrame, dataset: str = C.DATASET_OPEN.name
) -> tuple[plt.Figure, pd.DataFrame]:
    """A/B triggered yaw velocity and yaw, C duration and D amplitude histograms of body saccades."""
    fps = C.FRAME_RATES[dataset]
    half = round(C.TRIGGER_WINDOW_S * fps / 2)
    time_s = np.arange(-half, half + 1) / fps
    traj = trajectories[trajectories[C.COL_DATASET] == dataset]
    body = saccades[(saccades[C.COL_DATASET] == dataset) & (saccades[C.COL_SOURCE] == "body")]
    groups = {sp: body[body[C.COL_SPECIES] == sp] for sp in sorted(body[C.COL_SPECIES].unique())}
    if len(groups) > 1:
        groups["combined"] = body
    colours = {**C.SPECIES_COLOURS, "combined": COMBINED_COLOUR}
    fig, axes = plt.subplots(2, 2, figsize=(C.FIGURE_SIZE_DOUBLE[0], 2 * C.FIGURE_SIZE_SINGLE[1]))
    tables: list[pd.DataFrame] = []
    tv, ta = [], []
    for label, sacc in groups.items():
        sub = traj[traj[C.COL_SPECIES].isin(sacc[C.COL_SPECIES].unique())]
        mat_v, mat_a = viz.triggered_matrices(sub, sacc, C.COL_BODY_YAW, C.COL_BODY_VEL, half)
        if mat_v.shape[0]:
            tv.append(viz.triggered_average_table(mat_v, time_s, label))
            ta.append(viz.triggered_average_table(mat_a, time_s, label))
    viz.plot_triggered_average(tv, colours, YLABEL_VELOCITY, ax=axes[0, 0], ylim=C.TRIGGERED_YLIM["velocity"])
    viz.plot_triggered_average(ta, colours, YLABEL_ANGLE, ax=axes[0, 1], ylim=C.TRIGGERED_YLIM["angle"])
    tables += [t.assign(panel="A") for t in tv] + [t.assign(panel="B") for t in ta]
    dur_edges = np.arange(0, C.MAX_SACCADE_DURATION_S + DURATION_BIN_S, DURATION_BIN_S)
    amp_max = max(float(body["amplitude_deg"].abs().max()) if len(body) else 0.0, AMPLITUDE_BIN_DEG)
    amp_edges = np.arange(0, amp_max + AMPLITUDE_BIN_DEG, AMPLITUDE_BIN_DEG)
    for label, sacc in groups.items():
        tables.append(_probability_histogram(axes[1, 0], sacc["duration_s"], dur_edges, colours[label], label).assign(panel="C"))
        tables.append(_probability_histogram(axes[1, 1], sacc["amplitude_deg"].abs(), amp_edges, colours[label], label).assign(panel="D"))
    axes[1, 0].set_xlabel("saccade duration (s)")
    axes[1, 1].set_xlabel(r"amplitude ($^\circ$)")
    for ax in axes[1]:
        ax.set_ylabel("probability")
    axes[1, 0].legend(frameon=False)
    _letter_panels(axes.ravel())
    fig.tight_layout()
    return fig, pd.concat(tables, ignore_index=True)


# ┌─────────────────────────────────────────────────────────────────────┐
# │ Fig. 3  « head and body around associated and free head saccades »  │
# └─────────────────────────────────────────────────────────────────────┘
def figure_head_body_coordination(
    trajectories: pd.DataFrame, saccades: pd.DataFrame, species: str = "gentoo"
) -> tuple[plt.Figure, pd.DataFrame]:
    """Head-saccade triggered yaw velocity and yaw of head and body, one row per saccade type.

    Both cameras are pooled on the grid of ``POOLED_FRAME_RATE_HZ``; the
    row title gives the number of saccades per camera.
    """
    traj = trajectories[trajectories[C.COL_SPECIES] == species]
    head = saccades[(saccades[C.COL_SPECIES] == species) & (saccades[C.COL_SOURCE] == "head")]
    types = [t for t in ("associated", "free_head") if (head[C.COL_SACCADE_TYPE] == t).any()]
    fig, axes = plt.subplots(len(types), 2, figsize=(C.FIGURE_SIZE_DOUBLE[0], len(types) * C.FIGURE_SIZE_SINGLE[1]),
                             squeeze=False)
    tables: list[pd.DataFrame] = []
    for row, sacc_type in zip(axes, types):
        sacc = head[head[C.COL_SACCADE_TYPE] == sacc_type]
        tv, ta = [], []
        for part, yaw_col, vel_col in (("body", C.COL_BODY_YAW, C.COL_BODY_VEL), ("head", C.COL_HEAD_YAW, C.COL_HEAD_VEL)):
            mat_v, mat_a, cameras, time_s = viz.pooled_triggered_matrices(traj, sacc, yaw_col, vel_col)
            if mat_v.shape[0]:
                counts = _camera_counts(pd.Series(cameras))
                tv.append(viz.triggered_average_table(mat_v, time_s, part).assign(cameras=counts))
                ta.append(viz.triggered_average_table(mat_a, time_s, part).assign(cameras=counts))
        viz.plot_triggered_average(tv, C.BODYPART_COLOURS, YLABEL_VELOCITY, ax=row[0], ylim=C.TRIGGERED_YLIM["velocity"])
        viz.plot_triggered_average(ta, C.BODYPART_COLOURS, YLABEL_ANGLE, ax=row[1], ylim=C.TRIGGERED_YLIM["angle"])
        row[0].set_title(f"{TYPE_LABELS[sacc_type]} saccades\n({_camera_counts(sacc[C.COL_DATASET])})", loc="left",
                         fontsize=C.FONT_SIZE)
        tables += [t.assign(saccade_type=sacc_type, panel="velocity") for t in tv]
        tables += [t.assign(saccade_type=sacc_type, panel="angle") for t in ta]
    _letter_panels(axes.ravel())
    fig.tight_layout()
    return fig, pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def _camera_counts(datasets: pd.Series) -> str:
    """``"open water n = 159, closed pool n = 2"`` for the cameras present."""
    counts = datasets.value_counts()
    return ", ".join(f"{C.CAMERA_LABELS.get(d, d)} n = {int(counts[d])}" for d in sorted(counts.index))


# ┌─────────────────────────────────────────────────────────────┐
# │ Fig. 4  « saccade types: duration, peak speed, head lead »  │
# └─────────────────────────────────────────────────────────────┘
def figure_saccade_types(saccades: pd.DataFrame, species: str = "gentoo") -> tuple[plt.Figure, pd.DataFrame]:
    """A/B rainclouds of duration and peak speed per type; C/D duration histograms; E/F head-lead histograms.

    Both cameras are pooled; in the rainclouds the marker shape gives the camera.
    """
    sacc = saccades[saccades[C.COL_SPECIES] == species].copy()
    sacc["type_label"] = sacc[C.COL_SACCADE_TYPE].map(TYPE_LABELS)
    sacc.loc[sacc[C.COL_SOURCE] == "body", "type_label"] = "body"
    sacc["duration_ms"] = sacc["duration_s"] * 1000.0
    sacc["abs_peak_speed_degps"] = sacc["peak_speed_degps"].abs()
    order = [lab for lab in ("free head", "associated head", "body") if (sacc["type_label"] == lab).any()]
    colours = {"free head": C.SACCADE_TYPE_COLOURS["free_head"], "associated head": C.SACCADE_TYPE_COLOURS["associated"],
               "body": C.SACCADE_TYPE_COLOURS["free_body"]}
    fig, axes = plt.subplots(3, 2, figsize=(C.FIGURE_SIZE_DOUBLE[0], 3 * C.FIGURE_SIZE_SINGLE[1]))
    viz.plot_raincloud(sacc, "duration_ms", "type_label", colours, "saccade duration (ms)", order=order, log_y=True,
                       ax=axes[0, 0], marker_col=C.COL_DATASET)
    viz.plot_raincloud(sacc, "abs_peak_speed_degps", "type_label", colours, r"peak yaw velocity ($^\circ\,s^{-1}$)",
                       order=order, log_y=True, ax=axes[0, 1], marker_col=C.COL_DATASET)
    viz.add_camera_legend(axes[0, 1], sorted(sacc[C.COL_DATASET].unique()), loc="upper right", fontsize=C.FONT_SIZE - 2)
    tables = [sacc[[C.COL_DATASET, "type_label", "duration_ms", "abs_peak_speed_degps"]].assign(panel="AB")]
    dur_edges = np.arange(0, C.MAX_SACCADE_DURATION_S + DURATION_BIN_S, DURATION_BIN_S)
    for ax, part in zip(axes[1], ("body", "head")):
        part_sacc = sacc[sacc[C.COL_SOURCE] == part]
        tables.append(_probability_histogram(ax, part_sacc["duration_s"], dur_edges, C.BODYPART_COLOURS[part], part)
                      .assign(panel="CD"))
        ax.set_xlabel(f"{part} saccade duration (s)")
        ax.set_ylabel("probability")
    head = sacc[sacc[C.COL_SOURCE] == "head"]
    body = sacc[sacc[C.COL_SOURCE] == "body"]
    lead_edges = np.arange(-C.ASSOCIATION_WINDOW_S, C.ASSOCIATION_WINDOW_S + HEAD_LEAD_BIN_S, HEAD_LEAD_BIN_S)
    free_lead = nearest_body_lag(head[head[C.COL_SACCADE_TYPE] == "free_head"], body)
    assoc_lead = head.loc[head[C.COL_SACCADE_TYPE] == "associated", "head_lead_s"]
    for ax, lead, sacc_type in zip(axes[2], (free_lead, assoc_lead), ("free_head", "associated")):
        tables.append(_probability_histogram(ax, lead, lead_edges, C.SACCADE_TYPE_COLOURS[sacc_type], TYPE_LABELS[sacc_type])
                      .assign(panel="EF"))
        ax.set_xlabel("body peak minus head peak (s)\n← body first        head first →")
        ax.set_ylabel("probability")
        ax.axvline(0, color="k", lw=0.5, ls=":")
        ax.legend(frameon=False)
    _letter_panels(axes.ravel())
    fig.tight_layout()
    return fig, pd.concat(tables, ignore_index=True)


def nearest_body_lag(head: pd.DataFrame, body: pd.DataFrame, window_s: float = C.ASSOCIATION_WINDOW_S) -> pd.Series:
    """Body peak minus head peak to the nearest body saccade of any direction within the window.

    Free head saccades have no same-direction partner by definition; this
    is the lag to whichever body saccade is closest, NaN when none lies
    within ``window_s``.
    """
    keys = [C.COL_DATASET, C.COL_TRAJECTORY, C.COL_SEGMENT]
    lags = pd.Series(np.nan, index=head.index, dtype=float)
    for key, h_grp in head.groupby(keys):
        b_grp = body[(body[keys] == pd.Series(key, index=keys)).all(axis=1)]
        if b_grp.empty:
            continue
        for i, h in h_grp.iterrows():
            dt = b_grp["peak_time_s"] - h["peak_time_s"]
            j = dt.abs().idxmin()
            if abs(dt[j]) <= window_s:
                lags[i] = dt[j]
    return lags


# ┌────────────────────────────────────────────────────────────────┐
# │ Fig. 5  « saccade durations against intersaccadic intervals »  │
# └────────────────────────────────────────────────────────────────┘
def figure_durations_vs_intervals(saccades: pd.DataFrame, intervals: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame]:
    """A body saccades vs body intervals per species; B saccade types vs their intervals (gentoo).

    Species are pooled across cameras, the marker shape giving the camera.
    Each saccade group is compared with its interval group by a Fisher
    resampling test on the median; the p-values are in the returned table.
    """
    cols = [C.COL_DATASET, "duration_ms"]
    sacc = saccades.assign(duration_ms=lambda d: d["duration_s"] * 1000.0)
    ints = intervals.assign(duration_ms=lambda d: d["duration_s"] * 1000.0)
    panel_a = []
    for sp in sorted(sacc[C.COL_SPECIES].unique()):
        panel_a.append((f"{sp} saccade", sacc.loc[(sacc[C.COL_SPECIES] == sp) & (sacc[C.COL_SOURCE] == "body"), cols],
                        C.SPECIES_COLOURS[sp]))
        panel_a.append((f"{sp} ISI", ints.loc[(ints[C.COL_SPECIES] == sp) & (ints[C.COL_INTERVAL_CLASS] == "body"), cols],
                        C.SACCADE_TYPE_COLOURS["interval"]))
    head_species = sorted(sacc.loc[sacc[C.COL_SOURCE] == "head", C.COL_SPECIES].unique())
    panel_b = []
    for sp in head_species:
        sp_sacc, sp_int = sacc[sacc[C.COL_SPECIES] == sp], ints[ints[C.COL_SPECIES] == sp]
        for sacc_type, int_class in (("free_head", "head"), ("associated", "head"), ("free_body", "body")):
            sel = (sp_sacc[C.COL_SOURCE] == "body" if sacc_type == "free_body"
                   else (sp_sacc[C.COL_SACCADE_TYPE] == sacc_type) & (sp_sacc[C.COL_SOURCE] == "head"))
            panel_b.append((f"{TYPE_LABELS[sacc_type]}", sp_sacc.loc[sel, cols], C.SACCADE_TYPE_COLOURS[sacc_type]))
            panel_b.append((f"{TYPE_LABELS[sacc_type]} ISI", sp_int.loc[sp_int[C.COL_INTERVAL_CLASS] == int_class, cols],
                            C.SACCADE_TYPE_COLOURS["interval"]))
    fig, axes = plt.subplots(1, 2 if panel_b else 1, figsize=C.FIGURE_SIZE_DOUBLE, squeeze=False)
    tables = [_paired_rainclouds(axes[0, 0], panel_a).assign(panel="A")]
    if panel_b:
        tables.append(_paired_rainclouds(axes[0, 1], panel_b).assign(panel="B"))
    viz.add_camera_legend(axes[0, 0], sorted(sacc[C.COL_DATASET].unique()), loc="upper left", fontsize=C.FONT_SIZE - 2)
    _letter_panels(axes.ravel())
    fig.tight_layout()
    return fig, pd.concat(tables, ignore_index=True)


def _paired_rainclouds(ax: plt.Axes, groups: list[tuple[str, pd.DataFrame, str]]) -> pd.DataFrame:
    """Rainclouds of consecutive (saccade, interval) pairs with a Fisher test bracket over each pair.

    Each group is a frame with ``duration_ms`` and the data set column.
    """
    df = pd.concat([values.assign(group=label) for label, values, _ in groups], ignore_index=True)
    order = [label for label, _, _ in groups]
    viz.plot_raincloud(df, "duration_ms", "group", {label: colour for label, _, colour in groups}, "duration (ms)",
                       order=order, log_y=True, ax=ax, marker_col=C.COL_DATASET)
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")
    rows = []
    for i in range(0, len(groups) - 1, 2):
        a, b = groups[i][1]["duration_ms"].dropna(), groups[i + 1][1]["duration_ms"].dropna()
        p = two_sample_fisher(a, b) if len(a) > 1 and len(b) > 1 else np.nan
        rows.append({"saccade_group": groups[i][0], "interval_group": groups[i + 1][0], "n_saccades": len(a),
                     "n_intervals": len(b), "p_value": p})
        if np.isfinite(p):
            y = max(float(a.max()), float(b.max())) * 1.3
            ax.plot([i, i + 1], [y, y], color="k", lw=0.8)
            ax.text(i + 0.5, y, significance_label(p), ha="center", va="bottom", fontsize=C.FONT_SIZE)
    return df.merge(pd.DataFrame(rows), how="left", left_on="group", right_on="saccade_group")


# ┌────────────────────────────────────────────────────────────┐
# │ Supplement  « interval-triggered rotation »                │
# └────────────────────────────────────────────────────────────┘
def figure_interval_chunks(chunks: pd.DataFrame, species: str) -> tuple[plt.Figure, pd.DataFrame]:
    """Median yaw velocity of body and head across equal-length chunks of intersaccadic intervals, cameras pooled."""
    sub = chunks[chunks[C.COL_SPECIES] == species]
    fig, ax = plt.subplots(figsize=C.FIGURE_SIZE_SINGLE)
    _, table = viz.plot_interval_chunks(sub, {"body_yaw_speed_degps": "body", "head_yaw_speed_degps": "head"},
                                        C.BODYPART_COLOURS, ax=ax, ylim=C.TRIGGERED_YLIM["velocity"])
    ax.set_ylabel(YLABEL_VELOCITY)
    ax.set_title(f"{species} ({_camera_counts(sub.drop_duplicates([C.COL_DATASET, 'interval_id', 'chunk_id'])[C.COL_DATASET])} chunks)",
                 loc="left", fontsize=C.FONT_SIZE)
    fig.tight_layout()
    return fig, table


# ┌────────────────────────────────────────────────────────────┐
# │ Helpers  « histograms, panel letters, stars »              │
# └────────────────────────────────────────────────────────────┘
def _probability_histogram(ax: plt.Axes, values: pd.Series, edges: np.ndarray, colour: str, label: str) -> pd.DataFrame:
    x = values.to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    counts, _ = np.histogram(x, bins=edges)
    prob = counts / counts.sum() if counts.sum() else counts.astype(float)
    ax.stairs(prob, edges, color=colour, lw=1.2, label=f"{label} (n = {x.size})")
    return pd.DataFrame({"label": label, "bin_lo": edges[:-1], "bin_hi": edges[1:], "count": counts, "probability": prob})


def _letter_panels(axes: np.ndarray) -> None:
    for letter, ax in zip("ABCDEFGH", axes):
        ax.text(-0.15, 1.05, letter, transform=ax.transAxes, fontweight="bold", fontsize=C.FONT_SIZE + 2)


def significance_label(p: float) -> str:
    """Star convention of the submitted version."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."
