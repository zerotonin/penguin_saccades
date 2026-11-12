# ╔══════════════════════════════════════════════════════════════════╗
# ║  penguin_saccades — qc                                           ║
# ║  « is the detector losing data, and where? »                     ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Where frames go between a pose file and the saccade table: one  ║
# ║  count per gate and movie, keypoint score and yaw-jitter tables, ║
# ║  the noise floor under the saccade thresholds, the reasons the   ║
# ║  artefact filter rejects raw peaks, and a per-movie timeline of  ║
# ║  detected, usable and bridged frames. Numbers first, figures     ║
# ║  second: every plot here has a CSV twin.                         ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Detection quality control: frame budget, scores, jitter, rejections, timelines."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from penguin_saccades import constants as C
from penguin_saccades.ingest_closed import (
    _yaw_angles,
    bridge_implausible_frames,
    gate_likelihood,
    skeleton_is_plausible,
)
from penguin_saccades.ingest_open import mask_outside_water
from penguin_saccades.kinematics import split_segments
from penguin_saccades.saccades import bridged_window_share, detect_saccades, trigger_half_window
from penguin_saccades.viz import plot_raincloud

WONG_FALLBACK = C.WONG["black"]


# ┌────────────────────────────────────────────────────────────┐
# │ Frame budget  « one count per gate, per movie »            │
# └────────────────────────────────────────────────────────────┘
def frame_budget(pose: pd.DataFrame, spec: C.DatasetSpec, score_threshold: float) -> dict[str, int]:
    """Frames surviving each gate of the pose route, in pipeline order.

    Args:
        pose:            One instance per frame (SLEAP or DeepLabCut table).
        spec:            Data set the pose belongs to; the water ROI applies
                         to the open-water sets only.
        score_threshold: Keypoint score below which a point is dropped.
    """
    scores = pose[[f"{kp}_likelihood" for kp in C.DLC_KEYPOINTS]] >= score_threshold
    head_tail, head_tip = C.HEAD_VECTOR
    body_tail, body_tip = C.BODY_VECTOR
    counts = {
        "frames": len(pose),
        "detected": int(pose[f"{head_tail}_x"].notna().sum()),
        "head_points": int(scores[[f"{head_tail}_likelihood", f"{head_tip}_likelihood"]].all(axis=1).sum()),
        "body_points": int(scores[[f"{body_tail}_likelihood", f"{body_tip}_likelihood"]].all(axis=1).sum()),
        "all_points": int(scores.all(axis=1).sum()),
    }
    gated = gate_likelihood(pose, threshold=score_threshold)
    if spec.name == C.DATASET_OPEN.name:
        gated = mask_outside_water(gated, spec)
    head_yaw, body_yaw = _yaw_angles(gated)
    both = np.isfinite(head_yaw) & np.isfinite(body_yaw)
    counts["both_yaws"] = int(both.sum())
    plausible = skeleton_is_plausible(gated, head_yaw, body_yaw) & both
    counts["plausible"] = int(plausible.sum())
    head_yaw, body_yaw = _yaw_angles(bridge_implausible_frames(gated, plausible))
    valid = np.isfinite(head_yaw) & np.isfinite(body_yaw)
    counts["valid"] = int(valid.sum())
    min_len = round(C.MIN_SEGMENT_S * spec.frame_rate_hz)
    counts["in_segments"] = int(sum(len(s) for s in split_segments(np.where(valid)[0], 0, min_len)))
    return counts


def keypoint_scores(pose: pd.DataFrame, movie: str) -> pd.DataFrame:
    """Long table of keypoint scores on detected frames: movie, keypoint, score."""
    detected = pose[f"{C.HEAD_VECTOR[0]}_x"].notna()
    parts = [
        pd.DataFrame({"movie": movie, "keypoint": kp, "score": pose.loc[detected, f"{kp}_likelihood"].to_numpy()})
        for kp in C.DLC_KEYPOINTS
    ]
    return pd.concat(parts, ignore_index=True)


def yaw_jitter(pose: pd.DataFrame, score_threshold: float, frame_rate_hz: float) -> dict[str, float]:
    """Frame-to-frame yaw noise of the raw skeleton, from frames where every point passes the gate.

    The MAD of the one-frame yaw change is the noise a Gaussian-smoothed
    velocity trace has to beat; ``*_jump_share`` is the share of frame
    pairs whose yaw changes by more than ``C.QC_YAW_JUMP_DEG`` in one
    frame, which a bird cannot do and a keypoint swap does.
    """
    ok = (pose[[f"{kp}_likelihood" for kp in C.DLC_KEYPOINTS]] >= score_threshold).all(axis=1).to_numpy()
    pair = ok[:-1] & ok[1:]
    out: dict[str, float] = {"gated_pairs": int(pair.sum())}
    for part, (tail, tip) in (("head", C.HEAD_VECTOR), ("body", C.BODY_VECTOR)):
        dx = (pose[f"{tip}_x"] - pose[f"{tail}_x"]).to_numpy(dtype=float)
        dy = (pose[f"{tip}_y"] - pose[f"{tail}_y"]).to_numpy(dtype=float)
        step = np.rad2deg(np.angle(np.exp(1j * np.diff(np.arctan2(dy, dx)))))[pair]
        mad = 1.4826 * np.median(np.abs(step)) if step.size else np.nan
        out[f"{part}_length_px"] = float(np.nanmedian(np.hypot(dx, dy)[ok])) if ok.any() else np.nan
        out[f"{part}_jitter_deg"] = float(mad)
        out[f"{part}_jitter_degps"] = float(mad * frame_rate_hz)
        out[f"{part}_jump_share"] = float(np.mean(np.abs(step) > C.QC_YAW_JUMP_DEG)) if step.size else np.nan
    return out


# ┌────────────────────────────────────────────────────────────┐
# │ Saccade rejections  « why the artefact filter drops a raw peak » │
# └────────────────────────────────────────────────────────────┘
def saccade_rejections(trajectories: pd.DataFrame, frame_rates: dict[str, float]) -> pd.DataFrame:
    """Raw peaks per trajectory and source, labelled with the filter rule(s) they fail.

    Re-runs the detector so that rejected peaks, which the saccade table
    never holds, can be counted.  ``reason`` is ``kept`` or a ``+``-joined
    list of the failed rules.
    """
    rows: list[dict[str, object]] = []
    keys = [C.COL_DATASET, C.COL_SPECIES, C.COL_TRAJECTORY, C.COL_SEGMENT]
    for (dataset, species, trajectory, _), seg in trajectories.groupby(keys, sort=False):
        fps = frame_rates[dataset]
        bridged = seg[C.COL_BRIDGED].to_numpy(dtype=bool) if C.COL_BRIDGED in seg else np.zeros(len(seg), dtype=bool)
        for source, yaw_col in (("body", C.COL_BODY_YAW), ("head", C.COL_HEAD_YAW)):
            peaks = detect_saccades(seg[yaw_col].to_numpy(), fps,
                                    threshold_degps=C.SACCADE_VELOCITY_THRESHOLDS_DEGPS[source])
            share = bridged_window_share(bridged, peaks["peak_idx"].to_numpy(dtype=int), trigger_half_window(fps))
            for (_, peak), window_share in zip(peaks.iterrows(), share):
                failed = [name for name, bad in (
                    ("bridged_peak", bridged[int(peak["peak_idx"])]),
                    ("bridged_window", window_share > C.MAX_BRIDGED_WINDOW_SHARE),
                    ("amplitude", abs(peak["amplitude_deg"]) < C.MIN_SACCADE_AMPLITUDE_DEG),
                    ("duration", peak["duration_s"] > C.MAX_SACCADE_DURATION_S),
                    ("speed", abs(peak["peak_speed_degps"]) > C.MAX_SACCADE_SPEED_DEGPS),
                ) if bad]
                rows.append({C.COL_DATASET: dataset, C.COL_SPECIES: species, C.COL_TRAJECTORY: trajectory,
                             C.COL_SOURCE: source, "reason": "+".join(failed) or "kept",
                             "peak_speed_degps": float(peak["peak_speed_degps"]),
                             "amplitude_deg": float(peak["amplitude_deg"])})
    columns = [C.COL_DATASET, C.COL_SPECIES, C.COL_TRAJECTORY, C.COL_SOURCE, "reason", "peak_speed_degps", "amplitude_deg"]
    return pd.DataFrame(rows, columns=columns)


def rejection_counts(rejections: pd.DataFrame) -> pd.DataFrame:
    """Peaks per data set, species, source and reason."""
    keys = [C.COL_DATASET, C.COL_SPECIES, C.COL_SOURCE, "reason"]
    return rejections.groupby(keys).size().rename("n").reset_index()


# ┌────────────────────────────────────────────────────────────┐
# │ Figures  « budget, scores, noise floor, rejections, timeline » │
# └────────────────────────────────────────────────────────────┘
def plot_frame_budget(budget: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Share of each movie's frames surviving every gate, one line per movie."""
    if ax is None:
        _, ax = plt.subplots(figsize=C.FIGURE_SIZE_DOUBLE)
    gates = [g for g in C.QC_GATES if g in budget.columns]
    x = np.arange(len(gates))
    for _, row in budget.iterrows():
        share = row[gates].to_numpy(dtype=float) / max(float(row["frames"]), 1.0)
        colour = C.SPECIES_COLOURS.get(row[C.COL_SPECIES], WONG_FALLBACK)
        marker = C.CAMERA_MARKERS.get(row[C.COL_DATASET], "o")
        ax.plot(x, share, marker=marker, ms=4, lw=1, color=colour, alpha=0.8, label=str(row["movie"]).split(".")[0])
    ax.set_xticks(x)
    ax.set_xticklabels([C.QC_GATE_LABELS.get(g, g) for g in gates], rotation=30, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("share of movie frames")
    ax.legend(fontsize=5, ncol=1 + len(budget) // 20, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return ax


def plot_keypoint_scores(scores: pd.DataFrame, threshold: float, axes: np.ndarray | None = None) -> np.ndarray:
    """One raincloud per keypoint, movies as groups, with the score gate drawn."""
    if axes is None:
        _, axes = plt.subplots(1, len(C.DLC_KEYPOINTS), figsize=C.FIGURE_SIZE_A4_LANDSCAPE, sharey=True)
    movies = sorted(scores["movie"].unique())
    colours = {m: WONG_FALLBACK for m in movies}
    for ax, kp in zip(axes, C.DLC_KEYPOINTS):
        sub = scores[scores["keypoint"] == kp]
        plot_raincloud(sub, "score", "movie", colours, "keypoint score", order=movies, ax=ax)
        ax.axhline(threshold, color=C.WONG["vermilion"], ls="--", lw=0.8)
        ax.set_title(kp)
        ax.set_xticklabels([f"{m[:12]}\n(n = {(sub['movie'] == m).sum()})" for m in movies], rotation=90, fontsize=6)
    return axes


def plot_noise_floor(trajectories: pd.DataFrame, ax: plt.Axes | None = None) -> tuple[plt.Axes, pd.DataFrame]:
    """|yaw velocity| per movie and body part, with the saccade thresholds.

    Returns the axes and the long table behind the plot (``group``,
    ``part``, ``abs_velocity_degps``).  The median of each cloud is the
    noise floor the threshold has to clear.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=C.FIGURE_SIZE_DOUBLE)
    parts = []
    for part, col in (("body", C.COL_BODY_VEL), ("head", C.COL_HEAD_VEL)):
        parts.append(pd.DataFrame({
            "group": trajectories["qc_group"] + " " + part, "part": part,
            "abs_velocity_degps": trajectories[col].abs().to_numpy(),
        }))
    table = pd.concat(parts, ignore_index=True).dropna()
    order = sorted(table["group"].unique(), key=lambda g: (g.rsplit(" ", 1)[1], g))
    colours = {g: C.BODYPART_COLOURS[g.rsplit(" ", 1)[1]] for g in order}
    plot_raincloud(table, "abs_velocity_degps", "group", colours, r"|yaw velocity| ($^\circ\,s^{-1}$)",
                   order=order, log_y=True, ax=ax)
    for part, threshold in C.SACCADE_VELOCITY_THRESHOLDS_DEGPS.items():
        ax.axhline(threshold, color=C.BODYPART_COLOURS[part], ls="--", lw=0.8)
    ax.tick_params(axis="x", labelsize=6, rotation=90)
    return ax, table


def plot_rejections(rejections: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Raw peaks per reason, one point per trajectory, coloured by species."""
    if ax is None:
        _, ax = plt.subplots(figsize=C.FIGURE_SIZE_DOUBLE)
    reasons = ["kept", *sorted(r for r in rejections["reason"].unique() if r != "kept")]
    per_trajectory = rejections.groupby(
        [C.COL_DATASET, C.COL_SPECIES, C.COL_TRAJECTORY, C.COL_SOURCE, "reason"]).size().rename("n").reset_index()
    rng = np.random.default_rng(C.RANDOM_SEED)
    for i, reason in enumerate(reasons):
        sub = per_trajectory[per_trajectory["reason"] == reason]
        for source, offset in (("body", -0.15), ("head", 0.15)):
            s = sub[sub[C.COL_SOURCE] == source]
            ax.scatter(i + offset + rng.uniform(-0.05, 0.05, len(s)), s["n"], s=12,
                       marker="o" if source == "head" else "s",
                       c=[C.SPECIES_COLOURS.get(sp, WONG_FALLBACK) for sp in s[C.COL_SPECIES]], alpha=0.7, lw=0)
        total = rejections[rejections["reason"] == reason].groupby(C.COL_SOURCE).size()
        ax.annotate(" / ".join(f"{src} {total.get(src, 0)}" for src in ("body", "head")), (i, ax.get_ylim()[1]),
                    ha="center", va="bottom", fontsize=6, annotation_clip=False)
    ax.set_xticks(range(len(reasons)))
    ax.set_xticklabels(reasons, rotation=30, ha="right")
    ax.set_ylabel("raw peaks per trajectory")
    handles = [plt.Line2D([], [], ls="", marker="s", color=WONG_FALLBACK, label="body"),
               plt.Line2D([], [], ls="", marker="o", color=WONG_FALLBACK, label="head")]
    handles += [plt.Line2D([], [], ls="", marker="o", color=colour, label=species)
                for species, colour in C.SPECIES_COLOURS.items() if species in set(rejections[C.COL_SPECIES])]
    ax.legend(handles=handles, fontsize=6, frameon=False, loc="upper right")
    return ax


def plot_detection_timeline(
    movies: list[tuple[str, np.ndarray, float]],
    trajectories: pd.DataFrame,
    saccades: pd.DataFrame | None = None,
    axes: np.ndarray | None = None,
) -> np.ndarray:
    """One row per movie: detected frames, frames in the trajectory table, bridged frames, saccades.

    Args:
        movies:       ``(source_file, detected_mask, frame_rate_hz)`` per movie.
        trajectories: Trajectory table with a ``source_file`` column.
        saccades:     Kept saccades with ``source_file``; drawn as ticks.
    """
    if axes is None:
        _, axes = plt.subplots(len(movies), 1, figsize=(C.FIGURE_SIZE_A4_LANDSCAPE[0], 0.9 * len(movies) + 0.8),
                               squeeze=False, gridspec_kw={"hspace": 1.2})
    axes = axes.ravel()
    for ax, (name, detected, fps) in zip(axes, movies):
        runs = timeline_runs([(name, detected, fps)], trajectories)
        for kind, y, colour in (("detected", 2, C.WONG["black"]), ("usable", 1, C.WONG["bluish_green"]),
                                ("bridged", 0, C.WONG["orange"])):
            sub = runs[runs["kind"] == kind]
            ax.broken_barh(list(zip(sub["start_s"] / 60, (sub["stop_s"] - sub["start_s"]) / 60)), (y - 0.4, 0.8),
                           color=colour, lw=0)
        if saccades is not None and len(saccades):
            sacc = saccades[saccades["source_file"] == name]
            for source, y in (("body", 3), ("head", 3.6)):
                t = sacc.loc[sacc[C.COL_SOURCE] == source, "peak_time_s"].to_numpy() / 60
                ax.eventplot(t, lineoffsets=y, linelengths=0.5, colors=C.BODYPART_COLOURS[source], lw=0.6)
        ax.set_yticks([0, 1, 2, 3, 3.6])
        ax.set_yticklabels(["bridged", "usable", "detected", "body sacc.", "head sacc."], fontsize=6)
        ax.set_ylim(-0.6, 4.1)
        ax.set_xlim(0, detected.size / fps / 60)
        ax.set_title(f"{name.split('.')[0]}  ({detected.size / fps / 60:.1f} min)", fontsize=7, loc="left")
        ax.set_xlabel("time in movie (min)", fontsize=6, labelpad=1)
        ax.tick_params(axis="x", labelsize=6)
    return axes


def timeline_runs(
    movies: list[tuple[str, np.ndarray, float]], trajectories: pd.DataFrame
) -> pd.DataFrame:
    """The runs drawn by :func:`plot_detection_timeline`: movie, kind, start_s, stop_s."""
    rows = []
    for name, detected, fps in movies:
        traj = trajectories[trajectories["source_file"] == name]
        frames = traj[C.COL_FRAME].to_numpy(dtype=int)
        frames = frames[frames < detected.size]
        usable = np.zeros(detected.size, dtype=bool)
        usable[frames] = True
        bridged = np.zeros(detected.size, dtype=bool)
        if C.COL_BRIDGED in traj:
            flags = traj[C.COL_BRIDGED].to_numpy(dtype=bool)[: frames.size]
            bridged[frames[flags]] = True
        for kind, mask in (("detected", detected), ("usable", usable), ("bridged", bridged)):
            rows += [{"movie": name, "kind": kind, "start_s": s / fps, "stop_s": e / fps} for s, e in _runs(mask)]
    return pd.DataFrame(rows, columns=["movie", "kind", "start_s", "stop_s"])


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Start and stop (exclusive) of every run of True."""
    padded = np.r_[False, mask, False]
    edges = np.where(np.diff(padded.astype(int)) != 0)[0]
    return list(zip(edges[::2], edges[1::2]))
