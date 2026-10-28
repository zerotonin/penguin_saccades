# ╔══════════════════════════════════════════════════════════════════╗
# ║  penguin_saccades — saccades                                     ║
# ║  « one detector, one artefact filter, per-segment intervals »    ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  A saccade is a peak of absolute yaw velocity above threshold.   ║
# ║  Its boundaries are where the velocity falls back towards        ║
# ║  baseline. Intersaccadic intervals are the gaps between          ║
# ║  saccades of the same segment, so no interval can span a         ║
# ║  tracking gap or a movie boundary.                               ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Saccade detection, artefact filtering, intersaccadic intervals, triggered windows."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths

from penguin_saccades import constants as C
from penguin_saccades.constants import FloatArray
from penguin_saccades.kinematics import yaw_velocity_degps

SACCADE_COLUMNS: tuple[str, ...] = (
    "start_idx", "peak_idx", "stop_idx", "peak_time_s", "duration_s",
    "amplitude_deg", "peak_speed_degps", "direction",
)
INTERVAL_COLUMNS: tuple[str, ...] = ("start_idx", "stop_idx", "duration_s", "censored")


# ┌────────────────────────────────────────────────────────────────────┐
# │ Detection  « peaks of |yaw velocity| with width-based boundaries » │
# └────────────────────────────────────────────────────────────────────┘
def detect_saccades(
    yaw_rad: FloatArray,
    frame_rate_hz: float,
    threshold_degps: float = C.SACCADE_VELOCITY_THRESHOLD_DEGPS,
    min_separation_s: float = C.MIN_PEAK_SEPARATION_S,
    rel_height: float = C.BOUNDARY_REL_HEIGHT,
    settle_s: float = C.AMPLITUDE_SETTLE_S,
) -> pd.DataFrame:
    """Detect saccades in one continuous yaw trace (radians, may contain NaN).

    Indices refer to positions in ``yaw_rad``.  Amplitude is the heading
    change between the settle windows before and after the event; it and
    the peak speed are signed, and ``direction`` follows the
    image-coordinate convention in :mod:`penguin_saccades.constants`.
    """
    yaw_rad = np.asarray(yaw_rad, dtype=float)
    velocity = yaw_velocity_degps(yaw_rad, frame_rate_hz)
    abs_velocity = np.nan_to_num(np.abs(velocity), nan=0.0)
    distance = max(1, int(round(min_separation_s * frame_rate_hz)))
    peaks, _ = find_peaks(abs_velocity, height=threshold_degps, distance=distance)
    if peaks.size == 0:
        return _empty(SACCADE_COLUMNS)
    widths, _, left, right = peak_widths(abs_velocity, peaks, rel_height=rel_height)
    start = np.clip(np.floor(left).astype(int), 0, yaw_rad.size - 1)
    stop = np.clip(np.ceil(right).astype(int), 0, yaw_rad.size - 1)
    yaw_deg = np.rad2deg(yaw_rad)
    settle = max(1, int(round(settle_s * frame_rate_hz)))
    df = pd.DataFrame({
        "start_idx": start,
        "peak_idx": peaks,
        "stop_idx": stop,
        "peak_time_s": peaks / frame_rate_hz,
        "duration_s": widths / frame_rate_hz,
        "amplitude_deg": [_settled_amplitude(yaw_deg, s, e, settle) for s, e in zip(start, stop)],
        "peak_speed_degps": velocity[peaks],
    })
    df["direction"] = np.where(
        df["peak_speed_degps"] > 0, C.DIRECTION_OF_POSITIVE_VELOCITY, C.DIRECTION_OF_NEGATIVE_VELOCITY
    )
    touches_nan = [not np.all(np.isfinite(yaw_rad[s:e + 1])) for s, e in zip(start, stop)]
    return df.loc[~np.asarray(touches_nan)].reset_index(drop=True)


def filter_artefacts(
    saccades: pd.DataFrame,
    max_duration_s: float = C.MAX_SACCADE_DURATION_S,
    min_amplitude_deg: float = C.MIN_SACCADE_AMPLITUDE_DEG,
    max_speed_degps: float = C.MAX_SACCADE_SPEED_DEGPS,
) -> pd.DataFrame:
    """Drop events that are too long, too small or implausibly fast."""
    keep = (
        (saccades["duration_s"] <= max_duration_s)
        & (saccades["amplitude_deg"].abs() >= min_amplitude_deg)
        & (saccades["peak_speed_degps"].abs() <= max_speed_degps)
    )
    return saccades.loc[keep].reset_index(drop=True)


# ┌────────────────────────────────────────────────────────────┐
# │ Intervals  « gaps between saccades inside one segment »    │
# └────────────────────────────────────────────────────────────┘
def intersaccadic_intervals(saccades: pd.DataFrame, n_frames: int, frame_rate_hz: float) -> pd.DataFrame:
    """Frames not covered by any saccade, as contiguous intervals.

    Leading and trailing intervals are flagged ``censored`` because one
    of their boundaries is the recording edge rather than a saccade.
    """
    if n_frames <= 0:
        return _empty(INTERVAL_COLUMNS)
    covered = np.zeros(n_frames, dtype=bool)
    for s, e in zip(saccades["start_idx"], saccades["stop_idx"]):
        covered[int(s):int(e) + 1] = True
    free = ~covered
    edges = np.diff(free.astype(int), prepend=0, append=0)
    starts = np.where(edges == 1)[0]
    stops = np.where(edges == -1)[0] - 1
    df = pd.DataFrame({"start_idx": starts, "stop_idx": stops})
    df["duration_s"] = (df["stop_idx"] - df["start_idx"] + 1) / frame_rate_hz
    df["censored"] = (df["start_idx"] == 0) | (df["stop_idx"] == n_frames - 1)
    return df


# ┌───────────────────────────────────────────────────────────────────────────────┐
# │ Per-segment driver  « apply detector to every segment of a trajectory table » │
# └───────────────────────────────────────────────────────────────────────────────┘
def detect_in_trajectories(
    trajectories: pd.DataFrame,
    yaw_col: str,
    source: str,
    frame_rates: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run detection and interval extraction per (trajectory, segment).

    Args:
        trajectories: Shared trajectory table (one row per frame).
        yaw_col:      ``COL_BODY_YAW`` or ``COL_HEAD_YAW``.
        source:       Label written to the ``source`` column.
        frame_rates:  Frame rate per dataset name.

    Returns:
        ``(saccades, intervals)`` with indices expressed as global frame
        numbers of the segment and labelled by dataset, species,
        trajectory and segment.
    """
    keys = [C.COL_DATASET, C.COL_SPECIES, C.COL_TRAJECTORY, C.COL_SEGMENT]
    sacc_parts: list[pd.DataFrame] = []
    int_parts: list[pd.DataFrame] = []
    for key, seg in trajectories.groupby(keys, sort=False):
        fps = frame_rates[key[0]]
        yaw = seg[yaw_col].to_numpy()
        if not np.isfinite(yaw).any():
            continue
        frames = seg[C.COL_FRAME].to_numpy()
        sacc = filter_artefacts(detect_saccades(yaw, fps))
        ints = intersaccadic_intervals(sacc, len(yaw), fps)
        for df, parts in ((sacc, sacc_parts), (ints, int_parts)):
            df = df.copy()
            for col in ("start_idx", "peak_idx", "stop_idx"):
                if col in df:
                    df[col] = frames[df[col].to_numpy(dtype=int)]
            if "peak_time_s" in df:
                df["peak_time_s"] = df["peak_idx"] / fps
            for name, value in zip(keys, key):
                df[name] = value
            df[C.COL_SOURCE] = source
            parts.append(df)
    labels = (*keys, C.COL_SOURCE)
    saccades = (pd.concat(sacc_parts, ignore_index=True) if sacc_parts
                else _empty((*SACCADE_COLUMNS, *labels)))
    intervals = (pd.concat(int_parts, ignore_index=True) if int_parts
                 else _empty((*INTERVAL_COLUMNS, *labels)))
    return saccades, intervals


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ Triggered windows  « signal around each saccade peak, left turns mirrored » │
# └─────────────────────────────────────────────────────────────────────────────┘
def triggered_windows(
    signal: FloatArray, peak_positions: np.ndarray, half_window: int, flip: np.ndarray | None = None
) -> np.ndarray:
    """Matrix of ``signal`` around each peak, NaN-padded at the edges.

    Args:
        signal:         1-D trace (angle or velocity).
        peak_positions: Peak positions in ``signal``.
        half_window:    Frames on each side of the peak.
        flip:           Boolean per peak; rows flagged True are negated so
                        left and right turns can be pooled.
    """
    signal = np.asarray(signal, dtype=float)
    width = 2 * half_window + 1
    out = np.full((len(peak_positions), width), np.nan)
    for i, p in enumerate(np.asarray(peak_positions, dtype=int)):
        lo, hi = p - half_window, p + half_window + 1
        src_lo, src_hi = max(lo, 0), min(hi, signal.size)
        out[i, src_lo - lo:src_lo - lo + (src_hi - src_lo)] = signal[src_lo:src_hi]
    if flip is not None:
        out[np.asarray(flip, dtype=bool)] *= -1.0
    return out


def zero_offset_rows(matrix: np.ndarray, n_reference: int = 3) -> np.ndarray:
    """Subtract the mean of the first ``n_reference`` finite values of each row."""
    out = matrix.copy()
    for i, row in enumerate(out):
        finite = row[np.isfinite(row)]
        if finite.size:
            out[i] = row - np.mean(finite[:n_reference])
    return out


def _settled_amplitude(yaw_deg: FloatArray, start: int, stop: int, settle: int) -> float:
    """Mean heading after the stop minus mean heading before the start (settle windows)."""
    before = yaw_deg[max(0, start - settle):start]
    after = yaw_deg[stop + 1:stop + 1 + settle]
    before_val = np.nanmean(before) if np.isfinite(before).any() else yaw_deg[start]
    after_val = np.nanmean(after) if np.isfinite(after).any() else yaw_deg[stop]
    return float(after_val - before_val)


def _empty(columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=float) for c in columns})
