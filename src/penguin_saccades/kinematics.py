# ╔══════════════════════════════════════════════════════════════════╗
# ║  penguin_saccades — kinematics                                   ║
# ║  « from positions to yaw and speed, once, correctly »            ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Segment splitting, gap filling, Gaussian smoothing (order 0, a  ║
# ║  true low-pass), heading from displacement with a validity       ║
# ║  mask, yaw velocity and translational speed. Shared by both      ║
# ║  data sets.                                                      ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Trajectory kinematics shared by both data sets."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from penguin_saccades.constants import FloatArray, IndexArray


# ┌────────────────────────────────────────────────────────────┐
# │ Segments  « split at tracking gaps, bridge short ones »    │
# └────────────────────────────────────────────────────────────┘
def split_segments(
    frame_index: IndexArray, max_gap_frames: int, min_length_frames: int
) -> list[IndexArray]:
    """Split sorted frame numbers where more than ``max_gap_frames`` consecutive frames are missing.

    Returns:
        List of arrays of the original frame numbers, one per segment,
        keeping only segments spanning at least ``min_length_frames``.
    """
    frames = np.asarray(frame_index, dtype=int)
    if frames.size == 0:
        return []
    missing = np.diff(frames) - 1
    breaks = np.where(missing > max_gap_frames)[0] + 1
    pieces = np.split(frames, breaks)
    return [p for p in pieces if (p[-1] - p[0] + 1) >= min_length_frames]


def fill_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex a segment to consecutive frames and interpolate numeric columns linearly."""
    start, stop = int(df.index.min()), int(df.index.max())
    full = pd.RangeIndex(start, stop + 1, name=df.index.name)
    out = df.reindex(full)
    numeric = out.select_dtypes(include="number").columns
    out[numeric] = out[numeric].interpolate(limit_direction="both")
    other = out.columns.difference(numeric)
    if len(other):
        out[other] = out[other].ffill().bfill()
    return out


# ┌────────────────────────────────────────────────────────────┐
# │ Smoothing  « Gaussian low-pass that tolerates NaN runs »   │
# └────────────────────────────────────────────────────────────┘
def smooth(x: FloatArray, sigma_frames: float) -> FloatArray:
    """Gaussian low-pass (order 0).  NaN runs are left untouched and smoothed separately."""
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)
    if sigma_frames <= 0:
        return x.copy()
    for start, stop in finite_runs(x):
        out[start:stop] = gaussian_filter1d(x[start:stop], sigma_frames, order=0, mode="nearest")
    return out


def finite_runs(x: FloatArray) -> list[tuple[int, int]]:
    """Half-open ``(start, stop)`` index ranges of consecutive finite values."""
    finite = np.isfinite(np.asarray(x, dtype=float))
    if not finite.any():
        return []
    edges = np.diff(finite.astype(int), prepend=0, append=0)
    starts = np.where(edges == 1)[0]
    stops = np.where(edges == -1)[0]
    return list(zip(starts.tolist(), stops.tolist()))


def unwrap(angle_rad: FloatArray) -> FloatArray:
    """``np.unwrap`` applied separately to every finite run."""
    angle_rad = np.asarray(angle_rad, dtype=float)
    out = np.full_like(angle_rad, np.nan)
    for start, stop in finite_runs(angle_rad):
        out[start:stop] = np.unwrap(angle_rad[start:stop])
    return out


# ┌────────────────────────────────────────────────────────────────────┐
# │ Derived quantities  « heading, yaw velocity, translational speed » │
# └────────────────────────────────────────────────────────────────────┘
def heading_from_displacement(
    x_m: FloatArray, y_m: FloatArray, frame_rate_hz: float, min_speed_mps: float
) -> FloatArray:
    """Unwrapped heading (rad) of the displacement vector; NaN where the animal is too slow.

    Heading from the first derivative of position is not body orientation
    when the animal drifts, so frames below ``min_speed_mps`` are masked.
    """
    dx = np.gradient(np.asarray(x_m, dtype=float))
    dy = np.gradient(np.asarray(y_m, dtype=float))
    heading = np.arctan2(dy, dx)
    speed = np.hypot(dx, dy) * frame_rate_hz
    heading[speed < min_speed_mps] = np.nan
    return unwrap(heading)


def vector_angle(
    tail_x: FloatArray, tail_y: FloatArray, tip_x: FloatArray, tip_y: FloatArray
) -> FloatArray:
    """Unwrapped angle (rad) of the vector from ``tail`` to ``tip`` keypoints."""
    angle = np.arctan2(
        np.asarray(tip_y, float) - np.asarray(tail_y, float),
        np.asarray(tip_x, float) - np.asarray(tail_x, float),
    )
    return unwrap(angle)


def yaw_velocity_degps(yaw_rad: FloatArray, frame_rate_hz: float) -> FloatArray:
    """Central-difference yaw velocity in deg/s (same length as the input)."""
    yaw_rad = np.asarray(yaw_rad, dtype=float)
    out = np.full_like(yaw_rad, np.nan)
    for start, stop in finite_runs(yaw_rad):
        if stop - start >= 2:
            out[start:stop] = np.rad2deg(np.gradient(yaw_rad[start:stop])) * frame_rate_hz
    return out


def translational_speed(x_m: FloatArray, y_m: FloatArray, frame_rate_hz: float) -> FloatArray:
    """Speed along the path in m/s from central differences of x and y."""
    dx = np.gradient(np.asarray(x_m, dtype=float))
    dy = np.gradient(np.asarray(y_m, dtype=float))
    return np.hypot(dx, dy) * frame_rate_hz


def sigma_frames(sigma_s: float, frame_rate_hz: float) -> float:
    """Convert a smoothing width in seconds into frames."""
    return sigma_s * frame_rate_hz
