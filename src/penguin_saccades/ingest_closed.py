# ╔══════════════════════════════════════════════════════════════════╗
# ║  penguin_saccades — ingest_closed                                ║
# ║  « data set 2: DeepLabCut poses to head and body yaw »           ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Reads DLC output, gates every keypoint on its likelihood,       ║
# ║  bridges short gaps, derives head yaw (neck to beak) and body    ║
# ║  yaw (caudal to neck), and scales translation by the animal's    ║
# ║  own body length.                                                ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Ingestion of data set 2 (DeepLabCut head and body pose)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from penguin_saccades import constants as C
from penguin_saccades.kinematics import (
    sigma_frames,
    smooth,
    split_segments,
    translational_speed,
    vector_angle,
    yaw_velocity_degps,
)


# ┌────────────────────────────────────────────────────────────┐
# │ Public API  « read, gate, process »                        │
# └────────────────────────────────────────────────────────────┘
def read_dlc(path: Path) -> pd.DataFrame:
    """Read a DeepLabCut CSV or HDF5 file into flat ``<keypoint>_<x|y|likelihood>`` columns."""
    if path.suffix.lower() == ".h5":
        df = pd.read_hdf(path)
    else:
        df = pd.read_csv(path, header=[1, 2], index_col=0)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(str(level) for level in col[-2:]) for col in df.columns]
    df.index = np.arange(len(df))
    df.index.name = C.COL_FRAME
    return df


def read_sleap_analysis(path: Path, min_occupancy: int = 10) -> pd.DataFrame:
    """SLEAP analysis HDF5 (``sleap-convert --format analysis``) to the flat DLC-style table.

    Every track becomes its own block of rows, tagged by ``track_id`` so the
    focal bird can be chosen downstream.  Point scores are stored in the
    ``<kp>_likelihood`` columns; SLEAP scores are not probabilities and can
    exceed 1, so the gate threshold must be set for them.
    """
    import h5py

    with h5py.File(path, "r") as f:
        tracks = f["tracks"][:]                      # (n_tracks, 2, n_nodes, n_frames)
        scores = f["point_scores"][:]                # (n_tracks, n_nodes, n_frames)
        nodes = [n.decode() if isinstance(n, bytes) else str(n) for n in f["node_names"][:]]
    parts: list[pd.DataFrame] = []
    for t in range(tracks.shape[0]):
        present = np.isfinite(tracks[t, 0]).any(axis=0)
        if present.sum() < min_occupancy:
            continue
        frames = np.where(present)[0]
        block = {C.COL_FRAME: frames, "track_id": t}
        for j, kp in enumerate(nodes):
            block[f"{kp}_x"] = tracks[t, 0, j, frames]
            block[f"{kp}_y"] = tracks[t, 1, j, frames]
            block[f"{kp}_likelihood"] = np.nan_to_num(scores[t, j, frames], nan=0.0)
        parts.append(pd.DataFrame(block))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def focal_track(df: pd.DataFrame) -> pd.DataFrame:
    """Rows of the track with the most frames, re-indexed by frame for :func:`process_closed_trajectory`."""
    best = df.groupby("track_id").size().idxmax()
    out = df[df["track_id"] == best].drop(columns="track_id").set_index(C.COL_FRAME)
    full = pd.RangeIndex(0, int(out.index.max()) + 1, name=C.COL_FRAME)
    out = out.reindex(full)
    for col in out.columns:
        if col.endswith("_likelihood"):
            out[col] = out[col].fillna(0.0)
    return out


def gate_likelihood(
    df: pd.DataFrame,
    keypoints: tuple[str, ...] = C.DLC_KEYPOINTS,
    threshold: float | None = None,
    max_gap_frames: int | None = None,
) -> pd.DataFrame:
    """Set x and y to NaN where the likelihood is below threshold; bridge short gaps."""
    threshold = C.DLC_LIKELIHOOD_THRESHOLD if threshold is None else threshold
    max_gap_frames = C.MAX_INTERPOLATION_GAP_FRAMES if max_gap_frames is None else max_gap_frames
    out = df.copy()
    for kp in keypoints:
        low = out[f"{kp}_likelihood"] < threshold
        for axis in ("x", "y"):
            col = f"{kp}_{axis}"
            out.loc[low, col] = np.nan
            out[col] = out[col].interpolate(limit=max_gap_frames, limit_area="inside")
    return out


def process_closed_trajectory(
    df_dlc: pd.DataFrame,
    spec: C.DatasetSpec = C.DATASET_CLOSED,
    body_length_m: float = C.GENTOO_BODY_LENGTH_M,
) -> pd.DataFrame:
    """Gated DLC poses to the shared trajectory schema.

    Scale: the median beak-to-tail length in pixels is equated with
    ``body_length_m``.  Translation is the neck displacement.
    """
    fps = spec.frame_rate_hz
    sigma = sigma_frames(spec.smoothing_sigma_s, fps)
    gated = gate_likelihood(df_dlc)
    px_per_m = body_length_px(gated) / body_length_m
    head_tail, head_tip = C.HEAD_VECTOR
    body_tail, body_tip = C.BODY_VECTOR
    head_yaw = vector_angle(
        gated[f"{head_tail}_x"], gated[f"{head_tail}_y"], gated[f"{head_tip}_x"], gated[f"{head_tip}_y"]
    )
    body_yaw = vector_angle(
        gated[f"{body_tail}_x"], gated[f"{body_tail}_y"], gated[f"{body_tip}_x"], gated[f"{body_tip}_y"]
    )
    valid = np.isfinite(head_yaw) & np.isfinite(body_yaw) & skeleton_is_plausible(gated, head_yaw, body_yaw)
    min_len = round(C.MIN_SEGMENT_S * fps)
    parts: list[pd.DataFrame] = []
    for seg_id, frames in enumerate(split_segments(np.where(valid)[0], 0, min_len)):
        sl = slice(frames[0], frames[-1] + 1)
        x_m = smooth(gated["neck_x"].to_numpy()[sl] / px_per_m, sigma)
        y_m = smooth(gated["neck_y"].to_numpy()[sl] / px_per_m, sigma)
        h = smooth(head_yaw[sl], sigma)
        b = smooth(body_yaw[sl], sigma)
        parts.append(pd.DataFrame({
            C.COL_SEGMENT: seg_id,
            C.COL_FRAME: frames,
            C.COL_TIME: frames / fps,
            C.COL_X: x_m,
            C.COL_Y: y_m,
            C.COL_BODY_YAW: b,
            C.COL_BODY_VEL: yaw_velocity_degps(b, fps),
            C.COL_HEAD_YAW: h,
            C.COL_HEAD_VEL: yaw_velocity_degps(h, fps),
            C.COL_TRANS: translational_speed(x_m, y_m, fps),
        }))
    if not parts:
        return pd.DataFrame({c: pd.Series(dtype=float) for c in C.TRAJECTORY_COLUMNS})
    out = pd.concat(parts, ignore_index=True)
    out[C.COL_DATASET] = spec.name
    out[C.COL_SPECIES] = C.CLOSED_SPECIES
    return out


def skeleton_is_plausible(
    df: pd.DataFrame,
    head_yaw: np.ndarray,
    body_yaw: np.ndarray,
    length_ratio: tuple[float, float] | None = None,
    max_head_body_angle_deg: float | None = None,
) -> np.ndarray:
    """Frames whose skeleton belongs to one bird.

    Rejects frames where a segment length leaves ``length_ratio`` times
    its median (keypoints on two different penguins) or where the head
    points more than ``max_head_body_angle_deg`` off the body axis.
    """
    length_ratio = C.SKELETON_LENGTH_RATIO if length_ratio is None else length_ratio
    max_angle = C.MAX_HEAD_BODY_ANGLE_DEG if max_head_body_angle_deg is None else max_head_body_angle_deg
    ok = np.ones(len(df), dtype=bool)
    for a, b in (C.HEAD_VECTOR, C.BODY_VECTOR):
        length = np.hypot(df[f"{a}_x"] - df[f"{b}_x"], df[f"{a}_y"] - df[f"{b}_y"]).to_numpy()
        median = np.nanmedian(length)
        ok &= (length >= length_ratio[0] * median) & (length <= length_ratio[1] * median)
    relative = np.rad2deg(np.angle(np.exp(1j * (head_yaw - body_yaw))))
    ok &= np.abs(relative) <= max_angle
    return ok


def body_length_px(df: pd.DataFrame) -> float:
    """Median beak-to-tail length in pixels (neck-caudal plus beak-neck)."""
    neck_caudal = np.hypot(df["neck_x"] - df["caudal_x"], df["neck_y"] - df["caudal_y"])
    beak_neck = np.hypot(df["beak_x"] - df["neck_x"], df["beak_y"] - df["neck_y"])
    length = float(np.nanmedian(neck_caudal + beak_neck))
    if not np.isfinite(length) or length <= 0:
        raise ValueError("Cannot estimate body length: no frame has all three keypoints.")
    return length


def recording_id_from_filename(path: Path) -> str:
    """Leading digits of a DLC file name identify the recording (legacy convention)."""
    match = re.search(r"\d+", path.name)
    if match is None:
        raise ValueError(f"No numeric recording id in file name: {path.name}")
    return match.group(0)
