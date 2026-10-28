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


def gate_likelihood(
    df: pd.DataFrame,
    keypoints: tuple[str, ...] = C.DLC_KEYPOINTS,
    threshold: float = C.DLC_LIKELIHOOD_THRESHOLD,
    max_gap_frames: int = C.MAX_INTERPOLATION_GAP_FRAMES,
) -> pd.DataFrame:
    """Set x and y to NaN where the likelihood is below threshold; bridge short gaps."""
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
    sigma = sigma_frames(C.SMOOTHING_SIGMA_S, fps)
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
    valid = np.isfinite(head_yaw) & np.isfinite(body_yaw)
    min_len = int(round(C.MIN_SEGMENT_S * fps))
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
        return pd.DataFrame(columns=list(C.TRAJECTORY_COLUMNS))
    out = pd.concat(parts, ignore_index=True)
    out[C.COL_DATASET] = spec.name
    out[C.COL_SPECIES] = C.CLOSED_SPECIES
    return out


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
