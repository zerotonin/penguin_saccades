# ╔══════════════════════════════════════════════════════════════════╗
# ║  penguin_saccades — ingest_open                                  ║
# ║  « data set 1: charon boxes or SLEAP poses to trajectories »     ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Reads the Faster-RCNN ``.tra`` text format written by charon,   ║
# ║  keeps the best in-ROI detection per frame, and turns the        ║
# ║  centre of mass track into metres, heading and speeds. The       ║
# ║  SLEAP route takes the re-tracked poses instead and yields       ║
# ║  head and body yaw from the skeleton, not from displacement.     ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Ingestion of data set 1 (overhead camera, charon Faster-RCNN detections)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as PolygonPath

from penguin_saccades import constants as C
from penguin_saccades.ingest_closed import gate_likelihood, pose_to_trajectory
from penguin_saccades.kinematics import (
    fill_gaps,
    heading_from_displacement,
    sigma_frames,
    smooth,
    split_segments,
    translational_speed,
    yaw_velocity_degps,
)

RAW_COLUMNS: tuple[str, ...] = (
    "name", "quality", "bounding_box_x0", "bounding_box_y0",
    "bounding_box_x1", "bounding_box_y1", "center_of_mass_x", "center_of_mass_y",
)


# ┌────────────────────────────────────────────────────────────┐
# │ Public API  « load, process, label »                       │
# └────────────────────────────────────────────────────────────┘
def load_open_trajectory(path: Path) -> pd.DataFrame:
    """Load one pre-extracted trajectory (HDF5 written by the legacy reader)."""
    df = pd.read_hdf(path)
    df.index = df.index.astype(int)
    df.index.name = C.COL_FRAME
    return df


def process_open_trajectory(
    df_raw: pd.DataFrame,
    spec: C.DatasetSpec = C.DATASET_OPEN,
) -> pd.DataFrame:
    """Centre-of-mass detections (relative image coordinates) to the shared trajectory schema.

    Each tracking segment is interpolated over short gaps, positions are
    smoothed with one Gaussian, heading is derived from displacement and
    masked where the animal is too slow, and speeds are computed.
    """
    fps = spec.frame_rate_hz
    sigma = sigma_frames(spec.smoothing_sigma_s, fps)
    min_len = round(C.MIN_SEGMENT_S * fps)
    parts: list[pd.DataFrame] = []
    for seg_id, frames in enumerate(split_segments(df_raw.index.to_numpy(), C.MAX_GAP_FRAMES, min_len)):
        seg = fill_gaps(df_raw.loc[(df_raw.index >= frames[0]) & (df_raw.index <= frames[-1])])
        x_m = smooth(seg["center_of_mass_x"].to_numpy() * spec.image_width_px * spec.m_per_px, sigma)
        y_m = smooth(seg["center_of_mass_y"].to_numpy() * spec.image_height_px * spec.m_per_px, sigma)
        yaw = heading_from_displacement(x_m, y_m, fps, C.MIN_HEADING_SPEED_MPS)
        parts.append(pd.DataFrame({
            C.COL_SEGMENT: seg_id,
            C.COL_FRAME: seg.index.to_numpy(),
            C.COL_TIME: seg.index.to_numpy() / fps,
            C.COL_X: x_m,
            C.COL_Y: y_m,
            C.COL_BODY_YAW: yaw,
            C.COL_BODY_VEL: yaw_velocity_degps(yaw, fps),
            C.COL_HEAD_YAW: np.nan,
            C.COL_HEAD_VEL: np.nan,
            C.COL_TRANS: translational_speed(x_m, y_m, fps),
        }))
    if not parts:
        return pd.DataFrame(columns=list(C.TRAJECTORY_COLUMNS))
    out = pd.concat(parts, ignore_index=True)
    out[C.COL_DATASET] = spec.name
    return out


def process_open_pose_trajectory(
    df_pose: pd.DataFrame,
    spec: C.DatasetSpec = C.DATASET_OPEN_POSE,
    score_threshold: float | None = None,
) -> pd.DataFrame:
    """SLEAP poses in movie pixels (one instance per frame) to the shared trajectory schema.

    Poses are rescaled to the calibrated frame size, birds outside the water
    ROI are dropped (penguins on land), then head and body yaw come from
    the skeleton exactly as for data set 2.
    """
    threshold = C.SLEAP_SCORE_THRESHOLD if score_threshold is None else score_threshold
    gated = gate_likelihood(df_pose, threshold=threshold)
    gated = _mask_outside_water(gated, spec)
    px_per_m = C.OPEN_MOVIE_SIZE_PX[0] / (spec.image_width_px * spec.m_per_px)
    return pose_to_trajectory(gated, spec, px_per_m)


def species_from_filename(path: Path) -> str:
    """Infer species from the data set 1 file name."""
    for key, species in C.SPECIES_FROM_FILENAME.items():
        if key.lower() in path.name.lower():
            return species
    raise ValueError(f"Cannot infer species from file name: {path.name}")


# ┌───────────────────────────────────────────────────────────────────────────┐
# │ charon .tra reader  « text format: ``frame : >name,q,y0,x0,y1,x1< ...`` » │
# └───────────────────────────────────────────────────────────────────────────┘
def read_charon_tra(
    path: Path,
    roi_px: tuple[tuple[float, float], ...] | None = C.OPEN_WATER_ROI_PX,
    image_size_px: tuple[int, int] = (C.DATASET_OPEN.image_width_px, C.DATASET_OPEN.image_height_px),
    start_line: int = 0,
    max_lines: int | None = None,
) -> pd.DataFrame:
    """Parse a charon trajectory file, keeping the best in-ROI detection per frame.

    Bounding boxes are stored normalised to the image as ``y0, x0, y1, x1``
    (TensorFlow convention).  The returned frame-indexed table uses x/y
    naming and relative coordinates, matching the legacy HDF5 files.
    """
    polygon = _relative_polygon(roi_px, image_size_px) if roi_px else None
    rows: dict[int, dict[str, float | str]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            if line_no < start_line:
                continue
            if max_lines is not None and line_no >= start_line + max_lines:
                break
            frame, best = _parse_line(line, polygon)
            if best is not None:
                rows[frame] = best
    df = pd.DataFrame.from_dict(rows, orient="index", columns=list(RAW_COLUMNS))
    df.index.name = C.COL_FRAME
    return df.sort_index()


def _parse_line(line: str, polygon: PolygonPath | None) -> tuple[int, dict | None]:
    head, *objects = line.rstrip("\n").split(">")
    frame = int(head.split(":")[0])
    best: dict | None = None
    for obj in objects:
        obj = obj.strip().rstrip("<").strip()
        if not obj:
            continue
        name, quality, y0, x0, y1, x1 = obj.split(",")
        cx, cy = (float(x0) + float(x1)) / 2.0, (float(y0) + float(y1)) / 2.0
        if polygon is not None and not polygon.contains_point((cx, cy)):
            continue
        if best is None or float(quality) > best["quality"]:
            best = {
                "name": name, "quality": float(quality),
                "bounding_box_x0": float(x0), "bounding_box_y0": float(y0),
                "bounding_box_x1": float(x1), "bounding_box_y1": float(y1),
                "center_of_mass_x": cx, "center_of_mass_y": cy,
            }
    return frame, best


def _mask_outside_water(gated: pd.DataFrame, spec: C.DatasetSpec) -> pd.DataFrame:
    """NaN every keypoint of frames whose neck lies outside the water ROI."""
    polygon = _relative_polygon(C.OPEN_WATER_ROI_PX, (spec.image_width_px, spec.image_height_px))
    width, height = C.OPEN_MOVIE_SIZE_PX
    rel = np.column_stack([gated["neck_x"] / width, gated["neck_y"] / height])
    on_land = np.isfinite(rel).all(axis=1) & ~polygon.contains_points(rel)
    out = gated.copy()
    coord_cols = [c for c in out.columns if c.endswith(("_x", "_y"))]
    out.loc[on_land, coord_cols] = np.nan
    return out


def _relative_polygon(
    roi_px: tuple[tuple[float, float], ...], image_size_px: tuple[int, int]
) -> PolygonPath:
    width, height = image_size_px
    return PolygonPath([(x / width, y / height) for x, y in roi_px])


def read_sequences_from_index(
    tra_path: Path, index_csv: Path, **reader_kwargs
) -> list[pd.DataFrame]:
    """Read the frame windows listed for ``tra_path`` in the video index CSV.

    The index has columns ``filename`` (the ``.mp4`` name), ``start`` and
    ``end`` (line numbers).  One DataFrame per window is returned.
    """
    index = pd.read_csv(index_csv)
    movie = tra_path.with_suffix(".mp4").name
    windows = index[index["filename"] == movie]
    return [
        read_charon_tra(tra_path, start_line=int(w.start), max_lines=int(w.end - w.start), **reader_kwargs)
        for w in windows.itertuples()
    ]
