"""Synthetic penguins: known headings, known saccades, known speeds."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from penguin_saccades import constants as C

IM_W, IM_H = C.DATASET_OPEN.image_width_px, C.DATASET_OPEN.image_height_px


def piecewise_heading(t: np.ndarray, saccades: list[tuple[float, float]], ramp_s: float = 0.15) -> np.ndarray:
    """Heading (rad) that is flat except for smooth ramps of given amplitude (deg) at given times."""
    heading = np.zeros_like(t)
    for t0, amp_deg in saccades:
        ramp = (t >= t0) & (t < t0 + ramp_s)
        phase = (t[ramp] - t0) / ramp_s
        heading[ramp] += np.deg2rad(amp_deg) * (1 - np.cos(np.pi * phase)) / 2
        heading[t >= t0 + ramp_s] += np.deg2rad(amp_deg)
    return heading


def positions(heading: np.ndarray, speed_mps: float, fps: float) -> tuple[np.ndarray, np.ndarray]:
    dx = speed_mps * np.cos(heading) / fps
    dy = speed_mps * np.sin(heading) / fps
    return np.cumsum(dx) + 2.0, np.cumsum(dy) + 2.0


def open_track(
    duration_s: float = 10.0,
    saccades: list[tuple[float, float]] = ((5.0, 60.0),),
    speed_mps: float = 2.0,
    fps: float = C.DATASET_OPEN.frame_rate_hz,
    drop_frames: tuple[int, ...] = (),
) -> pd.DataFrame:
    """Charon-style detections in relative image coordinates, indexed by frame."""
    n = int(duration_s * fps)
    t = np.arange(n) / fps
    x_m, y_m = positions(piecewise_heading(t, list(saccades)), speed_mps, fps)
    rel_x = x_m / (IM_W * C.OPEN_M_PER_PX)
    rel_y = y_m / (IM_H * C.OPEN_M_PER_PX)
    df = pd.DataFrame({
        "name": "king", "quality": 0.99,
        "bounding_box_x0": rel_x - 0.01, "bounding_box_y0": rel_y - 0.01,
        "bounding_box_x1": rel_x + 0.01, "bounding_box_y1": rel_y + 0.01,
        "center_of_mass_x": rel_x, "center_of_mass_y": rel_y,
    }, index=pd.Index(np.arange(n), name=C.COL_FRAME))
    return df.drop(index=list(drop_frames))


def dlc_track(
    duration_s: float = 10.0,
    body_saccades: list[tuple[float, float]] = ((5.0, 60.0),),
    head_saccades: list[tuple[float, float]] = ((4.9, 50.0), (2.0, 30.0)),
    speed_mps: float = 1.0,
    fps: float = C.DATASET_CLOSED.frame_rate_hz,
    px_per_m: float = 200.0,
    body_len_m: float = C.GENTOO_BODY_LENGTH_M,
    low_likelihood_frames: tuple[int, ...] = (),
) -> pd.DataFrame:
    """DeepLabCut-style flat table (``<kp>_x``, ``<kp>_y``, ``<kp>_likelihood``) in pixels."""
    n = int(duration_s * fps)
    t = np.arange(n) / fps
    body = piecewise_heading(t, list(body_saccades))
    head = piecewise_heading(t, list(head_saccades))  # head yaw relative to body
    x_m, y_m = positions(body, speed_mps, fps)
    neck_x, neck_y = x_m * px_per_m, y_m * px_per_m
    body_px, head_px = 0.7 * body_len_m * px_per_m, 0.3 * body_len_m * px_per_m
    caudal_x, caudal_y = neck_x - body_px * np.cos(body), neck_y - body_px * np.sin(body)
    beak_x = neck_x + head_px * np.cos(body + head)
    beak_y = neck_y + head_px * np.sin(body + head)
    cols = {
        "beak_x": beak_x, "beak_y": beak_y,
        "neck_x": neck_x, "neck_y": neck_y,
        "left_fin_x": neck_x - 20, "left_fin_y": neck_y + 20,
        "right_fin_x": neck_x + 20, "right_fin_y": neck_y + 20,
        "caudal_x": caudal_x, "caudal_y": caudal_y,
    }
    df = pd.DataFrame(cols)
    for kp in C.DLC_KEYPOINTS:
        df[f"{kp}_likelihood"] = 0.99
    for f in low_likelihood_frames:
        df.loc[f, "beak_likelihood"] = 0.1
    df.index.name = C.COL_FRAME
    return df


def write_dlc_csv(df: pd.DataFrame, path) -> None:
    """Write a flat DLC table in the three-header-row CSV layout DeepLabCut produces."""
    scorer = "DLC_resnet50_penguinTest"
    cols = pd.MultiIndex.from_tuples(
        [(scorer, c.rsplit("_", 1)[0], c.rsplit("_", 1)[1]) for c in df.columns],
        names=["scorer", "bodyparts", "coords"],
    )
    out = df.copy()
    out.columns = cols
    out.to_csv(path)


@pytest.fixture
def open_df() -> pd.DataFrame:
    return open_track()


@pytest.fixture
def dlc_df() -> pd.DataFrame:
    return dlc_track()
