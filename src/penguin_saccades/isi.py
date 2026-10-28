# ╔══════════════════════════════════════════════════════════════════╗
# ║  penguin_saccades — isi                                          ║
# ║  « what happens between saccades is the actual test »            ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Per-interval rotation and translation content, the saccade      ║
# ║  budget (fraction of time and of yaw inside saccades), one-      ║
# ║  second chunks for interval-triggered averages, and head-body    ║
# ║  counter- rotation. These answer revision items 1.1 and 1.11.    ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Intersaccadic-interval content, saccade budget and head compensation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from penguin_saccades import constants as C

GROUP_KEYS: tuple[str, ...] = (C.COL_DATASET, C.COL_SPECIES, C.COL_TRAJECTORY, C.COL_SEGMENT)


# ┌─────────────────────────────────────────────────────────────────────┐
# │ Interval content  « rotation and translation inside each interval » │
# └─────────────────────────────────────────────────────────────────────┘
def interval_content(trajectories: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    """Add rotation and translation summaries to every interval row.

    New columns: ``net_yaw_deg`` (signed end minus start), ``path_yaw_deg``
    (accumulated absolute yaw), ``median_abs_body_speed_degps``,
    ``max_abs_body_speed_degps``, ``median_translation_mps`` and, where
    head data exist, ``median_abs_head_speed_degps``.
    """
    rows: list[dict] = []
    for key, seg in trajectories.groupby(list(GROUP_KEYS), sort=False):
        sel = intervals[(intervals[list(GROUP_KEYS)] == pd.Series(key, index=GROUP_KEYS)).all(axis=1)]
        if sel.empty:
            continue
        frames = seg[C.COL_FRAME].to_numpy()
        yaw_deg = np.rad2deg(seg[C.COL_BODY_YAW].to_numpy())
        body_v = seg[C.COL_BODY_VEL].to_numpy()
        head_v = seg[C.COL_HEAD_VEL].to_numpy()
        trans = seg[C.COL_TRANS].to_numpy()
        for _, it in sel.iterrows():
            a, b = np.searchsorted(frames, int(it["start_idx"])), np.searchsorted(frames, int(it["stop_idx"]))
            sl = slice(a, b + 1)
            row = it.to_dict()
            row.update({
                "net_yaw_deg": yaw_deg[b] - yaw_deg[a],
                "path_yaw_deg": np.nansum(np.abs(np.diff(yaw_deg[sl]))),
                "median_abs_body_speed_degps": np.nanmedian(np.abs(body_v[sl])),
                "max_abs_body_speed_degps": np.nanmax(np.abs(body_v[sl])),
                "median_translation_mps": np.nanmedian(trans[sl]),
                "median_abs_head_speed_degps": _nanmedian_or_nan(np.abs(head_v[sl])),
            })
            rows.append(row)
    return pd.DataFrame(rows)


def _nanmedian_or_nan(x: np.ndarray) -> float:
    return float(np.nanmedian(x)) if np.isfinite(x).any() else np.nan


# ┌───────────────────────────────────────────────────────────────────┐
# │ Saccade budget  « fraction of time and of yaw spent in saccades » │
# └───────────────────────────────────────────────────────────────────┘
def saccade_budget(trajectories: pd.DataFrame, saccades: pd.DataFrame, yaw_col: str) -> pd.DataFrame:
    """Per segment: fraction of recording time and of accumulated absolute yaw inside saccades.

    A strict saccadic strategy puts nearly all yaw inside saccades and
    little of the time; a flexible one does not.
    """
    rows: list[dict] = []
    for key, seg in trajectories.groupby(list(GROUP_KEYS), sort=False):
        yaw = np.rad2deg(seg[yaw_col].to_numpy())
        if not np.isfinite(yaw).any():
            continue
        frames = seg[C.COL_FRAME].to_numpy()
        sel = saccades[(saccades[list(GROUP_KEYS)] == pd.Series(key, index=GROUP_KEYS)).all(axis=1)]
        in_sacc = np.zeros(len(frames), dtype=bool)
        for s, e in zip(sel["start_idx"], sel["stop_idx"]):
            a, b = np.searchsorted(frames, int(s)), np.searchsorted(frames, int(e))
            in_sacc[a:b + 1] = True
        step = np.abs(np.diff(yaw))
        step_in = in_sacc[1:] & in_sacc[:-1]
        total_yaw = np.nansum(step)
        rows.append({
            **dict(zip(GROUP_KEYS, key)),
            "n_frames": len(frames),
            "n_saccades": len(sel),
            "time_fraction_in_saccades": in_sacc.mean(),
            "yaw_fraction_in_saccades": np.nansum(step[step_in]) / total_yaw if total_yaw > 0 else np.nan,
            "total_path_yaw_deg": total_yaw,
        })
    return pd.DataFrame(rows)


# ┌────────────────────────────────────────────────────────────────────────┐
# │ Interval chunks  « one-second pieces for interval-triggered averages » │
# └────────────────────────────────────────────────────────────────────────┘
def chunk_intervals(
    trajectories: pd.DataFrame,
    intervals: pd.DataFrame,
    frame_rates: dict[str, float],
    chunk_s: float = C.ISI_CHUNK_S,
) -> pd.DataFrame:
    """Cut every interval into chunks of ``chunk_s`` and return them in long format.

    Short remainders are centred with NaN padding.  Angles are offset to
    the first finite value of each chunk.  Columns: group keys,
    ``interval_class``, ``interval_id``, ``chunk_id``, ``variable``,
    ``frame`` (0-based within the chunk), ``value``.
    """
    variables = {
        "body_yaw_deg": (C.COL_BODY_YAW, True),
        "body_yaw_speed_degps": (C.COL_BODY_VEL, False),
        "head_yaw_deg": (C.COL_HEAD_YAW, True),
        "head_yaw_speed_degps": (C.COL_HEAD_VEL, False),
        "translational_speed_mps": (C.COL_TRANS, False),
    }
    rows: list[pd.DataFrame] = []
    for key, seg in trajectories.groupby(list(GROUP_KEYS), sort=False):
        fps = frame_rates[key[0]]
        size = round(chunk_s * fps)
        frames = seg[C.COL_FRAME].to_numpy()
        sel = intervals[(intervals[list(GROUP_KEYS)] == pd.Series(key, index=GROUP_KEYS)).all(axis=1)]
        for interval_id, it in sel.iterrows():
            a = np.searchsorted(frames, int(it["start_idx"]))
            b = np.searchsorted(frames, int(it["stop_idx"])) + 1
            for chunk_id, c0 in enumerate(range(a, b, size)):
                piece = slice(c0, min(c0 + size, b))
                for var, (col, is_angle) in variables.items():
                    values = seg[col].to_numpy()[piece]
                    if is_angle:
                        values = np.rad2deg(values)
                        finite = values[np.isfinite(values)]
                        if finite.size:
                            values = values - finite[0]
                    padded = _centre(values, size)
                    rows.append(pd.DataFrame({
                        **dict(zip(GROUP_KEYS, key)),
                        C.COL_INTERVAL_CLASS: it.get(C.COL_INTERVAL_CLASS, "any"),
                        "interval_id": interval_id,
                        "chunk_id": chunk_id,
                        "variable": var,
                        "frame": np.arange(size),
                        "value": padded,
                    }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _centre(values: np.ndarray, size: int) -> np.ndarray:
    out = np.full(size, np.nan)
    n = min(len(values), size)
    left = (size - n) // 2
    out[left:left + n] = values[:n]
    return out


# ┌────────────────────────────────────────────────────────────────────────┐
# │ Head compensation  « does the head counter-rotate residual body yaw? » │
# └────────────────────────────────────────────────────────────────────────┘
def head_body_compensation(
    trajectories: pd.DataFrame,
    intervals: pd.DataFrame,
    min_frames: int = C.MIN_COMPENSATION_FRAMES,
) -> pd.DataFrame:
    """Per interval: regression of head yaw velocity on body yaw velocity.

    Head yaw is measured in world coordinates, so a head rigidly carried
    by the body gives ``slope_head_per_body`` of 1 and perfect gaze
    stabilisation gives 0.  ``slope_neck_per_body`` is the same fit for
    the neck (head minus body) velocity: -1 means full compensation,
    0 none.
    """
    rows: list[dict] = []
    for key, seg in trajectories.groupby(list(GROUP_KEYS), sort=False):
        if not seg[C.COL_HEAD_VEL].notna().any():
            continue
        sel = intervals[(intervals[list(GROUP_KEYS)] == pd.Series(key, index=GROUP_KEYS)).all(axis=1)]
        frames = seg[C.COL_FRAME].to_numpy()
        body_v = seg[C.COL_BODY_VEL].to_numpy()
        head_v = seg[C.COL_HEAD_VEL].to_numpy()
        for interval_id, it in sel.iterrows():
            a = np.searchsorted(frames, int(it["start_idx"]))
            b = np.searchsorted(frames, int(it["stop_idx"])) + 1
            x, y = body_v[a:b], head_v[a:b]
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < min_frames or np.std(x[ok]) == 0:
                continue
            slope, _intercept = np.polyfit(x[ok], y[ok], 1)
            rows.append({
                **dict(zip(GROUP_KEYS, key)),
                C.COL_INTERVAL_CLASS: it.get(C.COL_INTERVAL_CLASS, "any"),
                "interval_id": interval_id,
                "n_frames": int(ok.sum()),
                "pearson_r": float(np.corrcoef(x[ok], y[ok])[0, 1]),
                "slope_head_per_body": float(slope),
                "slope_neck_per_body": float(slope) - 1.0,
            })
    return pd.DataFrame(rows)
