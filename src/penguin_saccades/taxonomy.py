# ╔══════════════════════════════════════════════════════════════════╗
# ║  penguin_saccades — taxonomy                                     ║
# ║  « associated, free head, free body: one rule, stated once »     ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  A head saccade is "associated" when a body saccade of the same  ║
# ║  direction peaks within ASSOCIATION_WINDOW_S; otherwise it is a  ║
# ║  free head saccade. Body saccades without a partner are free     ║
# ║  body saccades. Interval classes follow from which saccade       ║
# ║  source bounds them: body, head, or any.                         ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Head and body saccade association and interval classes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from penguin_saccades import constants as C
from penguin_saccades.saccades import intersaccadic_intervals

GROUP_KEYS: tuple[str, ...] = (C.COL_DATASET, C.COL_SPECIES, C.COL_TRAJECTORY, C.COL_SEGMENT)


# ┌────────────────────────────────────────────────────────────────────────┐
# │ Association  « nearest same-direction body saccade within the window » │
# └────────────────────────────────────────────────────────────────────────┘
def associate_head_body(
    head: pd.DataFrame, body: pd.DataFrame, window_s: float = C.ASSOCIATION_WINDOW_S
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Label head saccades as associated or free and body saccades accordingly.

    Adds to ``head``: ``saccade_type``, ``partner_body_idx`` (peak frame
    of the partner, NaN if free) and ``head_lead_s`` (body peak minus
    head peak; positive means the head moved first).  Adds to ``body``:
    ``saccade_type``.
    """
    head = head.copy()
    body = body.copy()
    head[C.COL_SACCADE_TYPE] = "free_head"
    head["partner_body_idx"] = np.nan
    head["head_lead_s"] = np.nan
    body[C.COL_SACCADE_TYPE] = "free_body"
    partner_rows: set[int] = set()
    for key, h_grp in head.groupby(list(GROUP_KEYS), sort=False):
        b_grp = body[(body[list(GROUP_KEYS)] == pd.Series(key, index=GROUP_KEYS)).all(axis=1)]
        for i, h in h_grp.iterrows():
            same_dir = b_grp[b_grp["direction"] == h["direction"]]
            if same_dir.empty:
                continue
            dt = same_dir["peak_time_s"] - h["peak_time_s"]
            j = dt.abs().idxmin()
            if abs(dt[j]) <= window_s:
                head.at[i, C.COL_SACCADE_TYPE] = "associated"
                head.at[i, "partner_body_idx"] = same_dir.at[j, "peak_idx"]
                head.at[i, "head_lead_s"] = dt[j]
                partner_rows.add(j)
    if partner_rows:
        body.loc[sorted(partner_rows), C.COL_SACCADE_TYPE] = "associated"
    return head, body


# ┌────────────────────────────────────────────────────────────────┐
# │ Interval classes  « which saccade source bounds the interval » │
# └────────────────────────────────────────────────────────────────┘
def classified_intervals(
    trajectories: pd.DataFrame,
    head: pd.DataFrame,
    body: pd.DataFrame,
    frame_rates: dict[str, float],
) -> pd.DataFrame:
    """Intersaccadic intervals per segment for classes ``body``, ``head`` and ``any``."""
    parts: list[pd.DataFrame] = []
    for key, seg in trajectories.groupby(list(GROUP_KEYS), sort=False):
        fps = frame_rates[key[0]]
        frames = seg[C.COL_FRAME].to_numpy()
        n = len(frames)
        sel = pd.Series(key, index=GROUP_KEYS)
        h = head[(head[list(GROUP_KEYS)] == sel).all(axis=1)] if len(head) else head
        b = body[(body[list(GROUP_KEYS)] == sel).all(axis=1)] if len(body) else body
        has_head = seg[C.COL_HEAD_YAW].notna().any()
        sources = {"body": b, "any": pd.concat([h, b], ignore_index=True)}
        if has_head:
            sources["head"] = h
        for cls, sacc in sources.items():
            local = sacc.copy()
            local["start_idx"] = np.searchsorted(frames, local["start_idx"].to_numpy())
            local["stop_idx"] = np.searchsorted(frames, local["stop_idx"].to_numpy())
            ints = intersaccadic_intervals(local, n, fps)
            ints["start_idx"] = frames[ints["start_idx"].to_numpy(dtype=int)]
            ints["stop_idx"] = frames[ints["stop_idx"].to_numpy(dtype=int)]
            ints[C.COL_INTERVAL_CLASS] = cls
            for name, value in zip(GROUP_KEYS, key):
                ints[name] = value
            parts.append(ints)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)
