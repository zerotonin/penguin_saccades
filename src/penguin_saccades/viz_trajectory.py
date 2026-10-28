# ╔══════════════════════════════════════════════════════════════════╗
# ║  penguin_saccades — viz_trajectory                               ║
# ║  « one segment, every panel a reviewer asks for »                ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Skeleton overlay with equal axes and a scale bar, yaw           ║
# ║  velocity, yaw angle and translation traces with detected        ║
# ║  saccades marked, and sample video frames. OpenCV is imported    ║
# ║  lazily (optional extra).                                        ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Per-segment overview figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from penguin_saccades import constants as C

SKELETON: tuple[tuple[str, str], ...] = (
    ("beak", "neck"), ("neck", "caudal"), ("neck", "left_fin"), ("neck", "right_fin"),
)
KEYPOINT_COLOURS: dict[str, str] = {
    "beak": C.WONG["bluish_green"], "neck": C.WONG["blue"], "caudal": C.WONG["vermilion"],
    "left_fin": C.WONG["sky_blue"], "right_fin": C.WONG["orange"],
}


def plot_segment_overview(
    segment: pd.DataFrame,
    saccades: pd.DataFrame,
    dlc: pd.DataFrame | None = None,
    video_path: Path | None = None,
    frame_times_s: list[float] | None = None,
    skeleton_every: int = 10,
    scale_bar_px: float | None = None,
) -> plt.Figure:
    """A4-landscape overview of one segment.

    Args:
        segment:       Rows of the trajectory table for one segment.
        saccades:      Saccade rows for that segment (any source).
        dlc:           Gated DLC table (pixel coordinates) for the overlay, optional.
        video_path:    Labelled video for the frame strip, optional.
        frame_times_s: Up to eight times for the frame strip.
        skeleton_every: Draw a solid skeleton every n frames.
        scale_bar_px:  Length of the scale bar in pixels (label states metres).
    """
    fig = plt.figure(figsize=C.FIGURE_SIZE_A4_LANDSCAPE)
    gs = GridSpec(5, 4, figure=fig, wspace=0.3, hspace=0.4)
    ax_traj = fig.add_subplot(gs[0:3, 0:2])
    ax_vel = fig.add_subplot(gs[0, 2:4])
    ax_ang = fig.add_subplot(gs[1, 2:4], sharex=ax_vel)
    ax_trans = fig.add_subplot(gs[2, 2:4], sharex=ax_vel)
    frame_axes = [fig.add_subplot(gs[r, c]) for r in (3, 4) for c in range(4)]

    _draw_trajectory(ax_traj, segment, dlc, skeleton_every, scale_bar_px)
    t = segment[C.COL_TIME].to_numpy()
    for col, colour, label in ((C.COL_BODY_VEL, C.BODYPART_COLOURS["body"], "body"),
                               (C.COL_HEAD_VEL, C.BODYPART_COLOURS["head"], "head")):
        if segment[col].notna().any():
            ax_vel.plot(t, segment[col], color=colour, lw=0.8, label=label)
    for col, colour in ((C.COL_BODY_YAW, C.BODYPART_COLOURS["body"]), (C.COL_HEAD_YAW, C.BODYPART_COLOURS["head"])):
        if segment[col].notna().any():
            ax_ang.plot(t, np.rad2deg(segment[col]), color=colour, lw=0.8)
    ax_trans.plot(t, segment[C.COL_TRANS], color="k", lw=0.8)
    for _, s in saccades.iterrows():
        colour = C.BODYPART_COLOURS.get(s.get(C.COL_SOURCE, "body"), "k")
        ax_vel.axvspan(s["start_idx"] / _fps(segment), s["stop_idx"] / _fps(segment), color=colour, alpha=0.15, lw=0)
    ax_vel.axhline(C.SACCADE_VELOCITY_THRESHOLD_DEGPS, color="k", ls=":", lw=0.5)
    ax_vel.axhline(-C.SACCADE_VELOCITY_THRESHOLD_DEGPS, color="k", ls=":", lw=0.5)
    ax_vel.set_ylabel(r"yaw velocity ($^\circ\,s^{-1}$)")
    ax_vel.legend(frameon=False, loc="upper right")
    ax_ang.set_ylabel(r"yaw ($^\circ$)")
    ax_trans.set_ylabel(r"speed (m$\,s^{-1}$)")
    ax_trans.set_xlabel("time (s)")
    for ax in (ax_vel, ax_ang):
        ax.tick_params(labelbottom=False)

    frames = _read_frames(video_path, frame_times_s) if video_path else []
    for ax, img in zip(frame_axes, frames):
        ax.imshow(img)
    for ax in frame_axes:
        ax.axis("off")
    return fig


def _fps(segment: pd.DataFrame) -> float:
    t = segment[C.COL_TIME].to_numpy()
    f = segment[C.COL_FRAME].to_numpy()
    return float((f[-1] - f[0]) / (t[-1] - t[0])) if len(t) > 1 and t[-1] > t[0] else 1.0


def _draw_trajectory(ax: plt.Axes, segment: pd.DataFrame, dlc: pd.DataFrame | None,
                     every: int, scale_bar_px: float | None) -> None:
    if dlc is not None:
        for i in range(len(dlc)):
            alpha = 1.0 if i % every == 0 else 0.08
            for a, b in SKELETON:
                xa, ya, xb, yb = dlc[f"{a}_x"].iat[i], dlc[f"{a}_y"].iat[i], dlc[f"{b}_x"].iat[i], dlc[f"{b}_y"].iat[i]
                if np.all(np.isfinite([xa, ya, xb, yb])):
                    ax.plot([xa, xb], [ya, yb], color="k", lw=0.6, alpha=alpha)
            if i % every == 0:
                for kp, colour in KEYPOINT_COLOURS.items():
                    ax.scatter(dlc[f"{kp}_x"].iat[i], dlc[f"{kp}_y"].iat[i], s=8, color=colour, lw=0)
        ax.invert_yaxis()
        if scale_bar_px:
            x0, y0 = ax.get_xlim()[0], ax.get_ylim()[0]
            ax.plot([x0, x0 + scale_bar_px], [y0, y0], color="k", lw=2)
    else:
        ax.plot(segment[C.COL_X], segment[C.COL_Y], color="k", lw=0.8)
        ax.invert_yaxis()
        ax.plot([segment[C.COL_X].min(), segment[C.COL_X].min() + 1.0],
                [segment[C.COL_Y].max()] * 2, color="k", lw=2)
        ax.text(segment[C.COL_X].min(), segment[C.COL_Y].max(), "1 m", va="bottom", fontsize=C.FONT_SIZE)
    ax.set_aspect("equal")
    ax.axis("off")


def _read_frames(video_path: Path, times_s: list[float] | None) -> list[np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or C.CLOSED_FRAME_RATE_HZ
    if times_s is None:
        times_s = list(np.linspace(0, (n - 1) / fps, 8))
    frames = []
    for t in times_s[:8]:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, img = cap.read()
        if ok:
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames
