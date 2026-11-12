# ╔══════════════════════════════════════════════════════════════════╗
# ║  penguin_saccades — viz_overlay                                  ║
# ║  « the skeleton on the bird, and the pipeline's verdict »        ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Renders a clip of an open-water movie with every SLEAP instance ║
# ║  drawn on the frame, low-score keypoints hollow, and a badge     ║
# ║  that says whether the pipeline kept, bridged or rejected the    ║
# ║  frame and whether it lies inside a detected saccade. The crop   ║
# ║  follows the bird so a 200 px penguin fills the picture. OpenCV  ║
# ║  is imported lazily (optional extra).                            ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Render a movie clip with the SLEAP skeletons and the pipeline's verdict on every frame."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from penguin_saccades import config
from penguin_saccades import constants as C
from penguin_saccades.viz_trajectory import KEYPOINT_COLOURS, SKELETON

CROP_PX: tuple[int, int] = (720, 480)
PLAYBACK_SLOWDOWN: int = 3
CENTRE_SMOOTHING_FRAMES: float = 15.0
CONTACT_SHEET_FRAMES: int = 8
STATUS_COLOURS: dict[str, str] = {
    "usable": C.WONG["bluish_green"], "bridged": C.WONG["orange"], "rejected": C.WONG["vermilion"],
}


@dataclass
class ClipAnnotation:
    """Per-frame verdicts of the pipeline for one movie, indexed by movie frame."""

    status: dict[int, str] = field(default_factory=dict)         # usable | bridged; absent = rejected
    saccade: dict[int, str] = field(default_factory=dict)        # "body saccade" | "head saccade"

    @classmethod
    def from_output_dir(cls, out: Path, stem: str) -> ClipAnnotation:
        """Read trajectories.csv and saccades.csv of a pipeline run for the movie ``stem``."""
        trajectories = pd.read_csv(out / "trajectories.csv")
        rows = trajectories[trajectories["source_file"] == f"{stem}.analysis.h5"]
        status = {
            int(f): "bridged" if b else "usable"
            for f, b in zip(rows[C.COL_FRAME], rows[C.COL_BRIDGED].astype(bool))
        }
        saccade: dict[int, str] = {}
        saccades_path = out / "saccades.csv"
        if saccades_path.exists() and not rows.empty:
            saccades = pd.read_csv(saccades_path)
            mine = saccades[saccades[C.COL_TRAJECTORY].isin(rows[C.COL_TRAJECTORY].unique())]
            for _, s in mine.iterrows():
                for f in range(int(s["start_idx"]), int(s["stop_idx"]) + 1):
                    saccade[f] = f"{s[C.COL_SOURCE]} saccade"
        return cls(status, saccade)


# ┌────────────────────────────────────────────────────────────┐
# │ Rendering  « frames in, annotated crops out »              │
# └────────────────────────────────────────────────────────────┘
def render_overlay(
    movie: Path,
    predictions: Path,
    start: int,
    end: int,
    out: Path,
    annotation: ClipAnnotation | None = None,
    score_threshold: float = C.SLEAP_SCORE_THRESHOLD,
) -> Path:
    """Write ``out`` (mp4, slowed down) and a contact sheet PNG next to it.

    Args:
        movie:        Open-water movie the predictions were made on.
        predictions:  SLEAP ``*.analysis.h5`` for that movie (movie pixels).
        start, end:   First and last movie frame of the clip (inclusive).
        out:          Output mp4; the contact sheet gets the same stem.
        annotation:   Pipeline verdicts; without it only the skeletons are drawn.
    """
    import cv2

    tracks, scores, nodes = _read_predictions(predictions)
    centres = _crop_centres(tracks, scores, start, end)
    cap = cv2.VideoCapture(str(movie))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open movie: {movie}")
    fps = cap.get(cv2.CAP_PROP_FPS) or C.DATASET_OPEN.frame_rate_hz
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    writer = _writer(cv2, out, fps / PLAYBACK_SLOWDOWN)
    sheet_at = set(np.linspace(start, end, CONTACT_SHEET_FRAMES).round().astype(int))
    sheet: list[np.ndarray] = []
    for frame_idx in range(start, end + 1):
        ok, image = cap.read()
        if not ok:
            break
        crop = _crop(image, centres[frame_idx - start])
        _draw_instances(cv2, crop, tracks[..., frame_idx], scores[..., frame_idx], nodes, score_threshold)
        _draw_badge(cv2, crop, frame_idx, fps, annotation, tracks[..., frame_idx], scores[..., frame_idx], nodes)
        writer.write(crop)
        if frame_idx in sheet_at:
            sheet.append(crop.copy())
    cap.release()
    writer.release()
    cv2.imwrite(str(out.with_suffix(".png")), _contact_sheet(sheet))
    return out


def _read_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with h5py.File(path, "r") as f:
        tracks = f["tracks"][:]                  # (n_tracks, 2, n_nodes, n_frames)
        scores = f["point_scores"][:]            # (n_tracks, n_nodes, n_frames)
        nodes = [n.decode() if isinstance(n, bytes) else str(n) for n in f["node_names"][:]]
    return tracks, scores, nodes


def _best_track(scores_frame: np.ndarray) -> int | None:
    """Index of the instance with the highest mean score in one frame, as the ingest picks it."""
    finite = np.isfinite(scores_frame)
    if not finite.any():
        return None
    mean = np.where(finite, scores_frame, 0).sum(axis=1) / np.maximum(finite.sum(axis=1), 1)
    return int(np.argmax(np.where(finite.any(axis=1), mean, -np.inf)))


def _crop_centres(tracks: np.ndarray, scores: np.ndarray, start: int, end: int) -> np.ndarray:
    """Smoothed neck position of the best instance per clip frame; the frame centre where there is none."""
    centre = np.full((end - start + 1, 2), np.nan)
    neck = C.BODY_VECTOR[1]
    for i, frame_idx in enumerate(range(start, end + 1)):
        best = _best_track(scores[..., frame_idx])
        if best is not None:
            centre[i] = tracks[best, :, C.DLC_KEYPOINTS.index(neck), frame_idx]
    if np.isfinite(centre).all(axis=1).any():
        for k in range(2):
            col = pd.Series(centre[:, k]).interpolate(limit_direction="both").to_numpy()
            centre[:, k] = gaussian_filter1d(col, CENTRE_SMOOTHING_FRAMES)
    else:
        centre[:] = np.asarray(C.OPEN_MOVIE_SIZE_PX) / 2
    return centre


def _crop(image: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """Crop CROP_PX around ``centre``, shifted inward at the frame edges."""
    h, w = image.shape[:2]
    cw, ch = CROP_PX
    x0 = int(np.clip(round(centre[0] - cw / 2), 0, max(w - cw, 0)))
    y0 = int(np.clip(round(centre[1] - ch / 2), 0, max(h - ch, 0)))
    return _Crop(np.ascontiguousarray(image[y0:y0 + ch, x0:x0 + cw]), x0, y0)


class _Crop(np.ndarray):
    """Image crop that remembers where it sits in the full frame."""

    def __new__(cls, image: np.ndarray, x0: int, y0: int) -> _Crop:
        obj = np.asarray(image).view(cls)
        obj.x0, obj.y0 = x0, y0
        return obj

    def __array_finalize__(self, obj) -> None:
        self.x0 = getattr(obj, "x0", 0)
        self.y0 = getattr(obj, "y0", 0)


def _draw_instances(cv2, crop: _Crop, tracks_frame: np.ndarray, scores_frame: np.ndarray,
                    nodes: list[str], threshold: float) -> None:
    """Every instance of the frame: the best one in colour, the others grey."""
    best = _best_track(scores_frame)
    for t in range(tracks_frame.shape[0]):
        pts = tracks_frame[t] - np.array([[crop.x0], [crop.y0]])
        if not np.isfinite(pts).any():
            continue
        is_best = t == best
        for a, b in SKELETON:
            pa, pb = pts[:, nodes.index(a)], pts[:, nodes.index(b)]
            if np.isfinite(pa).all() and np.isfinite(pb).all():
                cv2.line(crop, _pt(pa), _pt(pb), (255, 255, 255) if is_best else (160, 160, 160), 2 if is_best else 1,
                         cv2.LINE_AA)
        for j, node in enumerate(nodes):
            p = pts[:, j]
            if not np.isfinite(p).all():
                continue
            colour = _bgr(KEYPOINT_COLOURS[node]) if is_best else (160, 160, 160)
            low = not (scores_frame[t, j] >= threshold)
            cv2.circle(crop, _pt(p), 7, colour, 2 if low else -1, cv2.LINE_AA)


def _draw_badge(cv2, crop: _Crop, frame_idx: int, fps: float, annotation: ClipAnnotation | None,
                tracks_frame: np.ndarray, scores_frame: np.ndarray, nodes: list[str]) -> None:
    """Frame number, time, head-body angle of the best instance, and the pipeline's verdict."""
    lines = [(f"frame {frame_idx}   t = {frame_idx / fps:6.2f} s", (255, 255, 255))]
    best = _best_track(scores_frame)
    if best is not None:
        angle = _head_body_angle_deg(tracks_frame[best], nodes)
        if np.isfinite(angle):
            lines.append((f"head-body angle {angle:+.0f} deg", (255, 255, 255)))
    if annotation is not None:
        status = annotation.status.get(frame_idx, "rejected")
        lines.append((status, _bgr(STATUS_COLOURS[status])))
        if frame_idx in annotation.saccade:
            lines.append((annotation.saccade[frame_idx].upper(), _bgr(C.WONG["yellow"])))
    for i, (text, colour) in enumerate(lines):
        y = 24 + 22 * i
        cv2.putText(crop, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(crop, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 1, cv2.LINE_AA)


def _head_body_angle_deg(points: np.ndarray, nodes: list[str]) -> float:
    """Angle of the neck-to-beak vector relative to the caudal-to-neck vector."""
    def vec(tail: str, tip: str) -> np.ndarray:
        return points[:, nodes.index(tip)] - points[:, nodes.index(tail)]
    head, body = vec(*C.HEAD_VECTOR), vec(*C.BODY_VECTOR)
    if not (np.isfinite(head).all() and np.isfinite(body).all()):
        return np.nan
    return float(np.rad2deg(np.arctan2(head[1], head[0]) - np.arctan2(body[1], body[0]) + np.pi) % 360 - 180)


def _contact_sheet(frames: list[np.ndarray]) -> np.ndarray:
    """Two rows of annotated crops, for a quick look without playing the clip."""
    if not frames:
        return np.zeros((*CROP_PX[::-1], 3), dtype=np.uint8)
    per_row = int(np.ceil(len(frames) / 2))
    blank = np.zeros_like(frames[0])
    padded = [np.asarray(f) for f in frames] + [blank] * (2 * per_row - len(frames))
    return np.vstack([np.hstack(padded[r * per_row:(r + 1) * per_row]) for r in range(2)])


def _writer(cv2, out: Path, fps: float):
    out.parent.mkdir(parents=True, exist_ok=True)
    for codec in ("mp4v", "avc1"):      # mp4v first: avc1 needs a hardware encoder OpenCV rarely has
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*codec), fps, CROP_PX)
        if writer.isOpened():
            return writer
    raise RuntimeError(f"No mp4 codec available to write {out}")


def _pt(p: np.ndarray) -> tuple[int, int]:
    return round(float(p[0])), round(float(p[1]))


def _bgr(hex_colour: str) -> tuple[int, int, int]:
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    return b, g, r


# ┌────────────────────────────────────────────────────────────┐
# │ Entry point  « argparse »                                  │
# └────────────────────────────────────────────────────────────┘
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="penguin-saccades-overlay", description=__doc__)
    parser.add_argument("movie", type=Path)
    parser.add_argument("predictions", type=Path, help="SLEAP *.analysis.h5 of that movie")
    parser.add_argument("--start", type=int, required=True, help="first movie frame")
    parser.add_argument("--end", type=int, required=True, help="last movie frame (inclusive)")
    parser.add_argument("--out", type=Path, required=True, help="mp4 to write; a PNG contact sheet goes next to it")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="pipeline run whose verdicts to show; defaults to the profile output_dir")
    parser.add_argument("--profile", default=None, help="local_paths.json profile to use")
    parser.add_argument("--no-verdicts", action="store_true", help="draw the skeletons only")
    args = parser.parse_args(argv)
    if args.profile:
        os.environ[config.ENV_PROFILE] = args.profile
    annotation = None
    if not args.no_verdicts:
        try:
            out_dir = args.output_dir or config.output_dir()
        except config.PathConfigError as err:
            print(f"error: {err}", file=sys.stderr)
            return 1
        annotation = ClipAnnotation.from_output_dir(out_dir, args.predictions.name.replace(".analysis.h5", ""))
    path = render_overlay(args.movie, args.predictions, args.start, args.end, args.out, annotation)
    print(f"[overlay] frames {args.start}-{args.end} -> {path} (+ {path.with_suffix('.png').name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
