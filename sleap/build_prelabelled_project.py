# ╔══════════════════════════════════════════════════════════════════╗
# ║  penguin_saccades — sleap/build_prelabelled_project              ║
# ║  « start from the model's guess, label only what it got wrong »  ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Builds a SLEAP labelling project from existing predictions. The ║
# ║  suggested frames carry the model's instance, so labelling a new ║
# ║  species means fixing swapped heads and tails instead of placing ║
# ║  every point. Frames whose body axis points against the swimming ║
# ║  direction, the typical error, are over-represented. Label in    ║
# ║  the GUI with python -m sleap.gui.app <project>.                 ║
# ╚══════════════════════════════════════════════════════════════════╝
"""SLEAP labelling project seeded with predicted instances on suggested frames."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sleap.gui.suggestions import SuggestionFrame

from sleap import LabeledFrame, Labels

MOTION_LAG_FRAMES = 6          # 0.1 s at 60 fps
MIN_MOTION_PX = 15.0           # centre displacement over the lag that counts as swimming
MIN_FRAME_SPACING = 30         # suggestions closer than this are near duplicates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, nargs="+", help="<movie>.predictions.slp files")
    parser.add_argument("out", type=Path, help="project to write, e.g. rockhopper_labels.slp")
    parser.add_argument("--video-dir", type=Path, required=True, help="local folder holding the movies")
    parser.add_argument("--n-frames", type=int, default=150)
    parser.add_argument("--reversed-share", type=float, default=0.5,
                        help="fraction of suggestions drawn from frames that look tail-first")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    merged: Labels | None = None
    per_movie = np.full(len(args.predictions), args.n_frames // len(args.predictions))
    per_movie[: args.n_frames % len(args.predictions)] += 1
    for path, n in zip(args.predictions, per_movie):
        labels = Labels.load_file(str(path), video_search=[str(args.video_dir)])
        frames = pick_frames(labels, int(n), args.reversed_share, rng)
        chosen = [LabeledFrame(video=lf.video, frame_idx=lf.frame_idx, instances=[best_instance(lf)])
                  for lf in labels.labeled_frames if lf.frame_idx in frames and lf.predicted_instances]
        subset = Labels(labeled_frames=chosen, videos=labels.videos, skeletons=labels.skeletons)
        if merged is None:
            merged = subset
        else:
            merged.extend_from(subset, unify=True)
        print(f"{path.name}: {len(chosen)} pre-labelled suggestions", flush=True)
    # Suggestions are added last: merging replaces video objects, and a
    # suggestion must point at the video the project keeps.
    merged.suggestions.extend(SuggestionFrame(video=lf.video, frame_idx=lf.frame_idx) for lf in merged.labeled_frames)
    Labels.save_file(merged, str(args.out))
    print(f"{len(merged.videos)} videos, {len(merged.suggestions)} suggestions -> {args.out}")


def pick_frames(labels: Labels, n: int, reversed_share: float, rng: np.random.Generator) -> set[int]:
    """Spaced frames with a bird, ``reversed_share`` of them where the body axis opposes the motion."""
    centre, axis = body_centre_and_axis(labels)
    idx = np.arange(centre.shape[0])
    has_bird = np.isfinite(centre).all(1)
    motion = np.full_like(centre, np.nan)
    motion[:-MOTION_LAG_FRAMES] = centre[MOTION_LAG_FRAMES:] - centre[:-MOTION_LAG_FRAMES]
    speed = np.linalg.norm(motion, axis=1)
    cos = np.einsum("ij,ij->i", motion, axis) / (speed * np.linalg.norm(axis, axis=1))
    moving = np.isfinite(cos) & (speed > MIN_MOTION_PX)
    reversed_pool = idx[moving & (cos < -0.5)]
    other_pool = idx[has_bird & ~(moving & (cos < -0.5))]
    n_rev = min(round(n * reversed_share), reversed_pool.size)
    picked = spaced_sample(reversed_pool, n_rev, rng, set())
    return spaced_sample(other_pool, n - len(picked), rng, picked)


def spaced_sample(pool: np.ndarray, n: int, rng: np.random.Generator, taken: set[int]) -> set[int]:
    """Draw up to ``n`` frames from ``pool``, none within ``MIN_FRAME_SPACING`` of another."""
    out = set(taken)
    for f in rng.permutation(pool):
        if len(out) - len(taken) >= n:
            break
        if all(abs(int(f) - t) >= MIN_FRAME_SPACING for t in out):
            out.add(int(f))
    return out


def body_centre_and_axis(labels: Labels) -> tuple[np.ndarray, np.ndarray]:
    """Per frame: midpoint of neck and caudal, and the caudal-to-neck vector, of the best instance."""
    n_frames = labels.videos[0].num_frames
    centre = np.full((n_frames, 2), np.nan)
    axis = np.full((n_frames, 2), np.nan)
    nodes = [n.name for n in labels.skeletons[0].nodes]
    neck, caudal = nodes.index("neck"), nodes.index("caudal")
    for lf in labels.labeled_frames:
        if not lf.predicted_instances:
            continue
        pts = best_instance(lf).numpy()
        centre[lf.frame_idx] = (pts[neck] + pts[caudal]) / 2
        axis[lf.frame_idx] = pts[neck] - pts[caudal]
    return centre, axis


def best_instance(lf: LabeledFrame):
    """The predicted instance with the highest score."""
    return max(lf.predicted_instances, key=lambda inst: inst.score)


if __name__ == "__main__":
    main()
