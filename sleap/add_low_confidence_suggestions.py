# ╔══════════════════════════════════════════════════════════════════╗
# ║  penguin_saccades — sleap/add_low_confidence_suggestions         ║
# ║  « a second round aimed at the frames the model doubts »         ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Extends a labelling project with suggested frames where the     ║
# ║  current model found the bird but gave the beak or the tail a    ║
# ║  low score, restricted to the stretches the curated charon       ║
# ║  trajectories cover. Those are the frames the saccade pipeline   ║
# ║  loses, so labelling them moves the detector where it matters.   ║
# ║  Each suggestion carries the model's instance, to be corrected   ║
# ║  rather than placed from scratch.                                ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Add low-confidence frames inside the charon windows as pre-labelled suggestions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from build_prelabelled_project import MIN_FRAME_SPACING, best_instance, spaced_sample
from sleap.gui.suggestions import SuggestionFrame
from sleap.instance import PredictedInstance

from sleap import LabeledFrame, Labels

DOUBTFUL_NODES = ("beak", "caudal")
SUGGESTION_GROUP = 2        # shown in the GUI's suggestion list; round one has no group


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path, help="labelling project to extend")
    parser.add_argument("out", type=Path, help="extended project to write")
    parser.add_argument("--predictions", type=Path, nargs="+", required=True,
                        help="<movie>.predictions.slp from the model whose doubts to label")
    parser.add_argument("--index-csv", type=Path, required=True,
                        help="video index with filename, start, end rows (the charon windows)")
    parser.add_argument("--n-frames", type=int, default=100)
    parser.add_argument("--score-threshold", type=float, default=0.3,
                        help="a beak or tail below this score makes a frame doubtful")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    project = Labels.load_file(str(args.project))
    videos = {Path(v.filename).name: v for v in project.videos}
    windows = charon_windows(args.index_csv)
    pools: dict[Path, np.ndarray] = {}
    for path in args.predictions:
        movie = path.name.replace(".predictions.slp", ".mp4")
        if movie not in videos:
            raise SystemExit(f"{movie} is not a video of {args.project}")
        taken = {lf.frame_idx for lf in project.labeled_frames if lf.video is videos[movie]}
        labels = Labels.load_file(str(path), video_search=[str(Path(videos[movie].filename).parent)])
        pools[path] = doubtful_frames(labels, args.score_threshold, windows[movie], taken)
        print(f"{path.name}: {pools[path].size} doubtful frames inside the charon windows", flush=True)

    for path, n in zip(pools, proportional_split(args.n_frames, [p.size for p in pools.values()])):
        movie = path.name.replace(".predictions.slp", ".mp4")
        video = videos[movie]
        labels = Labels.load_file(str(path), video_search=[str(Path(video.filename).parent)])
        chosen = spaced_sample(pools[path], int(n), rng, set())
        by_frame = {lf.frame_idx: lf for lf in labels.labeled_frames}
        for frame_idx in sorted(chosen):
            instance = on_project_skeleton(best_instance(by_frame[frame_idx]), project.skeletons[0])
            project.append(LabeledFrame(video=video, frame_idx=frame_idx, instances=[instance]))
            project.suggestions.append(SuggestionFrame(video=video, frame_idx=frame_idx, group=SUGGESTION_GROUP))
        print(f"{movie}: {len(chosen)} suggestions added", flush=True)
    Labels.save_file(project, str(args.out))
    print(f"{len(project.labeled_frames)} frames, {len(project.suggestions)} suggestions -> {args.out}")


def charon_windows(index_csv: Path) -> dict[str, list[tuple[int, int]]]:
    """Frame windows per movie file name, from the video index CSV."""
    out: dict[str, list[tuple[int, int]]] = {}
    with open(index_csv, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            out.setdefault(row["filename"], []).append((int(row["start"]), int(row["end"])))
    return out


def doubtful_frames(
    labels: Labels, score_threshold: float, windows: list[tuple[int, int]], taken: set[int]
) -> np.ndarray:
    """Frames with a bird whose beak or tail scores below threshold, inside the windows, away from labelled frames."""
    names = labels.skeletons[0].node_names
    nodes = [names.index(n) for n in DOUBTFUL_NODES]
    frames = []
    for lf in labels.labeled_frames:
        if not lf.predicted_instances:
            continue
        if not any(lo <= lf.frame_idx <= hi for lo, hi in windows):
            continue
        if any(abs(lf.frame_idx - t) < MIN_FRAME_SPACING for t in taken):
            continue
        scores = best_instance(lf).scores[nodes]
        if np.nanmin(scores) < score_threshold:
            frames.append(lf.frame_idx)
    return np.asarray(frames, dtype=int)


def proportional_split(total: int, sizes: list[int]) -> list[int]:
    """Split ``total`` across pools in proportion to their size, never beyond a pool's size."""
    sizes_arr = np.asarray(sizes, dtype=float)
    if sizes_arr.sum() == 0:
        return [0] * len(sizes)
    share = np.floor(total * sizes_arr / sizes_arr.sum()).astype(int)
    for i in np.argsort(-(total * sizes_arr / sizes_arr.sum() - share)):
        if share.sum() >= total:
            break
        share[i] += 1
    return [int(min(s, n)) for s, n in zip(share, sizes)]


def on_project_skeleton(instance: PredictedInstance, skeleton) -> PredictedInstance:
    """Copy a predicted instance onto the project's skeleton, matching nodes by name."""
    if instance.skeleton.node_names != skeleton.node_names:
        raise ValueError(f"Skeleton nodes differ: {instance.skeleton.node_names} vs {skeleton.node_names}")
    return PredictedInstance.from_numpy(instance.numpy(), instance.scores, instance.score, skeleton=skeleton)


if __name__ == "__main__":
    main()
