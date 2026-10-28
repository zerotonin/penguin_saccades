# ╔══════════════════════════════════════════════════════════════════╗
# ║  penguin_saccades — sleap/collect_predictions                    ║
# ║  « nineteen prediction files in, one proofreading project out »  ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Loads every <rec>.predictions.slp written on the cluster,       ║
# ║  points its video at the local copy of <rec>.mp4, merges the     ║
# ║  user labels of the training package, and saves one project for  ║
# ║  the SLEAP GUI. Open it with python -m sleap.gui.app <file>;     ║
# ║  the console script aborts on the 1.4.1 packaging conflict.      ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Merge cluster predictions into one local SLEAP project for proofreading."""

from __future__ import annotations

import argparse
from pathlib import Path

from sleap import Labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions_dir", type=Path, help="folder of <rec>.predictions.slp")
    parser.add_argument("video_dir", type=Path, help="local folder holding <rec>.mp4")
    parser.add_argument("out", type=Path, help="merged project, e.g. penguins_proofread.slp")
    parser.add_argument("--labels", type=Path, default=None, help="training package to merge user labels from")
    args = parser.parse_args()
    merged = collect(args.predictions_dir, args.video_dir, args.labels)
    Labels.save_file(merged, str(args.out))
    n_pred = sum(len(lf.predicted_instances) for lf in merged.labeled_frames)
    n_user = sum(len(lf.user_instances) for lf in merged.labeled_frames)
    print(f"{len(merged.videos)} videos, {len(merged.labeled_frames)} frames, "
          f"{n_pred} predicted and {n_user} user instances -> {args.out}")


def collect(predictions_dir: Path, video_dir: Path, labels_path: Path | None) -> Labels:
    merged = Labels()
    for path in sorted(predictions_dir.glob("*.predictions.slp")):
        labels = Labels.load_file(str(path), video_search=[str(video_dir)])
        merged.extend_from(labels, unify=True)
    if labels_path is not None:
        user = Labels.load_file(str(labels_path), video_search=[str(video_dir)])
        merged.extend_from(user, unify=True)
    return merged


if __name__ == "__main__":
    main()
