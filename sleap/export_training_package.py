# ╔══════════════════════════════════════════════════════════════════╗
# ║  penguin_saccades — sleap/export_training_package                ║
# ║  « GUI projects in, one self-contained training package out »    ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Keeps only the frames that carry user labels, drops the         ║
# ║  suggestion list and embeds the frame images, so the package     ║
# ║  trains on the cluster without the movies. Several projects      ║
# ║  (gentoo and rockhopper) merge into one package on one           ║
# ║  skeleton; --accept-predictions turns predictions the labeller   ║
# ║  left untouched into labels. Run inside the SLEAP conda          ║
# ║  environment.                                                    ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Export labelled SLEAP projects as one embedded ``.pkg.slp`` training package."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sleap.instance import Instance, LabeledFrame, PredictedInstance

from sleap import Labels, Skeleton, load_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path, nargs="+", help="projects saved by the SLEAP GUI")
    parser.add_argument("out", type=Path, help="output .pkg.slp")
    parser.add_argument("--accept-predictions", action="store_true",
                        help="label frames that hold only predictions with their best prediction")
    args = parser.parse_args()
    projects = [load_file(str(path)) for path in args.labels]
    package = merged_labelled_subset(projects, args.accept_predictions)
    Labels.save_file(package, str(args.out), save_frame_data=True)
    n_full = sum(np.isfinite(inst.numpy()).all() for inst in package.user_instances)
    print(
        f"{len(package.labeled_frames)} labelled frames from {len(package.videos)} videos, "
        f"{len(package.user_instances)} instances ({n_full} complete) -> {args.out}"
    )


def merged_labelled_subset(projects: list[Labels], accept_predictions: bool = False) -> Labels:
    """Frames with user instances from every project, re-expressed on the first skeleton.

    Predictions are dropped. With ``accept_predictions`` a frame that holds
    predictions but no user instance keeps its best prediction as a label: the
    labeller checked it and left it as it was.
    """
    skeleton = projects[0].skeleton
    frames = []
    for project in projects:
        for frame in project.labeled_frames:
            instances = labelled_instances(frame, accept_predictions)
            if instances:
                frames.append(LabeledFrame(
                    video=frame.video, frame_idx=frame.frame_idx,
                    instances=[on_skeleton(inst, skeleton) for inst in instances]))
    if not frames:
        raise ValueError("The projects hold no user-labelled frames.")
    return Labels(labeled_frames=frames)


def labelled_instances(frame: LabeledFrame, accept_predictions: bool) -> list[Instance]:
    if frame.user_instances or not accept_predictions:
        return list(frame.user_instances)
    predictions = [inst for inst in frame.instances if isinstance(inst, PredictedInstance)]
    return [max(predictions, key=lambda inst: inst.score)] if predictions else []


def on_skeleton(instance: Instance, skeleton: Skeleton) -> Instance:
    """Copy an instance onto ``skeleton``, matching nodes by name."""
    if instance.skeleton.node_names != skeleton.node_names:
        raise ValueError(f"Skeleton nodes differ: {instance.skeleton.node_names} vs {skeleton.node_names}")
    return Instance.from_numpy(instance.numpy(), skeleton=skeleton)


if __name__ == "__main__":
    main()
