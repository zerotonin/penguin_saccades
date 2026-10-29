# ╔══════════════════════════════════════════════════════════════════╗
# ║  penguin_saccades — sleap/export_training_package                ║
# ║  « GUI project in, self-contained training package out »         ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Keeps only the frames that carry user labels, drops the         ║
# ║  suggestion list and embeds the frame images, so the package     ║
# ║  trains on the cluster without the movies. Run inside the        ║
# ║  SLEAP conda environment.                                        ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Export a labelled SLEAP project as an embedded ``.pkg.slp`` training package."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sleap import Labels, load_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path, help="project saved by the SLEAP GUI")
    parser.add_argument("out", type=Path, help="output .pkg.slp")
    args = parser.parse_args()
    package = labelled_subset(load_file(str(args.labels)))
    Labels.save_file(package, str(args.out), save_frame_data=True)
    n_full = sum(np.isfinite(inst.numpy()).all() for inst in package.user_instances)
    print(
        f"{len(package.labeled_frames)} labelled frames from {len(package.videos)} videos, "
        f"{len(package.user_instances)} instances ({n_full} complete) -> {args.out}"
    )


def labelled_subset(labels: Labels) -> Labels:
    """Frames with at least one user instance; suggestions and predictions dropped."""
    frames = [frame for frame in labels.user_labeled_frames]
    if not frames:
        raise ValueError("The project holds no user-labelled frames.")
    return Labels(labeled_frames=frames)


if __name__ == "__main__":
    main()
