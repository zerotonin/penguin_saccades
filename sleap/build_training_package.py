# ╔══════════════════════════════════════════════════════════════════╗
# ║  penguin_saccades — sleap/build_training_package                 ║
# ║  « DeepLabCut labels in, one self-contained SLEAP package out »  ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Converts the CollectedData CSVs of the 2023 DeepLabCut project  ║
# ║  into a SLEAP training package attached to the real videos.      ║
# ║  Labelled PNG imgNNN.png is frame NNN of <rec>.mp4 (verified     ║
# ║  pixel-wise), so predictions can later be merged with the        ║
# ║  labels for proofreading. Run inside the SLEAP conda             ║
# ║  environment.                                                    ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Build ``penguins_train.pkg.slp`` from the DeepLabCut labelled frames."""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sleap.instance import Point

from sleap import Instance, LabeledFrame, Labels, Skeleton, Video

NODES: tuple[str, ...] = ("beak", "neck", "left_fin", "right_fin", "caudal")
EDGES: tuple[tuple[str, str], ...] = (
    ("beak", "neck"), ("neck", "left_fin"), ("neck", "right_fin"), ("neck", "caudal"),
)
MIN_POINTS: int = 3
SCORER: str = "Bart"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dlc_project", type=Path, help="DeepLabCut project folder (holds labeled-data/)")
    parser.add_argument("video_dir", type=Path, help="folder with <rec>.mp4")
    parser.add_argument("out", type=Path, help="output .pkg.slp")
    parser.add_argument("--skip", nargs="*", default=(), help="rec:frame pairs known to mismatch the video")
    args = parser.parse_args()
    skip = {tuple(s.split(":")) for s in args.skip}
    labels = build_labels(args.dlc_project, args.video_dir, skip)
    Labels.save_file(labels, str(args.out), save_frame_data=True)
    print(f"{len(labels.labeled_frames)} labelled frames, {len(labels.videos)} videos -> {args.out}")


def build_labels(dlc_project: Path, video_dir: Path, skip: set[tuple[str, str]]) -> Labels:
    skeleton = build_skeleton()
    frames: list[LabeledFrame] = []
    for csv in sorted(glob.glob(str(dlc_project / "labeled-data" / "[0-9]*" / f"CollectedData_{SCORER}.csv"))):
        rec = Path(csv).parent.name
        video_path = video_dir / f"{rec}.mp4"
        if rec.endswith("_labeled") or not video_path.exists():
            continue
        video = Video.from_filename(str(video_path))
        table = pd.read_csv(csv, header=[0, 1, 2], index_col=0)
        for image, row in table.iterrows():
            frame = int(re.search(r"img(\d+)", str(image)).group(1))
            if (rec, str(frame)) in skip:
                continue
            points = labelled_points(row, skeleton)
            if "neck" not in points or len(points) < MIN_POINTS:
                continue
            try:
                video.get_frame(frame)
            except KeyError:
                print(f"skip {rec} frame {frame}: not decodable from the video")
                continue
            frames.append(LabeledFrame(video=video, frame_idx=frame,
                                       instances=[Instance(skeleton=skeleton, points=points)]))
    return Labels(labeled_frames=frames)


def build_skeleton() -> Skeleton:
    skeleton = Skeleton("penguin")
    for node in NODES:
        skeleton.add_node(node)
    for a, b in EDGES:
        skeleton.add_edge(a, b)
    return skeleton


def labelled_points(row: pd.Series, skeleton: Skeleton) -> dict[str, Point]:
    points: dict[str, Point] = {}
    for node in NODES:
        x, y = row[(SCORER, node, "x")], row[(SCORER, node, "y")]
        if np.isfinite(x) and np.isfinite(y):
            points[node] = Point(x, y)
    return points


if __name__ == "__main__":
    main()
