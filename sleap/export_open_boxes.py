# ╔══════════════════════════════════════════════════════════════════╗
# ║  penguin_saccades — sleap/export_open_boxes                      ║
# ║  « curated charon boxes to a CSV the SLEAP env can read »        ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  The sorted_and_filtered trajectories are pandas HDF files,      ║
# ║  which need PyTables; the SLEAP environment has none. This       ║
# ║  writes one CSV (movie, frame, relative box, quality) that       ║
# ║  build_open_labelling_project.py consumes. Run it in the         ║
# ║  analysis environment.                                           ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Export the curated open-water detections to CSV for the labelling project builder."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

BOX = ("bounding_box_x0", "bounding_box_y0", "bounding_box_x1", "bounding_box_y1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectories_dir", type=Path, help="folder of sorted_and_filtered <movie>.h5")
    parser.add_argument("out_csv", type=Path)
    args = parser.parse_args()
    parts = []
    for h5 in sorted(args.trajectories_dir.glob("*.h5")):
        df = pd.read_hdf(h5, key="trace")
        part = df[[*BOX, "quality"]].copy()
        part.insert(0, "frame", df.index.to_numpy())
        part.insert(0, "movie", h5.stem)
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(args.out_csv, index=False)
    print(f"{len(out)} boxes from {len(parts)} movies -> {args.out_csv}")


if __name__ == "__main__":
    main()
