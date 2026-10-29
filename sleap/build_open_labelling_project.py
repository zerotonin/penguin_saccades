# ╔══════════════════════════════════════════════════════════════════╗
# ║  penguin_saccades — sleap/build_open_labelling_project           ║
# ║  « only frames that hold a bird reach the labeller »             ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Builds a SLEAP project for the open-water GoPro movies: the     ║
# ║  five-node skeleton with fin symmetry, every movie as a video,   ║
# ║  and a stratified list of suggested frames drawn from the        ║
# ║  charon detections (.tra) so each suggestion contains at least   ║
# ║  one penguin. Label in the GUI with python -m sleap.gui.app      ║
# ║  <project>.                                                      ║
# ╚══════════════════════════════════════════════════════════════════╝
"""SLEAP labelling project for the open-water movies, seeded from charon detections."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from sleap.gui.suggestions import SuggestionFrame

from sleap import Labels, Skeleton, Video

NODES = ("beak", "neck", "left_fin", "right_fin", "caudal")
EDGES = (("beak", "neck"), ("neck", "left_fin"), ("neck", "right_fin"), ("neck", "caudal"))
DETECTION = re.compile(r">([^,<]+),([0-9.]+),([0-9.]+),([0-9.]+),([0-9.]+),([0-9.]+)<")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_dir", type=Path, help="folder with <movie>.mp4/.avi and <movie>.tra")
    parser.add_argument("out", type=Path, help="project to write, e.g. open_water_labels.slp")
    parser.add_argument("--n-frames", type=int, default=200)
    parser.add_argument("--min-score", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exclude", nargs="*", default=(), help="substrings of movie names to leave out")
    parser.add_argument("--boxes-csv", type=Path, default=None, help="curated boxes from export_open_boxes.py; raw .tra otherwise")
    args = parser.parse_args()
    labels = Labels(skeletons=[build_skeleton()])
    movies = [m for m in movie_files(args.video_dir) if not any(x in m[0].name for x in args.exclude)]
    rng = np.random.default_rng(args.seed)
    per_movie = allocate(len(movies), args.n_frames)
    curated = parse_boxes_csv(args.boxes_csv) if args.boxes_csv else {}
    for (video_path, tra_path), n in zip(movies, per_movie):
        detections = curated.get(tra_path.stem) if curated else parse_tra(tra_path, args.min_score)
        if not detections:
            print(f"{video_path.name}: no curated boxes, skipped")
            continue
        video = Video.from_filename(str(video_path))
        frames = pick_frames(detections, n, len(video), rng)
        labels.videos.append(video)
        labels.suggestions.extend(SuggestionFrame(video=video, frame_idx=int(f)) for f in frames)
        print(f"{video_path.name}: {len(detections)} frames with a bird, {len(frames)} suggested")
    Labels.save_file(labels, str(args.out))
    print(f"{len(labels.videos)} videos, {len(labels.suggestions)} suggestions -> {args.out}")


def build_skeleton() -> Skeleton:
    skeleton = Skeleton("penguin")
    for node in NODES:
        skeleton.add_node(node)
    for a, b in EDGES:
        skeleton.add_edge(a, b)
    skeleton.add_symmetry("left_fin", "right_fin")
    return skeleton


def allocate(n_movies: int, n_frames: int) -> list[int]:
    """Equal shares, remainder to the first movies."""
    base, extra = divmod(n_frames, n_movies)
    return [base + (i < extra) for i in range(n_movies)]


def movie_files(video_dir: Path) -> list[tuple[Path, Path]]:
    """(movie, .tra) pairs; an mp4 wins over an avi of the same name."""
    pairs = []
    for tra in sorted(video_dir.glob("*.tra")):
        candidates = [p for p in video_dir.iterdir() if p.suffix.lower() in (".mp4", ".avi") and p.stem.startswith(tra.stem)]
        if not candidates:
            continue
        candidates.sort(key=lambda p: (p.suffix.lower() != ".mp4", len(p.name)))
        pairs.append((candidates[0], tra))
    return pairs


def parse_tra(path: Path, min_score: float) -> dict[int, list[tuple[float, float, float, float]]]:
    """Frame index -> boxes (x0, y0, x1, y1, relative coordinates) above ``min_score``."""
    detections: dict[int, list[tuple[float, float, float, float]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        head, _, body = line.partition(":")
        if not head.strip().isdigit():
            continue
        boxes = [tuple(map(float, m.groups()[2:])) for m in DETECTION.finditer(body) if float(m.group(2)) >= min_score]
        if boxes:
            detections[int(head)] = boxes
    return detections


def parse_boxes_csv(path: Path) -> dict[str, dict[int, list[tuple[float, float, float, float]]]]:
    """Movie stem -> frame -> boxes, from export_open_boxes.py."""
    out: dict[str, dict[int, list]] = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            box = tuple(float(row[k]) for k in ("bounding_box_x0", "bounding_box_y0", "bounding_box_x1", "bounding_box_y1"))
            out.setdefault(row["movie"], {}).setdefault(int(row["frame"]), []).append(box)
    return out


def pick_frames(detections: dict[int, list], n: int, n_video_frames: int, rng: np.random.Generator) -> list[int]:
    """Spread ``n`` frames over the movie's time, four fifths of them from the larger (sharper) boxes."""
    frames = np.array([f for f in sorted(detections) if f < n_video_frames])
    if len(frames) <= n:
        return frames.tolist()
    area = np.array([max((b[2] - b[0]) * (b[3] - b[1]) for b in detections[f]) for f in frames])
    large = frames[area >= np.median(area)]          # nearer the surface, sharper
    small = frames[area < np.median(area)]
    n_small = min(len(small), n // 5)
    chosen = set(large[np.linspace(0, len(large) - 1, n - n_small).astype(int)].tolist())
    chosen.update(small[np.linspace(0, len(small) - 1, n_small).astype(int)].tolist())
    return sorted(chosen)


if __name__ == "__main__":
    main()
