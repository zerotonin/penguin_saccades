# ╔══════════════════════════════════════════════════════════════════╗
# ║  penguin_saccades — sleap/track_videos                           ║
# ║  « models loaded once, every clip tracked and exported »         ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Replaces the sleap-track console script, which aborts on the    ║
# ║  1.4.1 packaging conflict, and python -m sleap.nn.inference,     ║
# ║  which has no main guard and does nothing. Simple tracker with   ║
# ║  centroid similarity and Hungarian matching, single-frame        ║
# ║  breaks reconnected. A track cap keeps the analysis export       ║
# ║  small on hour-long movies where a bird dives and resurfaces.    ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Track every movie of one data set with trained centroid and centred-instance models."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from sleap.nn.tracking import Tracker

import sleap
from sleap import Labels, Video


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_dir", type=Path, help="folder of movies")
    parser.add_argument("models_dir", type=Path, help="folder holding centroid/ and centered_instance/")
    parser.add_argument("out_dir", type=Path, help="destination for .predictions.slp and .analysis.h5")
    parser.add_argument("--peak-threshold", type=float, default=0.2)
    parser.add_argument("--track-window", type=int, default=5)
    parser.add_argument("--max-tracks", type=int, default=None, help="reuse at most this many tracks")
    parser.add_argument("--pattern", default="[0-9]*.mp4", help="glob for the movies to track")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    predictor = build_predictor(args.models_dir, args.peak_threshold, args.track_window, args.max_tracks)
    for video_path in movie_paths(args.video_dir, args.pattern):
        t0 = time.time()
        labels = predictor.predict(Video.from_filename(str(video_path)))
        stem = args.out_dir / video_path.stem
        Labels.save_file(labels, f"{stem}.predictions.slp")
        labels.export(f"{stem}.analysis.h5")
        n_tracks = len(labels.tracks)
        n_inst = sum(len(lf.predicted_instances) for lf in labels.labeled_frames)
        print(f"{video_path.name}: {n_inst} instances in {n_tracks} tracks, {time.time() - t0:.0f} s", flush=True)


def build_predictor(models_dir: Path, peak_threshold: float, track_window: int, max_tracks: int | None):
    predictor = sleap.load_model(
        [str(models_dir / "centroid"), str(models_dir / "centered_instance")],
        peak_threshold=peak_threshold,
        progress_reporting="none",
    )
    predictor.tracker = Tracker.make_tracker_by_name(
        tracker="simple",
        similarity="centroid",
        match="hungarian",
        track_window=track_window,
        post_connect_single_breaks=True,
        max_tracking=max_tracks is not None,
        max_tracks=max_tracks,
    )
    return predictor


def movie_paths(video_dir: Path, pattern: str) -> list[Path]:
    """Movies matching the glob, without DeepLabCut-labelled copies."""
    return sorted(p for p in video_dir.glob(pattern) if "DLC" not in p.name)


if __name__ == "__main__":
    main()
