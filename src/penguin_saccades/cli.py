# ╔══════════════════════════════════════════════════════════════════╗
# ║  penguin_saccades — cli                                          ║
# ║  « five re-runnable stages with CSV hand-offs »                  ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Each stage reads the CSV its predecessor wrote in output_dir    ║
# ║  and writes its own, so any stage can be rerun alone after a     ║
# ║  constant changes. Progress bars live here, never in the         ║
# ║  library modules.                                                ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Command-line pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from penguin_saccades import config
from penguin_saccades import constants as C
from penguin_saccades.ingest_closed import process_closed_trajectory, read_dlc, recording_id_from_filename
from penguin_saccades.ingest_open import load_open_trajectory, process_open_trajectory, species_from_filename
from penguin_saccades.isi import chunk_intervals, head_body_compensation, interval_content, saccade_budget
from penguin_saccades.saccades import detect_in_trajectories, triggered_windows, zero_offset_rows
from penguin_saccades.stats import pairwise_fisher, summarise
from penguin_saccades.taxonomy import associate_head_body, classified_intervals

FRAME_RATES: dict[str, float] = {
    C.DATASET_OPEN.name: C.DATASET_OPEN.frame_rate_hz,
    C.DATASET_CLOSED.name: C.DATASET_CLOSED.frame_rate_hz,
}
FILES = {
    "trajectories": "trajectories.csv",
    "saccades": "saccades.csv",
    "intervals": "intervals.csv",
    "saccade_content": "saccade_content.csv",
    "interval_content": "interval_content.csv",
    "budget": "saccade_budget.csv",
    "compensation": "head_body_compensation.csv",
    "chunks": "isi_chunks.csv",
}


# ┌────────────────────────────────────────────────────────────┐
# │ Stages  « one function per CLI subcommand »                │
# └────────────────────────────────────────────────────────────┘
def stage_ingest(out: Path) -> None:
    """Both data sets to one trajectory table."""
    parts: list[pd.DataFrame] = []
    trajectory_id = 0
    open_dir = config.get_path("open_trajectories_dir")
    for path in tqdm(sorted(open_dir.rglob("*.h5")), desc="data set 1 (open)"):
        df = process_open_trajectory(load_open_trajectory(path))
        if df.empty:
            continue
        df[C.COL_SPECIES] = species_from_filename(path)
        df[C.COL_TRAJECTORY] = trajectory_id
        df["source_file"] = path.name
        trajectory_id += 1
        parts.append(df)
    closed_dir = config.get_path("closed_dlc_dir")
    dlc_files = sorted(closed_dir.glob("*DLC*.csv")) + sorted(closed_dir.glob("*DLC*.h5"))
    for path in tqdm(dlc_files, desc="data set 2 (closed)"):
        df = process_closed_trajectory(read_dlc(path))
        if df.empty:
            continue
        df[C.COL_TRAJECTORY] = trajectory_id
        df["source_file"] = path.name
        df["recording_id"] = recording_id_from_filename(path)
        trajectory_id += 1
        parts.append(df)
    if not parts:
        raise FileNotFoundError(f"No trajectories found under {open_dir} or {closed_dir}.")
    trajectories = pd.concat(parts, ignore_index=True)
    trajectories.to_csv(out / FILES["trajectories"], index=False)
    print(f"[ingest] {len(trajectories)} frames, {trajectory_id} trajectories -> {out / FILES['trajectories']}")


def stage_saccades(out: Path) -> None:
    """Detect body and head saccades, associate them, and extract intervals."""
    trajectories = _read(out, "trajectories")
    body, _ = detect_in_trajectories(trajectories, C.COL_BODY_YAW, "body", FRAME_RATES)
    with_head = trajectories[trajectories[C.COL_HEAD_YAW].notna()]
    head, _ = detect_in_trajectories(with_head, C.COL_HEAD_YAW, "head", FRAME_RATES)
    head, body = associate_head_body(head, body)
    saccades = pd.concat([body, head], ignore_index=True)
    saccades.to_csv(out / FILES["saccades"], index=False)
    intervals = classified_intervals(trajectories, head, body, FRAME_RATES)
    intervals.to_csv(out / FILES["intervals"], index=False)
    print(f"[saccades] {len(body)} body, {len(head)} head saccades; {len(intervals)} intervals")


def stage_isi(out: Path) -> None:
    """Measure what happens inside saccades and inside intervals."""
    trajectories = _read(out, "trajectories")
    saccades = _read(out, "saccades")
    intervals = _read(out, "intervals")
    interval_content(trajectories, saccades).to_csv(out / FILES["saccade_content"], index=False)
    interval_content(trajectories, intervals).to_csv(out / FILES["interval_content"], index=False)
    body_budget = saccade_budget(trajectories, saccades[saccades[C.COL_SOURCE] == "body"], C.COL_BODY_YAW)
    body_budget[C.COL_SOURCE] = "body"
    head_budget = saccade_budget(trajectories, saccades[saccades[C.COL_SOURCE] == "head"], C.COL_HEAD_YAW)
    head_budget[C.COL_SOURCE] = "head"
    pd.concat([body_budget, head_budget], ignore_index=True).to_csv(out / FILES["budget"], index=False)
    any_intervals = intervals[intervals[C.COL_INTERVAL_CLASS] == "any"]
    head_body_compensation(trajectories, any_intervals).to_csv(out / FILES["compensation"], index=False)
    chunk_intervals(trajectories, any_intervals, FRAME_RATES).to_csv(out / FILES["chunks"], index=False)
    print("[isi] saccade_content, interval_content, saccade_budget, head_body_compensation, isi_chunks written")


def stage_stats(out: Path) -> None:
    """Summary tables with bootstrap CIs and pairwise Fisher tests."""
    saccades = _read(out, "saccade_content")
    intervals = _read(out, "interval_content")
    budget = _read(out, "budget")
    saccades["duration_ms"] = saccades["duration_s"] * 1000.0
    saccades["abs_amplitude_deg"] = saccades["amplitude_deg"].abs()
    saccades["abs_peak_speed_degps"] = saccades["peak_speed_degps"].abs()
    intervals["duration_ms"] = intervals["duration_s"] * 1000.0
    sacc_keys = [C.COL_DATASET, C.COL_SPECIES, C.COL_SOURCE, C.COL_SACCADE_TYPE]
    int_keys = [C.COL_DATASET, C.COL_SPECIES, C.COL_INTERVAL_CLASS]
    summaries = [summarise(saccades, v, sacc_keys) for v in
                 ("duration_ms", "abs_amplitude_deg", "abs_peak_speed_degps", "median_translation_mps")]
    summaries += [summarise(intervals, v, int_keys) for v in
                  ("duration_ms", "net_yaw_deg", "path_yaw_deg", "median_abs_body_speed_degps",
                   "median_translation_mps")]
    summaries += [summarise(budget, v, [C.COL_DATASET, C.COL_SPECIES, C.COL_SOURCE]) for v in
                  ("time_fraction_in_saccades", "yaw_fraction_in_saccades")]
    pd.concat(summaries, ignore_index=True).to_csv(out / "summary_tables.csv", index=False)

    saccades["group"] = saccades[sacc_keys].astype(str).agg(" | ".join, axis=1)
    tests = [pairwise_fisher(saccades, v, "group") for v in
             tqdm(("duration_ms", "abs_amplitude_deg", "abs_peak_speed_degps"), desc="saccade tests")]
    # Translation inside saccades versus inside intervals, per data set and species.
    sacc_t = saccades[[C.COL_DATASET, C.COL_SPECIES, "median_translation_mps"]].assign(event="saccade")
    int_t = intervals.loc[intervals[C.COL_INTERVAL_CLASS] == "any",
                          [C.COL_DATASET, C.COL_SPECIES, "median_translation_mps"]].assign(event="interval")
    both = pd.concat([sacc_t, int_t], ignore_index=True)
    both["group"] = both[[C.COL_DATASET, C.COL_SPECIES, "event"]].astype(str).agg(" | ".join, axis=1)
    tests.append(pairwise_fisher(both, "median_translation_mps", "group"))
    pd.concat([t for t in tests if not t.empty], ignore_index=True).to_csv(out / "pairwise_tests.csv", index=False)
    print(f"[stats] summary_tables.csv and pairwise_tests.csv written to {out}")


def stage_plots(out: Path) -> None:
    """Standard figure set."""
    import matplotlib
    matplotlib.use("Agg")
    from penguin_saccades import viz

    viz.apply_style()
    figdir = out / "figures"
    trajectories = _read(out, "trajectories")
    saccades = _read(out, "saccade_content")
    intervals = _read(out, "interval_content")
    saccades["duration_ms"] = saccades["duration_s"] * 1000.0
    saccades["abs_peak_speed_degps"] = saccades["peak_speed_degps"].abs()
    saccades["label"] = saccades[C.COL_SPECIES] + " " + saccades[C.COL_SACCADE_TYPE]
    colours = {lab: C.SACCADE_TYPE_COLOURS.get(lab.split(" ")[-1], "k") for lab in saccades["label"].unique()}
    for value, ylabel, log_y in (("duration_ms", "saccade duration (ms)", True),
                                 ("abs_peak_speed_degps", r"peak yaw velocity ($^\circ\,s^{-1}$)", True)):
        fig, ax = matplotlib.pyplot.subplots(figsize=C.FIGURE_SIZE_DOUBLE)
        viz.plot_raincloud(saccades, value, "label", colours, ylabel, log_y=log_y, ax=ax)
        viz.save_figure(fig, f"raincloud_{value}", figdir, saccades[["label", value]])
        matplotlib.pyplot.close(fig)
    # Translation: saccades versus intervals.
    both = pd.concat([
        saccades[[C.COL_SPECIES, "median_translation_mps"]].assign(event="saccade"),
        intervals.loc[intervals[C.COL_INTERVAL_CLASS] == "any", [C.COL_SPECIES, "median_translation_mps"]].assign(event="interval"),
    ], ignore_index=True)
    both["label"] = both[C.COL_SPECIES] + " " + both["event"]
    fig, ax = matplotlib.pyplot.subplots(figsize=C.FIGURE_SIZE_DOUBLE)
    viz.plot_raincloud(both, "median_translation_mps", "label",
                       {lab: C.SPECIES_COLOURS.get(lab.split(" ")[0], "k") for lab in both["label"].unique()},
                       r"translational speed (m$\,s^{-1}$)", ax=ax)
    viz.save_figure(fig, "raincloud_translation_saccade_vs_interval", figdir, both)
    matplotlib.pyplot.close(fig)
    # Saccade-triggered averages per data set and source.
    for (dataset, source), sacc in saccades.groupby([C.COL_DATASET, C.COL_SOURCE]):
        fps = FRAME_RATES[dataset]
        half = int(round(C.TRIGGER_WINDOW_S * fps / 2))
        time_s = np.arange(-half, half + 1) / fps
        tables_v, tables_a = [], []
        for part, yaw_col, vel_col in (("body", C.COL_BODY_YAW, C.COL_BODY_VEL), ("head", C.COL_HEAD_YAW, C.COL_HEAD_VEL)):
            mats_v, mats_a = [], []
            for key, seg in trajectories[trajectories[C.COL_DATASET] == dataset].groupby(
                    [C.COL_TRAJECTORY, C.COL_SEGMENT]):
                if not seg[yaw_col].notna().any():
                    continue
                s = sacc[(sacc[C.COL_TRAJECTORY] == key[0]) & (sacc[C.COL_SEGMENT] == key[1])]
                if s.empty:
                    continue
                frames = seg[C.COL_FRAME].to_numpy()
                peaks = np.searchsorted(frames, s["peak_idx"].to_numpy(dtype=int))
                flip = (s["direction"] == C.DIRECTION_OF_NEGATIVE_VELOCITY).to_numpy()
                mats_v.append(triggered_windows(seg[vel_col].to_numpy(), peaks, half, flip))
                mats_a.append(zero_offset_rows(triggered_windows(np.rad2deg(seg[yaw_col].to_numpy()), peaks, half, flip)))
            if mats_v:
                tables_v.append(viz.triggered_average_table(np.vstack(mats_v), time_s, part))
                tables_a.append(viz.triggered_average_table(np.vstack(mats_a), time_s, part))
        if not tables_v:
            continue
        fig, axes = matplotlib.pyplot.subplots(1, 2, figsize=C.FIGURE_SIZE_DOUBLE)
        viz.plot_triggered_average(tables_v, C.BODYPART_COLOURS, r"yaw velocity ($^\circ\,s^{-1}$)", ax=axes[0])
        viz.plot_triggered_average(tables_a, C.BODYPART_COLOURS, r"yaw ($^\circ$)", ax=axes[1])
        fig.suptitle(f"{dataset}: {source}-saccade triggered")
        viz.save_figure(fig, f"triggered_{dataset}_{source}", figdir,
                        pd.concat(tables_v + tables_a, ignore_index=True))
        matplotlib.pyplot.close(fig)
    print(f"[plots] figures written to {figdir}")


# ┌────────────────────────────────────────────────────────────┐
# │ Entry point  « argparse »                                  │
# └────────────────────────────────────────────────────────────┘
STAGES = {"ingest": stage_ingest, "saccades": stage_saccades, "isi": stage_isi,
          "stats": stage_stats, "plots": stage_plots}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="penguin-saccades", description=__doc__)
    parser.add_argument("stage", choices=[*STAGES, "all"])
    parser.add_argument("--output-dir", type=Path, default=None, help="defaults to the profile output_dir")
    parser.add_argument("--profile", default=None, help="local_paths.json profile to use")
    args = parser.parse_args(argv)
    if args.profile:
        import os
        os.environ[config.ENV_PROFILE] = args.profile
    try:
        out = args.output_dir or config.output_dir()
        out.mkdir(parents=True, exist_ok=True)
        for name in (list(STAGES) if args.stage == "all" else [args.stage]):
            STAGES[name](out)
    except config.PathConfigError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    return 0


def _read(out: Path, key: str) -> pd.DataFrame:
    path = out / FILES[key]
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run the earlier stage first.")
    return pd.read_csv(path)


if __name__ == "__main__":
    raise SystemExit(main())
