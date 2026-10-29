from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from penguin_saccades import constants as C
from penguin_saccades import ingest_closed as IC
from penguin_saccades import ingest_open as IO
from tests.conftest import dlc_track, open_track, write_dlc_csv

# Objects are ``name, quality, y0, x0, y1, x1`` in relative image coordinates.
# Land is the top-right block of the frame (x > 660 px and y < 272 px).
TRA = (
    "0 : >king,0.9,0.05,0.70,0.15,0.80< >king,0.8,0.50,0.70,0.60,0.80<\n"   # on land, and in water
    "1 : >king,0.9,0.50,0.75,0.60,0.85< >king,0.95,0.52,0.75,0.62,0.85<\n"  # two in water: pick quality .95
    "2 :\n"
)


def test_read_charon_tra_keeps_best_in_roi_detection(tmp_path: Path):
    path = tmp_path / "Rockhopper_test.tra"
    path.write_text(TRA, encoding="utf-8")
    df = IO.read_charon_tra(path)
    assert list(df.index) == [0, 1]
    assert df.loc[0, "center_of_mass_y"] == pytest.approx(0.55)   # the water one, not the land one
    assert df.loc[1, "quality"] == 0.95
    assert IO.species_from_filename(path) == "rockhopper"


def test_process_open_trajectory_recovers_turn_and_speed():
    traj = IO.process_open_trajectory(open_track(saccades=[(5.0, 60.0)], drop_frames=(100, 101, 102)))
    assert set(C.TRAJECTORY_COLUMNS) - {C.COL_SPECIES, C.COL_TRAJECTORY} <= set(traj.columns)
    assert traj[C.COL_SEGMENT].nunique() == 1          # a 3-frame gap is bridged
    yaw = np.rad2deg(traj[C.COL_BODY_YAW].to_numpy())
    assert abs(yaw[30]) < 1 and abs(yaw[-30] - 60) < 1
    assert abs(np.nanmedian(traj[C.COL_TRANS]) - 2.0) < 0.05
    assert traj[C.COL_BODY_VEL].abs().max() > 200


def test_open_long_gap_splits_segments():
    traj = IO.process_open_trajectory(open_track(drop_frames=tuple(range(140, 170))))
    assert traj[C.COL_SEGMENT].nunique() == 2


def test_read_dlc_csv_and_gate(tmp_path: Path):
    df = dlc_track(low_likelihood_frames=(10, 11, 50))
    path = tmp_path / "12_penguinDLC_resnet50.csv"
    write_dlc_csv(df, path)
    read = IC.read_dlc(path)
    assert "beak_likelihood" in read.columns and len(read) == len(df)
    gated = IC.gate_likelihood(read)
    assert np.isfinite(gated.loc[10, "beak_x"])          # short gap interpolated
    assert IC.recording_id_from_filename(path) == "12"


def test_process_closed_recovers_head_offset_and_scale():
    traj = IC.process_closed_trajectory(dlc_track(body_saccades=[], head_saccades=[(2.0, 40.0)]))
    assert traj[C.COL_DATASET].iloc[0] == "closed" and traj[C.COL_SPECIES].iloc[0] == "gentoo"
    rel = np.rad2deg(traj[C.COL_HEAD_YAW] - traj[C.COL_BODY_YAW]).to_numpy()
    assert abs(rel[20]) < 1 and abs(rel[-20] - 40) < 1
    assert abs(np.nanmedian(traj[C.COL_TRANS]) - 1.0) < 0.05


def test_read_sleap_analysis_and_focal_track(tmp_path: Path):
    import h5py

    df = dlc_track(duration_s=4.0)
    n = len(df)
    nodes = list(C.DLC_KEYPOINTS)
    tracks = np.full((2, 2, len(nodes), n + 20), np.nan)
    scores = np.full((2, len(nodes), n + 20), np.nan)
    for j, kp in enumerate(nodes):
        tracks[0, 0, j, :n] = df[f"{kp}_x"]
        tracks[0, 1, j, :n] = df[f"{kp}_y"]
        scores[0, j, :n] = 0.9
        tracks[1, :, j, 5:9] = 100.0           # a second bird seen for four frames
        scores[1, j, 5:9] = 0.5
    path = tmp_path / "07.analysis.h5"
    with h5py.File(path, "w") as f:
        f["tracks"] = tracks
        f["point_scores"] = scores
        f["node_names"] = np.array([s.encode() for s in nodes])
    table = IC.read_sleap_analysis(path, min_occupancy=1)
    assert set(table.track_id) == {0, 1}
    focal = IC.focal_track(table)
    assert len(focal) == n and focal.beak_likelihood.iloc[0] == 0.9
    traj = IC.process_closed_trajectory(focal)
    assert traj[C.COL_HEAD_YAW].notna().sum() > 0.8 * n


def test_best_instance_per_frame_keeps_highest_score_any_track():
    base = dlc_track(duration_s=2.0).reset_index()
    first = base.assign(track_id=0).drop(index=[7])
    second = base[(base[C.COL_FRAME] >= 10) & (base[C.COL_FRAME] < 20)].assign(track_id=1)
    second = second.assign(**{f"{kp}_likelihood": 0.999 for kp in C.DLC_KEYPOINTS})
    second = second.assign(neck_x=second["neck_x"] + 1000.0)
    best = IC.best_instance_per_frame(pd.concat([first, second], ignore_index=True))
    assert len(best) == len(base) and "track_id" not in best.columns
    assert best.loc[15, "neck_x"] == pytest.approx(base.loc[15, "neck_x"] + 1000.0)
    assert best.loc[5, "neck_x"] == pytest.approx(base.loc[5, "neck_x"])
    assert best.loc[7, "neck_likelihood"] == 0.0 and np.isnan(best.loc[7, "neck_x"])


def test_process_open_pose_trajectory_scales_movie_pixels():
    px_per_m = C.OPEN_MOVIE_SIZE_PX[0] / (C.DATASET_OPEN.image_width_px * C.OPEN_M_PER_PX)
    df = dlc_track(duration_s=4.0, body_saccades=[(2.0, 60.0)], head_saccades=[(1.0, 30.0)],
                   speed_mps=1.0, fps=C.DATASET_OPEN.frame_rate_hz, px_per_m=px_per_m)
    traj = IO.process_open_pose_trajectory(df)
    assert traj[C.COL_DATASET].iloc[0] == "open" and traj[C.COL_SEGMENT].nunique() == 1
    assert abs(np.nanmedian(traj[C.COL_TRANS]) - 1.0) < 0.05
    yaw = np.rad2deg(traj[C.COL_BODY_YAW].to_numpy())
    assert abs(yaw[30]) < 1 and abs(yaw[-30] - 60) < 1
    rel = np.rad2deg(traj[C.COL_HEAD_YAW] - traj[C.COL_BODY_YAW]).to_numpy()
    assert abs(rel[20]) < 1 and abs(rel[-20] - 30) < 1


def test_open_pose_on_land_is_dropped():
    df = dlc_track(duration_s=2.0, speed_mps=0.0, fps=C.DATASET_OPEN.frame_rate_hz)
    for col in df.columns:
        if col.endswith("_x"):
            df[col] += 1500.0           # into the top-right land block of the frame
        elif col.endswith("_y"):
            df[col] -= 500.0
    assert IO.process_open_pose_trajectory(df).empty
