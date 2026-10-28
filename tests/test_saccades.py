import numpy as np
import pandas as pd

from penguin_saccades import constants as C
from penguin_saccades import saccades as S
from tests.conftest import piecewise_heading

FPS = 30.0


def _yaw(saccades, duration_s=10.0):
    t = np.arange(int(duration_s * FPS)) / FPS
    return piecewise_heading(t, saccades)


def test_single_saccade_detected_with_correct_amplitude_and_sign():
    yaw = _yaw([(5.0, 60.0)])
    df = S.filter_artefacts(S.detect_saccades(yaw, FPS))
    assert len(df) == 1
    row = df.iloc[0]
    assert abs(row.amplitude_deg - 60) < 3
    assert row.direction == C.DIRECTION_OF_POSITIVE_VELOCITY and row.peak_speed_degps > 200
    assert 4.9 < row.peak_time_s < 5.4
    assert 0.15 < row.duration_s < 0.5


def test_left_turn_has_negative_amplitude():
    df = S.filter_artefacts(S.detect_saccades(_yaw([(3.0, -45.0)]), FPS))
    assert len(df) == 1 and df.amplitude_deg.iloc[0] < -40
    assert df.direction.iloc[0] == C.DIRECTION_OF_NEGATIVE_VELOCITY


def test_amplitude_is_not_converted_twice():
    yaw = _yaw([(5.0, 40.0)])
    df = S.detect_saccades(yaw, FPS)
    assert df.amplitude_deg.abs().max() < 60


def test_small_blip_is_rejected_by_artefact_filter():
    yaw = np.zeros(300)
    yaw[150] = np.deg2rad(20.0)  # one-frame 20 deg jitter: fast, no net turn
    df = S.detect_saccades(yaw, FPS)
    assert len(df) >= 1
    assert S.filter_artefacts(df).empty


def test_saccade_touching_nan_is_dropped():
    yaw = _yaw([(5.0, 60.0)])
    yaw[152:154] = np.nan
    assert S.detect_saccades(yaw, FPS).empty


def test_intervals_per_segment_bracket_saccades_and_flag_censoring():
    yaw = _yaw([(3.0, 60.0), (6.0, -60.0)])
    sacc = S.filter_artefacts(S.detect_saccades(yaw, FPS))
    ints = S.intersaccadic_intervals(sacc, len(yaw), FPS)
    assert len(ints) == 3
    assert ints.censored.tolist() == [True, False, True]
    middle = ints.iloc[1]
    assert middle.start_idx == sacc.stop_idx.iloc[0] + 1 and middle.stop_idx == sacc.start_idx.iloc[1] - 1
    assert abs(middle.duration_s - (middle.stop_idx - middle.start_idx + 1) / FPS) < 1e-9


def test_no_saccade_gives_one_censored_interval():
    ints = S.intersaccadic_intervals(S.detect_saccades(np.zeros(100), FPS), 100, FPS)
    assert len(ints) == 1 and ints.censored.iloc[0] and ints.duration_s.iloc[0] == 100 / FPS


def test_detect_in_trajectories_uses_global_frames_and_never_crosses_segments():
    yaw_a = _yaw([(2.0, 60.0)], 5.0)
    yaw_b = _yaw([(2.0, 60.0)], 5.0)
    traj = pd.concat([
        pd.DataFrame({C.COL_FRAME: np.arange(150), C.COL_BODY_YAW: yaw_a, C.COL_SEGMENT: 0}),
        pd.DataFrame({C.COL_FRAME: np.arange(1000, 1150), C.COL_BODY_YAW: yaw_b, C.COL_SEGMENT: 1}),
    ], ignore_index=True)
    traj[C.COL_DATASET] = "open"
    traj[C.COL_SPECIES] = "gentoo"
    traj[C.COL_TRAJECTORY] = 0
    sacc, ints = S.detect_in_trajectories(traj, C.COL_BODY_YAW, "body", {"open": FPS})
    assert len(sacc) == 2 and sacc.peak_idx.iloc[1] > 1000
    assert ints.duration_s.max() < 5.0  # no interval spans the 850-frame gap


def test_triggered_windows_pad_and_flip():
    sig = np.arange(10, dtype=float)
    m = S.triggered_windows(sig, np.array([1, 8]), 2, flip=np.array([False, True]))
    assert np.isnan(m[0, 0]) and m[0, 1] == 0 and m[0, 3] == 2
    assert m[1, 2] == -8 and np.isnan(m[1, 4])
    z = S.zero_offset_rows(np.array([[np.nan, 2.0, 4.0, 6.0]]), n_reference=2)
    assert np.allclose(z[0, 1:], [-1, 1, 3])
