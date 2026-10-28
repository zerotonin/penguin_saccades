import numpy as np
import pandas as pd

from penguin_saccades import kinematics as K
from penguin_saccades.constants import COL_FRAME


def test_split_segments_breaks_on_gaps_and_drops_short():
    frames = np.array([0, 1, 2, 3, 20, 21, 22, 23, 24, 25, 60])
    segs = K.split_segments(frames, max_gap_frames=2, min_length_frames=4)
    assert [list(s) for s in segs] == [[0, 1, 2, 3], [20, 21, 22, 23, 24, 25]]


def test_fill_gaps_interpolates_numeric_and_ffills_text():
    df = pd.DataFrame({"v": [0.0, 2.0], "name": ["a", "a"]}, index=pd.Index([0, 2], name=COL_FRAME))
    out = K.fill_gaps(df)
    assert list(out.index) == [0, 1, 2]
    assert out.loc[1, "v"] == 1.0 and out.loc[1, "name"] == "a"


def test_smooth_is_a_low_pass_not_a_derivative():
    ramp = np.linspace(0, 10, 200)
    out = K.smooth(ramp, 3)
    assert np.allclose(out[20:-20], ramp[20:-20], atol=1e-6)


def test_smooth_keeps_nan_runs():
    x = np.ones(50)
    x[10:15] = np.nan
    out = K.smooth(x, 2)
    assert np.isnan(out[10:15]).all() and np.allclose(out[:10], 1) and np.allclose(out[15:], 1)


def test_finite_runs_and_unwrap():
    x = np.array([np.nan, 1, 2, np.nan, np.nan, 3, 4, 5])
    assert K.finite_runs(x) == [(1, 3), (5, 8)]
    a = np.array([3.0, -3.0, 3.0, np.nan, 0.0])
    out = K.unwrap(a)
    assert np.isnan(out[3]) and abs(out[1] - out[0]) < np.pi


def test_heading_and_speed_recover_a_known_turn():
    fps, n = 30.0, 300
    t = np.arange(n) / fps
    heading = np.where(t < 5, 0.0, np.deg2rad(60))
    x = np.cumsum(2.0 * np.cos(heading) / fps)
    y = np.cumsum(2.0 * np.sin(heading) / fps)
    yaw = K.heading_from_displacement(x, y, fps, min_speed_mps=0.1)
    assert abs(np.rad2deg(yaw[10])) < 1e-6
    assert abs(np.rad2deg(yaw[-10]) - 60) < 1e-6
    assert np.allclose(K.translational_speed(x, y, fps)[10:140], 2.0, atol=1e-6)


def test_heading_masked_when_stationary():
    x = np.ones(50)
    y = np.ones(50)
    yaw = K.heading_from_displacement(x, y, 30.0, min_speed_mps=0.1)
    assert np.isnan(yaw).all()


def test_yaw_velocity_units():
    fps = 25.0
    yaw = np.deg2rad(np.linspace(0, 100, 101))  # 1 deg per frame
    v = K.yaw_velocity_degps(yaw, fps)
    assert np.allclose(v[1:-1], 25.0)
    assert K.sigma_frames(0.1, fps) == 2.5
