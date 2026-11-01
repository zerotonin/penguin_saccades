"""Figure helpers on synthetic trajectory tables."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from penguin_saccades import constants as C
from penguin_saccades import viz


def test_yaw_velocity_histogram_counts_every_finite_frame() -> None:
    rng = np.random.default_rng(C.RANDOM_SEED)
    n = 500
    trajectories = pd.DataFrame({
        C.COL_DATASET: ["open"] * n + ["closed"] * n,
        C.COL_SPECIES: ["gentoo"] * n + ["gentoo"] * n,
        C.COL_BODY_VEL: rng.normal(0, 50, 2 * n),
        C.COL_HEAD_VEL: np.r_[np.full(n, np.nan), rng.normal(0, 300, n)],
    })
    trajectories.loc[0, C.COL_BODY_VEL] = np.nan
    fig, table = viz.plot_yaw_velocity_histograms(trajectories)
    matplotlib.pyplot.close(fig)
    totals = table.groupby([C.COL_DATASET, "part"])["frames"].sum()
    in_range = lambda s: int(((s.abs() >= C.YAW_VELOCITY_HIST_BINS[0]) & (s.abs() <= C.YAW_VELOCITY_HIST_BINS[1])).sum())  # noqa: E731
    assert totals[("open", "body")] == in_range(trajectories.loc[:n - 1, C.COL_BODY_VEL])
    assert totals[("closed", "head")] == in_range(trajectories.loc[n:, C.COL_HEAD_VEL])
    assert ("open", "head") not in totals.index


def test_pooled_triggered_matrices_resample_the_slow_camera_onto_the_fast_grid() -> None:
    rates = {"fast": 60.0, "slow": 20.0}
    rows = []
    for dataset, fps in rates.items():
        n = int(4 * fps)
        t = np.arange(n) / fps
        rows.append(pd.DataFrame({
            C.COL_DATASET: dataset, C.COL_SPECIES: "gentoo", C.COL_TRAJECTORY: 0, C.COL_SEGMENT: 0,
            C.COL_FRAME: np.arange(n), C.COL_TIME: t,
            C.COL_BODY_YAW: np.deg2rad(45.0 * (t - 2.0)), C.COL_BODY_VEL: np.full(n, 45.0),
        }))
    trajectories = pd.concat(rows, ignore_index=True)
    saccades = pd.DataFrame({
        C.COL_DATASET: list(rates), C.COL_TRAJECTORY: 0, C.COL_SEGMENT: 0,
        "peak_idx": [int(2 * fps) for fps in rates.values()], "direction": C.DIRECTION_OF_POSITIVE_VELOCITY,
    })
    mat_v, mat_a, cameras, time_s = viz.pooled_triggered_matrices(
        trajectories, saccades, C.COL_BODY_YAW, C.COL_BODY_VEL, window_s=1.0, frame_rates=rates, target_fps=60.0)
    assert mat_v.shape == (2, 61) and list(cameras) == ["fast", "slow"]
    assert np.allclose(mat_v, 45.0)
    # the 20 fps ramp, resampled, is the same 45 deg/s ramp as the 60 fps one (offsets differ by design)
    centre = mat_a.shape[1] // 2
    assert np.allclose(mat_a[1] - mat_a[1, centre], mat_a[0] - mat_a[0, centre], atol=1e-9)
    assert np.isclose(time_s[-1] - time_s[0], 1.0)


def test_resample_rows_leaves_the_outside_of_the_finite_support_nan() -> None:
    source_t = np.array([0.0, 1.0, 2.0, 3.0])
    matrix = np.array([[np.nan, 1.0, 2.0, np.nan], [np.nan, np.nan, np.nan, 1.0]])
    out = viz.resample_rows(matrix, source_t, np.array([0.5, 1.5, 2.5]))
    assert np.isnan(out[0, 0]) and np.isclose(out[0, 1], 1.5) and np.isnan(out[0, 2])
    assert np.isnan(out[1]).all()
