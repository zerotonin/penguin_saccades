import numpy as np
import pandas as pd

from penguin_saccades import constants as C
from penguin_saccades import isi as I
from penguin_saccades import saccades as S
from penguin_saccades.ingest_open import process_open_trajectory
from tests.conftest import open_track

FPS = {"open": C.DATASET_OPEN.frame_rate_hz, "closed": C.DATASET_CLOSED.frame_rate_hz}


def _traj():
    traj = process_open_trajectory(open_track(saccades=[(3.0, 60.0), (7.0, -60.0)]))
    traj[C.COL_SPECIES] = "gentoo"
    traj[C.COL_TRAJECTORY] = 0
    sacc, ints = S.detect_in_trajectories(traj, C.COL_BODY_YAW, "body", FPS)
    ints[C.COL_INTERVAL_CLASS] = "body"
    return traj, sacc, ints


def test_budget_puts_nearly_all_yaw_inside_saccades():
    traj, sacc, _ = _traj()
    b = I.saccade_budget(traj, sacc, C.COL_BODY_YAW).iloc[0]
    assert b.n_saccades == 2
    assert b.yaw_fraction_in_saccades > 0.85
    assert b.time_fraction_in_saccades < 0.2


def test_interval_content_is_straight_and_fast_between_saccades():
    traj, sacc, ints = _traj()
    content = I.interval_content(traj, ints)
    middle = content[~content.censored].iloc[0]
    assert abs(middle.net_yaw_deg) < 3 and middle.path_yaw_deg < 5
    assert abs(middle.median_translation_mps - 2.0) < 0.1
    sc = I.interval_content(traj, sacc)
    assert (sc.path_yaw_deg > 50).all()


def test_chunks_are_one_second_long_and_zero_offset():
    traj, _, ints = _traj()
    chunks = I.chunk_intervals(traj, ints, FPS)
    size = round(C.ISI_CHUNK_S * FPS["open"])
    assert set(chunks.frame.unique()) == set(range(size))
    ang = chunks[chunks.variable == "body_yaw_deg"]
    first = ang.groupby(["interval_id", "chunk_id"]).apply(lambda g: g.dropna().value.iloc[0])
    assert np.allclose(first, 0.0)


def test_head_compensation_detects_gaze_stabilisation():
    n = 100
    body_v = np.sin(np.linspace(0, 6, n)) * 50
    traj = pd.DataFrame({
        C.COL_FRAME: np.arange(n), C.COL_BODY_VEL: body_v, C.COL_HEAD_VEL: 0.0 * body_v,
        C.COL_DATASET: "closed", C.COL_SPECIES: "gentoo", C.COL_TRAJECTORY: 0, C.COL_SEGMENT: 0,
    })
    ints = pd.DataFrame({"start_idx": [0], "stop_idx": [n - 1], C.COL_DATASET: "closed",
                         C.COL_SPECIES: "gentoo", C.COL_TRAJECTORY: 0, C.COL_SEGMENT: 0})
    comp = I.head_body_compensation(traj, ints)
    assert abs(comp.slope_head_per_body.iloc[0]) < 1e-9 and abs(comp.slope_neck_per_body.iloc[0] + 1) < 1e-9
