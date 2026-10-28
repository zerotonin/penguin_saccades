import numpy as np
import pandas as pd

from penguin_saccades import constants as C
from penguin_saccades import taxonomy as T

KEYS = {C.COL_DATASET: "closed", C.COL_SPECIES: "gentoo", C.COL_TRAJECTORY: 0, C.COL_SEGMENT: 0}


def _sacc(peaks, directions, fps=25.0):
    return pd.DataFrame({
        "start_idx": [p - 3 for p in peaks], "peak_idx": peaks, "stop_idx": [p + 3 for p in peaks],
        "peak_time_s": [p / fps for p in peaks], "duration_s": 7 / fps,
        "amplitude_deg": [30 if d == "right" else -30 for d in directions],
        "peak_speed_degps": [400 if d == "right" else -400 for d in directions],
        "direction": directions, **KEYS,
    })


def test_association_rule():
    head = _sacc([50, 100, 200], ["right", "left", "right"])
    body = _sacc([55, 100, 400], ["right", "right", "right"])
    head, body = T.associate_head_body(head, body, window_s=0.5)
    assert head[C.COL_SACCADE_TYPE].tolist() == ["associated", "free_head", "free_head"]
    assert body[C.COL_SACCADE_TYPE].tolist() == ["associated", "free_body", "free_body"]
    assert abs(head.head_lead_s.iloc[0] - (55 - 50) / 25.0) < 1e-9 and head.partner_body_idx.iloc[0] == 55


def test_classified_intervals_have_three_classes_when_head_exists():
    n = 300
    traj = pd.DataFrame({C.COL_FRAME: np.arange(n), C.COL_HEAD_YAW: 0.0, C.COL_BODY_YAW: 0.0, **KEYS})
    head = _sacc([100], ["right"])
    body = _sacc([200], ["right"])
    head, body = T.associate_head_body(head, body)
    ints = T.classified_intervals(traj, head, body, {"closed": 25.0})
    counts = ints.groupby(C.COL_INTERVAL_CLASS).size().to_dict()
    assert counts == {"any": 3, "body": 2, "head": 2}
    assert (ints.stop_idx >= ints.start_idx).all()
