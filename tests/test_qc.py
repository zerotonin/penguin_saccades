"""Quality control on synthetic poses with known dropouts and swaps."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from penguin_saccades import constants as C
from penguin_saccades import qc
from tests.conftest import dlc_track, piecewise_heading

FPS = C.DATASET_CLOSED.frame_rate_hz


def pose_with_holes() -> tuple[pd.DataFrame, int, int]:
    """A clean track with 5 low-score beak frames and 31 undetected frames."""
    pose = dlc_track(duration_s=12.0, body_saccades=[], head_saccades=[])
    pose.loc[100:104, "beak_likelihood"] = 0.1
    undetected = slice(200, 230)
    coords = [f"{kp}_{axis}" for kp in C.DLC_KEYPOINTS for axis in ("x", "y")]
    pose.loc[undetected, coords] = np.nan
    pose.loc[undetected, [f"{kp}_likelihood" for kp in C.DLC_KEYPOINTS]] = 0.0
    return pose, 5, 31


def test_frame_budget_counts_every_gate() -> None:
    pose, n_low, n_missing = pose_with_holes()
    counts = qc.frame_budget(pose, C.DATASET_CLOSED, C.DLC_LIKELIHOOD_THRESHOLD)
    n = len(pose)
    assert counts["frames"] == n and counts["detected"] == n - n_missing
    assert counts["all_points"] == n - n_missing - n_low
    assert counts["head_points"] == n - n_missing - n_low and counts["body_points"] == n - n_missing
    assert counts["both_yaws"] == n - n_missing          # the 5-frame hole is bridged
    assert counts["plausible"] == counts["valid"] == n - n_missing
    assert counts["in_segments"] == n - n_missing        # both runs are longer than 1 s
    assert list(counts) == list(C.QC_GATES)


def test_yaw_jitter_is_zero_on_a_clean_track_and_sees_a_beak_swap() -> None:
    pose, _, _ = pose_with_holes()
    clean = qc.yaw_jitter(pose, C.DLC_LIKELIHOOD_THRESHOLD, FPS)
    assert clean["head_jitter_deg"] < 1e-6 and clean["head_jump_share"] == 0.0
    assert clean["body_length_px"] > clean["head_length_px"] > 0
    swapped = pose.copy()
    swapped.loc[50, "beak_x"] = 2 * swapped.loc[50, "neck_x"] - swapped.loc[50, "beak_x"]   # beak behind the neck
    assert qc.yaw_jitter(swapped, C.DLC_LIKELIHOOD_THRESHOLD, FPS)["head_jump_share"] > 0.0


def test_keypoint_scores_skip_undetected_frames() -> None:
    pose, _, n_missing = pose_with_holes()
    table = qc.keypoint_scores(pose, "clip")
    assert set(table["keypoint"]) == set(C.DLC_KEYPOINTS)
    assert (table["keypoint"] == "beak").sum() == len(pose) - n_missing


def test_saccade_rejections_name_the_failed_rule() -> None:
    n = int(10 * FPS)
    t = np.arange(n) / FPS
    head = piecewise_heading(t, [(3.0, 60.0), (7.0, 60.0)])
    bridged = np.zeros(n, dtype=bool)
    bridged[int(6.9 * FPS):int(7.3 * FPS)] = True            # 40 % of the window, below the window rule
    trajectories = pd.DataFrame({
        C.COL_DATASET: "closed", C.COL_SPECIES: "gentoo", C.COL_TRAJECTORY: 0, C.COL_SEGMENT: 0,
        C.COL_HEAD_YAW: head, C.COL_BODY_YAW: np.zeros(n), C.COL_BRIDGED: bridged,
    })
    rejections = qc.saccade_rejections(trajectories, {"closed": FPS})
    head_peaks = rejections[rejections[C.COL_SOURCE] == "head"]
    assert sorted(head_peaks["reason"]) == ["bridged_peak", "kept"]
    assert (rejections[C.COL_SOURCE] == "body").sum() == 0
    counts = qc.rejection_counts(rejections)
    assert counts.set_index("reason")["n"].to_dict() == {"bridged_peak": 1, "kept": 1}


def test_figures_draw_on_small_tables() -> None:
    pose, _, _ = pose_with_holes()
    budget = pd.DataFrame([{"movie": "clip", C.COL_DATASET: "closed", C.COL_SPECIES: "gentoo",
                            **qc.frame_budget(pose, C.DATASET_CLOSED, C.DLC_LIKELIHOOD_THRESHOLD)}])
    ax = qc.plot_frame_budget(budget)
    assert len(ax.lines) == 1
    axes = qc.plot_keypoint_scores(qc.keypoint_scores(pose, "clip"), C.DLC_LIKELIHOOD_THRESHOLD)
    assert len(axes) == len(C.DLC_KEYPOINTS)
    n = len(pose)
    trajectories = pd.DataFrame({
        C.COL_DATASET: "closed", C.COL_SPECIES: "gentoo", C.COL_TRAJECTORY: 0, C.COL_SEGMENT: 0,
        C.COL_FRAME: np.arange(n), C.COL_BODY_VEL: 5.0, C.COL_HEAD_VEL: 10.0,
        C.COL_BRIDGED: False, "source_file": "clip.csv", "qc_group": "closed gentoo",
    })
    _, table = qc.plot_noise_floor(trajectories)
    assert set(table["part"]) == {"body", "head"}
    detected = pose["neck_x"].notna().to_numpy()
    axes = qc.plot_detection_timeline([("clip.csv", detected, FPS)], trajectories)
    runs = qc.timeline_runs([("clip.csv", detected, FPS)], trajectories)
    assert len(axes) == 1 and set(runs["kind"]) == {"detected", "usable"}
    assert np.isclose(runs.loc[runs["kind"] == "detected", "stop_s"].max(), n / FPS)
    matplotlib.pyplot.close("all")
