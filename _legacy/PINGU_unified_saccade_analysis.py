"""
detect_saccades.py

Pipeline script to detect saccades (body only for 'open' dataset, body+head for 'closed' dataset)
based on the unified CSV with 'segment_id' and 'trajectory_id'.
We produce a final CSV with saccade + intersaccade rows, including median translational speed.
"""

import os
import numpy as np
import pandas as pd
from SaccadeAnalysis import SaccadeAnalysis

# Files
UNIFIED_CSV: str = "/path/to/Penguin_Rostock/unified_outputs/unified_trajectories_head_body.csv"
OUTPUT_SACCADES_CSV: str = "/path/to/Penguin_Rostock/unified_outputs/unified_saccade_intersaccade_results.csv"

# Thresholds & parameters
ANGLE_VEL_THRESHOLD: float = 200.0  # deg/s
TIME_THRESHOLD: float = 1.0         # for "associated" classification
WINDOW_LENGTH: int = 25

def find_saccades_in_signal(angles_deg, frame_rate, threshold):
    """
    Helper to run SaccadeAnalysis on a single 1D angle array (deg).
    Returns a DataFrame of saccades or empty DataFrame if none are found.
    """
    sa = SaccadeAnalysis(frame_rate=frame_rate)
    saccades, _, _, _, _ = sa.main(
        angles_deg, threshold=threshold, window_size=WINDOW_LENGTH, plot_now=False
    )
    return pd.DataFrame(saccades)

def associate_head_body_saccades(df_head, df_body, time_threshold):
    """
    Label head saccades as 'associated' if near a body saccade in time & direction;
    otherwise 'free'. Body saccades are given 'saccade_type' = 'body'.
    """
    df_head = df_head.copy()
    df_body = df_body.copy()
    df_head.sort_values("saccade_peak_s", inplace=True)
    df_body.sort_values("saccade_peak_s", inplace=True)

    sacc_type_list = []
    for _, row in df_head.iterrows():
        s_type = "free"
        direction = row["direction"]
        peak_time = row["saccade_peak_s"]
        # Filter body saccades by matching direction
        df_body_dir = df_body[df_body["direction"] == direction]
        if not df_body_dir.empty:
            diffs = (df_body_dir["saccade_peak_s"] - peak_time).abs()
            if diffs.min() < time_threshold:
                s_type = "associated"
        sacc_type_list.append(s_type)

    df_head["saccade_type"] = sacc_type_list
    df_body["saccade_type"] = "body"
    return df_head, df_body

def find_intersaccades(df_sacc, n_frames, frame_rate):
    """
    Identify intervals between saccades. If no saccade => entire range is intersaccadic.
    Returns a DataFrame of interval rows or empty if not applicable.
    """
    if df_sacc.empty:
        # One entire interval from 0..(n_frames-1)
        return pd.DataFrame([{
            "saccade_start_idx": 0,
            "saccade_stop_idx": n_frames - 1,
            "saccade_peak_s": (n_frames-1) / (2.0*frame_rate),
            "saccade_duration_s": (n_frames-1) / frame_rate,
            "angle_amplitude_deg": np.nan,
            "top_speed_degPs": np.nan,
            "direction": np.nan,
            "saccade_type": "intersaccadic",
        }])

    df_sacc_sorted = df_sacc.sort_values("saccade_start_idx", ignore_index=True)
    intervals = []

    # Leading
    first_start = df_sacc_sorted.iloc[0]["saccade_start_idx"]
    if first_start > 0:
        intervals.append({
            "saccade_start_idx": 0,
            "saccade_stop_idx": first_start-1,
            "saccade_peak_s": (first_start - 1)/ (2.0*frame_rate),
            "saccade_duration_s": (first_start)/frame_rate,
            "angle_amplitude_deg": np.nan,
            "top_speed_degPs": np.nan,
            "direction": np.nan,
            "saccade_type": "intersaccadic",
        })
    
    # Middle intervals
    for i in range(len(df_sacc_sorted)-1):
        this_stop = df_sacc_sorted.iloc[i]["saccade_stop_idx"]
        next_start = df_sacc_sorted.iloc[i+1]["saccade_start_idx"]
        if next_start > this_stop+1:
            intervals.append({
                "saccade_start_idx": this_stop+1,
                "saccade_stop_idx": next_start-1,
                "saccade_peak_s": ( (this_stop+1)+(next_start-1) )/(2.0*frame_rate),
                "saccade_duration_s": (next_start-1 - (this_stop+1))/frame_rate,
                "angle_amplitude_deg": np.nan,
                "top_speed_degPs": np.nan,
                "direction": np.nan,
                "saccade_type": "intersaccadic",
            })

    # Trailing
    last_stop = df_sacc_sorted.iloc[-1]["saccade_stop_idx"]
    if last_stop < n_frames-1:
        intervals.append({
            "saccade_start_idx": last_stop+1,
            "saccade_stop_idx": n_frames-1,
            "saccade_peak_s": ( (last_stop+1)+(n_frames-1) )/(2.0*frame_rate),
            "saccade_duration_s": (n_frames-1 - (last_stop+1))/frame_rate,
            "angle_amplitude_deg": np.nan,
            "top_speed_degPs": np.nan,
            "direction": np.nan,
            "saccade_type": "intersaccadic",
        })

    return pd.DataFrame(intervals)

def compute_median_trans_speed(df_sacc, df_traj):
    """
    Compute the median translational speed over [start..stop] 
    for each saccade or interval.
    """
    df_sacc = df_sacc.copy()
    speeds = []
    for i, row in df_sacc.iterrows():
        st = int(row["saccade_start_idx"])
        sp = int(row["saccade_stop_idx"])
        st = max(0, st)
        sp = min(sp, len(df_traj)-1)
        if "translational_velocity_mPs" not in df_traj.columns:
            speeds.append(np.nan)
            continue

        arr = df_traj["translational_velocity_mPs"].iloc[st : sp+1].to_numpy()
        speeds.append(np.median(arr) if len(arr) > 0 else np.nan)

    df_sacc["med_trans_speed_mPs"] = speeds
    return df_sacc

def main():
    """
    Main pipeline for detecting saccades in each trajectory_id 
    (body only in open, body + head in closed).
    If either body_sacc or head_sacc is empty (but not both), 
    we label the non-empty saccades as 'free'.
    If both are empty, the entire trajectory is an intersaccadic interval only.
    """
    if not os.path.isfile(UNIFIED_CSV):
        raise FileNotFoundError(f"Could not find: {UNIFIED_CSV}")

    df_trajectory = pd.read_csv(UNIFIED_CSV)
    if "trajectory_id" not in df_trajectory.columns:
        raise KeyError("unified CSV must contain 'trajectory_id'")

    all_results = []
    grouped = df_trajectory.groupby("trajectory_id")

    for traj_id, df_seg in grouped:
        dataset_vals = df_seg["dataset"].unique()
        if len(dataset_vals) > 1:
            # fallback if multiple dataset labels appear in the same trajectory
            dataset = dataset_vals[0]
        else:
            dataset = dataset_vals[0]

        # Choose a local frame rate based on the dataset
        local_fr = 30 if dataset == "open" else 25

        seg_id_vals = df_seg["segment_id"].unique()
        if len(seg_id_vals) == 1:
            seg_id = seg_id_vals[0]
        else:
            seg_id = seg_id_vals

        print(f"[INFO] trajectory_id={traj_id}, dataset={dataset}, segment_id={seg_id}, frames={len(df_seg)}")

        # Prepare body angle array
        body_ang_deg = df_seg["body_yaw_deg"].fillna(0).to_numpy() if "body_yaw_deg" in df_seg.columns else np.zeros(len(df_seg))
        # Prepare head angle array if closed
        head_ang_deg = df_seg["head_yaw_deg"].fillna(0).to_numpy() if "head_yaw_deg" in df_seg.columns else np.zeros(len(df_seg))

        if dataset == "open":
            # We skip HEAD saccade detection entirely
            df_body_sacc = find_saccades_in_signal(body_ang_deg, local_fr, ANGLE_VEL_THRESHOLD)
            df_body_sacc.rename(columns={"amplitude_deg":"angle_amplitude_deg"}, inplace=True)

            if df_body_sacc.empty:
                # No saccade => entire trajectory => intersaccadic
                df_sacc = pd.DataFrame(columns=[
                    "saccade_peak_s","saccade_start_idx","sacc_peak_idx",
                    "saccade_stop_idx","angle_amplitude_deg","top_speed_degPs",
                    "saccade_duration_s","saccade_source","saccade_type","direction"
                ])
            else:
                # All body saccades => saccade_source='body', saccade_type='none'
                df_body_sacc["saccade_source"] = "body"
                df_body_sacc["saccade_type"] = "none"  # or 'free'
                df_sacc = df_body_sacc

        else:
            # CLOSED => detect both head and body
            df_body_sacc = find_saccades_in_signal(body_ang_deg, local_fr, ANGLE_VEL_THRESHOLD)
            df_body_sacc.rename(columns={"amplitude_deg":"angle_amplitude_deg"}, inplace=True)
            df_body_sacc["saccade_source"] = "body"  

            df_head_sacc = find_saccades_in_signal(head_ang_deg, local_fr, ANGLE_VEL_THRESHOLD)
            df_head_sacc.rename(columns={"amplitude_deg":"angle_amplitude_deg"}, inplace=True)
            df_head_sacc["saccade_source"] = "head"

            if df_body_sacc.empty and df_head_sacc.empty:
                # Both empty => entire trajectory is intersaccadic
                df_sacc = pd.DataFrame(columns=[
                    "saccade_peak_s","saccade_start_idx","sacc_peak_idx",
                    "saccade_stop_idx","angle_amplitude_deg","top_speed_degPs",
                    "saccade_duration_s","saccade_source","saccade_type","direction"
                ])
            elif df_body_sacc.empty and not df_head_sacc.empty:
                # Only head => label them all 'free'
                df_head_sacc["saccade_type"] = "free"
                df_sacc = df_head_sacc
            elif df_head_sacc.empty and not df_body_sacc.empty:
                # Only body => label them all 'free' 
                df_body_sacc["saccade_type"] = "free"
                df_sacc = df_body_sacc
            else:
                # Both exist => associate them
                df_head_sacc, df_body_sacc = associate_head_body_saccades(df_head_sacc, df_body_sacc, TIME_THRESHOLD)
                df_sacc = pd.concat([df_head_sacc, df_body_sacc], ignore_index=True)

        # If df_sacc is empty => no saccade => entire intersacc
        if df_sacc.empty:
            df_sacc = pd.DataFrame(columns=[
                "saccade_peak_s","saccade_start_idx","sacc_peak_idx","saccade_stop_idx",
                "angle_amplitude_deg","top_speed_degPs","saccade_duration_s","saccade_source",
                "saccade_type","direction"
            ])

        # Additional columns
        df_sacc["dataset"] = dataset
        df_sacc["trajectory_id"] = traj_id
        df_sacc["segment_id"] = seg_id
        df_sacc["saccade_duration_ms"] = df_sacc["saccade_duration_s"]*1000

        # Make sure integer
        for c in ["saccade_start_idx","saccade_stop_idx","sacc_peak_idx"]:
            df_sacc[c] = df_sacc[c].fillna(0).astype(int)

        # Intersaccadic intervals
        df_intersacc = find_intersaccades(df_sacc, len(df_seg), local_fr)
        if not df_intersacc.empty:
            df_intersacc["dataset"] = dataset
            df_intersacc["trajectory_id"] = traj_id
            df_intersacc["segment_id"] = seg_id
            df_intersacc["saccade_duration_ms"] = df_intersacc["saccade_duration_s"]*1000

        # Merge
        df_events = pd.concat([df_sacc, df_intersacc], ignore_index=True)
        # Compute median translational speed
        df_events = compute_median_trans_speed(df_events, df_seg)

        all_results.append(df_events)

    if all_results:
        df_all = pd.concat(all_results, ignore_index=True)
    else:
        df_all = pd.DataFrame()

    # Final columns
    col_order = [
        "saccade_peak_s",
        "saccade_start_idx",
        "sacc_peak_idx",
        "saccade_stop_idx",
        "angle_amplitude_deg",
        "top_speed_degPs",
        "med_trans_speed_mPs",
        "saccade_duration_s",
        "saccade_duration_ms",
        "saccade_source",   # head, body, intersaccadic
        "saccade_type",     # none, free, associated, etc
        "direction",
        "dataset",
        "trajectory_id",
        "segment_id",
    ]
    for col in col_order:
        if col not in df_all.columns:
            df_all[col] = np.nan
    df_all = df_all[col_order]

    df_all.to_csv(OUTPUT_SACCADES_CSV, index=False)
    print(f"[INFO] Saccade + intersaccadic results saved:\n  {OUTPUT_SACCADES_CSV}")


if __name__=="__main__":
    main()
