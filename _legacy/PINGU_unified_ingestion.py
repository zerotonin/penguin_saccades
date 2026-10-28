"""
PINGU_unified_ingestion.py

Provides a straightforward ingestion of two datasets (open HDF5 and closed CSV),
assigning each file (open) or each CSV group (closed) a *globally unique* 
trajectory_id that increments by 1 as we go along.

We preserve the original 'segment_id' from the open dataset (if it exists)
and for the closed dataset we convert the 'Identifier' to 'segment_id'.
Hence, no grouping on dataset+segment_id is needed. We simply maintain 
a global integer counter for 'trajectory_id'.
"""

import os
import glob
import numpy as np
import pandas as pd

from peguin_ana01 import TrajectoryProcessor
from PINGU_HEAD_saccadeTypeSpecific_trigAvg import read_cvs_into_dataframe, calculate_translational_velocity

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

PATH_OPEN_DATA   = "/path/to/Penguin_Rostock/penguins/sorted_and_filtered/"  # HDF5
PATH_CLOSED_DATA = "/path/to/Penguin_Rostock/pengu_head_movies/"            # CSV
OUTPUT_PATH      = "/path/to/Penguin_Rostock/unified_outputs/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

IM_WIDTH = 1402
IM_HEIGHT = 788
FRAME_RATE_OPEN = 30
PIX2M = np.array([0.97/124.6, 0.3/66, 0.4/76.2]).mean()
FRAME_RATE_CLOSED = 25
CURRENT_TID = 0

# Final columns we expect
REQUIRED_COLS = [
    "segment_id",
    "trajectory_id",
    "frame_index",
    "body_yaw_rad",
    "body_yaw_speed_degPs",
    "head_yaw_rad",
    "head_yaw_speed_degPs",
    "translational_velocity_mPs",
    "x_position_m",
    "y_position_m",
    "species",
    "dataset",
]

def create_empty_unified_df() -> pd.DataFrame:
    """Empty DataFrame with required columns."""
    return pd.DataFrame(columns=REQUIRED_COLS)

def ingest_open_data(path_open: str) -> pd.DataFrame:
    """
    Reads all *.h5 files from the open dataset, then for each file:
      1) Process with TrajectoryProcessor to get df_interp.
      2) If df_interp has multiple segments (in column 'segment'), 
         each segment is treated as its own trajectory => new trajectory_id.
      3) If 'segment' is missing, default 'segment_id' = 0 => one trajectory in that file.
      4) Fill missing columns, label dataset='open', species from filename, etc.

    Returns
    -------
    pd.DataFrame
        A combined table of open data, with one row per frame. Each group of
        frames that share a segment_id in the same file gets a unique trajectory_id.
    """
    import glob
    import os

    pattern = os.path.join(path_open, "**", "*.h5")
    file_list = glob.glob(pattern, recursive=True)

    all_trajectories = []

    for file_path in file_list:
        df_raw = pd.read_hdf(file_path)
        if df_raw.empty:
            continue

        # Run TrajectoryProcessor
        tp = TrajectoryProcessor(df_raw, IM_WIDTH, IM_HEIGHT, PIX2M, FRAME_RATE_OPEN)
        df_interp, _ = tp.main()
        if df_interp is None or df_interp.empty:
            continue

        # If 'segment' does not exist, create it = 0 => single segment
        if "segment" not in df_interp.columns:
            df_interp["segment"] = 0

        # We now rename that to 'segment_id', but we'll keep it for reference
        # Then group by it, so each segment => new trajectory_id
        df_interp.rename(columns={"segment": "segment_id"}, inplace=True)

        # We'll group by segment_id within the same .h5 file
        grouped_segments = df_interp.groupby("segment_id")

        for seg_value, df_seg in grouped_segments:
            df_seg = df_seg.copy()

            # The user’s original segment ID is preserved as is
            # We also keep the original frame index
            df_seg["frame_index"] = df_seg.index

            # rename columns if present
            if "yaw_rad" in df_seg.columns:
                df_seg.rename(columns={"yaw_rad":"body_yaw_rad"}, inplace=True)
            if "rot_speed_degPs" in df_seg.columns:
                df_seg.rename(columns={"rot_speed_degPs":"body_yaw_speed_degPs"}, inplace=True)
            if "trans_speed_mPs" in df_seg.columns:
                df_seg.rename(columns={"trans_speed_mPs":"translational_velocity_mPs"}, inplace=True)
            if "center_of_mass_x" in df_seg.columns:
                df_seg.rename(columns={"center_of_mass_x":"x_position_m"}, inplace=True)
            if "center_of_mass_y" in df_seg.columns:
                df_seg.rename(columns={"center_of_mass_y":"y_position_m"}, inplace=True)

            # Insert missing head columns
            df_seg["head_yaw_rad"] = np.nan
            df_seg["head_yaw_speed_degPs"] = np.nan

            # Infer species from filename
            species = "gentoo"
            if "Rockhopper" in file_path:
                species = "rockhopper"
            elif "Gentoo" in file_path:
                species = "gentoo"
            df_seg["species"] = species

            df_seg["dataset"] = "open"
            global CURRENT_TID
            # Assign a new trajectory_id for this segment
            df_seg["trajectory_id"] = CURRENT_TID
            CURRENT_TID += 1

            # Ensure required columns
            for col in REQUIRED_COLS:
                if col not in df_seg.columns:
                    df_seg[col] = np.nan

            # Retain only needed columns
            df_seg = df_seg[REQUIRED_COLS].copy()
            all_trajectories.append(df_seg)

    if not all_trajectories:
        return create_empty_unified_df()

    df_open = pd.concat(all_trajectories, ignore_index=True)
    return df_open

def ingest_closed_data(path_closed: str) -> pd.DataFrame:
    """
    For each unique 'Identifier' group in the CSV data, 
    assign a new trajectory_id, rename columns, preserve 'Identifier' as segment_id.

    Returns
    -------
    pd.DataFrame
        Combined closed dataset with a globally unique trajectory_id for each identifier group.
    """
    df_all = read_cvs_into_dataframe(path_closed, frame_rate=FRAME_RATE_CLOSED)
    if df_all.empty:
        return create_empty_unified_df()

    grouped = df_all.groupby("Identifier")
    all_parts = []
   
    for ident_value, df_sub in grouped:
        df_sub = df_sub.copy()
        df_sub["frame_index"] = df_sub.index

        if "body_yaw_speed" in df_sub.columns:
            df_sub.rename(columns={"body_yaw_speed":"body_yaw_speed_degPs"}, inplace=True)
        if "head_yaw_speed" in df_sub.columns:
            df_sub.rename(columns={"head_yaw_speed":"head_yaw_speed_degPs"}, inplace=True)
        
        # rename 'Identifier' => 'segment_id'
        df_sub.rename(columns={"Identifier":"segment_id"}, inplace=True)

        # no x/y => NaN
        df_sub["x_position_m"] = np.nan
        df_sub["y_position_m"] = np.nan

        # species => gentoo, dataset => closed
        df_sub["species"] = "gentoo"
        df_sub["dataset"] = "closed"

        # new trajectory_id
        global CURRENT_TID
        df_sub["trajectory_id"] = CURRENT_TID
        CURRENT_TID += 1

        # fill missing
        for col in REQUIRED_COLS:
            if col not in df_sub.columns:
                df_sub[col] = np.nan

        df_out = df_sub[REQUIRED_COLS].copy()
        all_parts.append(df_out)

    if not all_parts:
        return create_empty_unified_df()
    return pd.concat(all_parts, ignore_index=True)

def main():
    """
    Main function that ingests open and closed datasets, assigning a 
    globally incremented 'trajectory_id' for each file (open) or 
    each Identifier group (closed). The final output is saved to CSV.
    """
    df_open = ingest_open_data(PATH_OPEN_DATA)
    df_closed = ingest_closed_data(PATH_CLOSED_DATA)

    df_unified = pd.concat([df_open, df_closed], ignore_index=True)

    # Sort primarily by dataset, then by trajectory_id, then by frame_index (optional)
    df_unified.sort_values(["dataset","trajectory_id","frame_index"], inplace=True, ignore_index=True)

    out_path = os.path.join(OUTPUT_PATH, "unified_trajectories_head_body.csv")
    df_unified.to_csv(out_path, index=False)
    print(f"[INFO] Final unified CSV saved:\n  {out_path}")

if __name__ == "__main__":
    main()
