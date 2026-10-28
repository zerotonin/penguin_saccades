import numpy as np
import pandas as pd
import scipy.stats

################################################################################
# 1) Saccade Detection
################################################################################
def detect_saccades(yaw_velocity_degPs, threshold=200):
    """
    Returns a list of saccade events with start/stop/peak indices.
    """
    saccade_list = []
    n = len(yaw_velocity_degPs)
    i = 0
    
    while i < n:
        if abs(yaw_velocity_degPs[i]) >= threshold:
            # Found a threshold crossing
            direction = 'left' if yaw_velocity_degPs[i] < 0 else 'right'
            peak_idx = i
            peak_val = yaw_velocity_degPs[i]
            
            # Search backwards for start
            sstart = peak_idx
            while sstart > 1 and yaw_velocity_degPs[sstart]*yaw_velocity_degPs[sstart-1] > 0:
                sstart -= 1
                
            # Search forwards for stop
            sstop = peak_idx
            while sstop < n-1 and yaw_velocity_degPs[sstop]*yaw_velocity_degPs[sstop+1] > 0:
                sstop += 1
            
            saccade_list.append({
                'saccade_start_idx': sstart,
                'saccade_peak_idx': peak_idx,
                'saccade_stop_idx': sstop,
                'direction': direction,
                'peak_velocity': peak_val
            })
            
            i = sstop + 1
        else:
            i += 1

    return saccade_list

################################################################################
# 2) Identify Inter-Saccadic Intervals
################################################################################
def identify_intersaccadic_intervals(saccade_list, total_length):
    """
    Find intervals in [0..total_length-1] that are outside saccades.
    """
    if not saccade_list:
        return [{'isi_start_idx': 0, 'isi_stop_idx': total_length - 1}]
    
    saccade_list_sorted = sorted(saccade_list, key=lambda x: x['saccade_start_idx'])
    intervals = []

    # Before first saccade
    first_start = saccade_list_sorted[0]['saccade_start_idx']
    if first_start > 0:
        intervals.append({'isi_start_idx': 0, 'isi_stop_idx': first_start - 1})
    
    # Between saccades
    for i in range(len(saccade_list_sorted)-1):
        left_stop = saccade_list_sorted[i]['saccade_stop_idx']
        right_start = saccade_list_sorted[i+1]['saccade_start_idx']
        if right_start > (left_stop + 1):
            intervals.append({
                'isi_start_idx': left_stop + 1,
                'isi_stop_idx': right_start - 1
            })
    
    # After last
    last_stop = saccade_list_sorted[-1]['saccade_stop_idx']
    if last_stop < total_length - 1:
        intervals.append({'isi_start_idx': last_stop + 1, 'isi_stop_idx': total_length - 1})
    return intervals

################################################################################
# 3) Center short intervals with NaNs
################################################################################
def center_data_in_chunk(array_1d, chunk_size=25):
    """
    If length < chunk_size, center the data with NaN padding.
    If length >= chunk_size, return the first chunk_size frames.
    """
    L = len(array_1d)
    if L == 0:
        return np.full(chunk_size, np.nan)
    if L >= chunk_size:
        return array_1d[:chunk_size]

    padded = np.full(chunk_size, np.nan)
    left_nans = (chunk_size - L) // 2
    padded[left_nans:left_nans+L] = array_1d
    return padded

################################################################################
# 4) Extract 1-s Chunks from each ISI
################################################################################
def extract_isi_chunks(
    intervals,
    body_yaw_rad,
    body_yaw_vel_degPs,
    head_yaw_rad,
    head_yaw_vel_degPs,
    transl_speed_mPs,
    frame_rate=25
):
    """
    For each ISI, split into 1-s chunks (25 frames). If leftover <25 frames,
    we center it with NaNs. Return a DataFrame with columns for each frame:
      bodyYawDeg_0..24, bodyYawVel_0..24, headYawDeg_0..24, headYawVel_0..24, translSpeed_0..24
    plus 'isi_index' and 'chunk_index'.
    """
    chunk_size = int(frame_rate * 1)  # 1 second => 25 frames at 25 fps
    rows = []

    for isi_idx, interval in enumerate(intervals):
        s = interval['isi_start_idx']
        e = interval['isi_stop_idx']
        length = e - s + 1
        if length <= 0:
            continue

        # Convert to deg
        body_deg_slice = np.degrees(body_yaw_rad[s:e+1])
        head_deg_slice = np.degrees(head_yaw_rad[s:e+1])

        b_vel_slice = body_yaw_vel_degPs[s:e+1]   # already deg/s
        h_vel_slice = head_yaw_vel_degPs[s:e+1]   # deg/s
        tsl_slice   = transl_speed_mPs[s:e+1]     # m/s

        num_full = length // chunk_size
        remainder = length % chunk_size

        # Full 1-s chunks
        for cidx in range(num_full):
            start_f = cidx * chunk_size
            end_f   = start_f + chunk_size

            row = make_chunk_dict(
                isi_index=isi_idx,
                chunk_index=cidx,
                body_yaw_deg=body_deg_slice[start_f:end_f],
                body_yaw_vel=b_vel_slice[start_f:end_f],
                head_yaw_deg=head_deg_slice[start_f:end_f],
                head_yaw_vel=h_vel_slice[start_f:end_f],
                transl_speed=tsl_slice[start_f:end_f]
            )
            rows.append(row)

        # Leftover
        if remainder > 0 or num_full == 0:
            # chunk_index = num_full if we had some full chunks; else 0
            leftover_start = num_full * chunk_size
            body_deg_lf = body_deg_slice[leftover_start : leftover_start + remainder]
            b_vel_lf    = b_vel_slice[leftover_start : leftover_start + remainder]
            head_deg_lf = head_deg_slice[leftover_start : leftover_start + remainder]
            h_vel_lf    = h_vel_slice[leftover_start : leftover_start + remainder]
            tsl_lf      = tsl_slice[leftover_start : leftover_start + remainder]

            bdg_pad = center_data_in_chunk(body_deg_lf, chunk_size)
            bvl_pad = center_data_in_chunk(b_vel_lf, chunk_size)
            hdg_pad = center_data_in_chunk(head_deg_lf, chunk_size)
            hvl_pad = center_data_in_chunk(h_vel_lf, chunk_size)
            tsl_pad = center_data_in_chunk(tsl_lf, chunk_size)

            row = make_chunk_dict(
                isi_index=isi_idx,
                chunk_index=num_full,
                body_yaw_deg=bdg_pad,
                body_yaw_vel=bvl_pad,
                head_yaw_deg=hdg_pad,
                head_yaw_vel=hvl_pad,
                transl_speed=tsl_pad
            )
            rows.append(row)

    return pd.DataFrame(rows)

def center_data_in_chunk(array_1d, chunk_size=25):
    """
    If length < chunk_size, center the data with NaN padding.
    If length >= chunk_size, return the first chunk_size frames (no shift).
    """
    L = len(array_1d)
    if L == 0:
        return np.full(chunk_size, np.nan)
    if L >= chunk_size:
        return array_1d[:chunk_size]

    padded = np.full(chunk_size, np.nan)
    left_nans = (chunk_size - L) // 2
    padded[left_nans:left_nans+L] = array_1d
    return padded

def offset_first_valid(array):
    """
    Subtract the first non-NaN element from the entire array.
    If all NaNs, does nothing. This is for angle arrays.
    """
    idx = np.where(~np.isnan(array))[0]
    if len(idx) > 0:
        offset = array[idx[0]]
        array = array - offset
    return array

def make_chunk_dict(
    isi_index,
    chunk_index,
    body_yaw_deg,
    body_yaw_vel,
    head_yaw_deg,
    head_yaw_vel,
    transl_speed
):
    """
    Build a dict with columns for each frame: bodyYawDeg_0..24, bodyYawVel_0..24, etc.
    **New**: We subtract the 'first valid angle' from bodyYawDeg & headYawDeg in each chunk.
    """
    # 1) Offset angles so chunk starts at zero
    body_yaw_deg = offset_first_valid(body_yaw_deg)
    head_yaw_deg = offset_first_valid(head_yaw_deg)

    row = {
        'isi_index': isi_index,
        'chunk_index': chunk_index
    }
    chunk_size = len(body_yaw_deg)  # typically 25
    for i in range(chunk_size):
        row[f'bodyYawDeg_{i}'] = body_yaw_deg[i]
        row[f'bodyYawVel_{i}'] = body_yaw_vel[i]
        row[f'headYawDeg_{i}'] = head_yaw_deg[i]
        row[f'headYawVel_{i}'] = head_yaw_vel[i]
        row[f'translSpeed_{i}'] = transl_speed[i]
    return row

################################################################################
# 5) Main Pipeline
################################################################################
def main_intersaccade_analysis(
    df,      # DataFrame for a single trajectory or group
    body_yaw_col='body_yaw_rad',
    body_vel_col='body_yaw_speed_degPs',
    head_yaw_col='head_yaw_rad',
    head_vel_col='head_yaw_speed_degPs',
    transl_col='translational_velocity_mPs',
    threshold=200,
    frame_rate=25
):
    """
    1) Detect saccades using 'body_yaw_speed_degPs'
    2) Identify inter-saccadic intervals
    3) Chunk each ISI into 1-s segments
    4) Return DataFrame in wide format (with columns for each frame)
    """
    # 1) Saccade detection
    body_vel = df[body_vel_col].values  # deg/s
    saccades = detect_saccades(body_vel, threshold=threshold)

    # 2) Intervals
    total_len = len(df)
    intervals = identify_intersaccadic_intervals(saccades, total_len)

    # 3) Extract chunked data
    body_yaw_rad = df[body_yaw_col].values
    head_yaw_rad = df[head_yaw_col].values
    head_vel_deg = df[head_vel_col].values
    transl_speed = df[transl_col].values

    chunk_df = extract_isi_chunks(
        intervals=intervals,
        body_yaw_rad=body_yaw_rad,
        body_yaw_vel_degPs=body_vel,
        head_yaw_rad=head_yaw_rad,
        head_yaw_vel_degPs=head_vel_deg,
        transl_speed_mPs=transl_speed,
        frame_rate=frame_rate
    )
    return chunk_df

################################################################################
# 6) Melt the chunked DataFrame => compute stats by species, dataset, variable, frame
################################################################################
def melt_chunks_for_stats(chunk_df, species, dataset, trajectory_id):
    """
    Convert from wide format (bodyYawDeg_0..24, etc.) into long format:
      species, dataset, trajectory_id, isi_index, chunk_index, frame, variable, value
    Where variable is one of:
      'bodyYawDeg', 'bodyYawVel', 'headYawDeg', 'headYawVel', 'translSpeed'
    """
    # We'll gather columns that contain e.g. 'bodyYawDeg_', 'bodyYawVel_', etc.
    id_cols = ['isi_index', 'chunk_index']
    # Add extra identifying info
    chunk_df['species'] = species
    chunk_df['dataset'] = dataset
    chunk_df['trajectory_id'] = trajectory_id

    # Move them to front for clarity
    id_cols_extended = ['species','dataset','trajectory_id'] + id_cols

    # We want to produce rows of the form: (species, dataset, trajectory, isi_index, chunk_index, frame, variable, value).
    # Let's gather all columns that match a pattern: variableName_<frame>.
    # e.g. 'bodyYawDeg_13' => variable='bodyYawDeg', frame=13, value=...
    # We'll do it by scanning the columns:
    melted_rows = []
    for col in chunk_df.columns:
        # skip standard or id columns
        if col in id_cols_extended:
            continue
        # parse e.g. 'bodyYawDeg_13'
        if '_' not in col:
            continue
        var_name, frame_str = col.rsplit('_', 1)
        if not frame_str.isdigit():
            continue
        frame_idx = int(frame_str)
        # Now gather the series
        for row_i, val in enumerate(chunk_df[col]):
            row_data = {
                'species':       chunk_df.at[row_i, 'species'],
                'dataset':       chunk_df.at[row_i, 'dataset'],
                'trajectory_id': chunk_df.at[row_i, 'trajectory_id'],
                'isi_index':     chunk_df.at[row_i, 'isi_index'],
                'chunk_index':   chunk_df.at[row_i, 'chunk_index'],
                'variable':      var_name,   # e.g. 'bodyYawDeg'
                'frame':         frame_idx,  # 0..24
                'value':         val
            }
            melted_rows.append(row_data)

    return pd.DataFrame(melted_rows)

def compute_confidence_interval(arr, confidence=0.95):
    """
    Compute mean, median, sem, and the lower/upper confidence intervals (t-based).
    """
    arr_clean = arr[~np.isnan(arr)]
    n = len(arr_clean)
    if n < 2:
        return {
            'mean': np.nan if n==0 else arr_clean[0],
            'median': np.nan if n==0 else arr_clean[0],
            'sem': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan
        }
    mean_ = np.mean(arr_clean)
    median_ = np.median(arr_clean)
    sem_ = scipy.stats.sem(arr_clean)
    t_val = scipy.stats.t.ppf((1+confidence)/2, n-1)
    ci_lower = mean_ - t_val*sem_
    ci_upper = mean_ + t_val*sem_
    return {
        'mean': mean_,
        'median': median_,
        'sem': sem_,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }

def compute_stats_by_species_dataset_variable_frame(melted_df):
    """
    Group by (species, dataset, variable, frame) and compute mean, median, sem, 95% CI.
    Returns a summary DataFrame.
    """
    group_cols = ['species','dataset','variable','frame']
    
    def stats_func(sub):
        vals = sub['value'].values
        st = compute_confidence_interval(vals, confidence=0.95)
        return pd.Series(st)
    
    summary = melted_df.groupby(group_cols).apply(stats_func).reset_index()
    # columns: species, dataset, variable, frame, mean, median, sem, ci_lower, ci_upper
    return summary

################################################################################
# 7) Putting It All Together for Multiple Datasets / Species
################################################################################
if __name__ == "__main__":
    # Example: read your "unified" CSV which presumably has columns:
    # [dataset, species, trajectory_id, body_yaw_rad, body_yaw_speed_degPs,
    #  head_yaw_rad, head_yaw_speed_degPs, translational_velocity_mPs, ...]
    df = pd.read_csv("/path/to/Penguin_Rostock/unified_outputs/unified_trajectories_head_body.csv")

    # We'll build a final melted table of all chunks from all trajectories
    all_melted = []

    # Group by dataset + species + trajectory
    for (ds, sp, tid), group in df.groupby(["dataset","species","trajectory_id"]):
        # For each group, find inter-saccade intervals, chunk them
        chunk_df = main_intersaccade_analysis(
            group,
            body_yaw_col='body_yaw_rad',
            body_vel_col='body_yaw_speed_degPs',
            head_yaw_col='head_yaw_rad',
            head_vel_col='head_yaw_speed_degPs',
            transl_col='translational_velocity_mPs',
            threshold=200,
            frame_rate=25
        )
        # Melt the chunk DataFrame => long format with one row per (frame, variable)
        melted = melt_chunks_for_stats(chunk_df, species=sp, dataset=ds, trajectory_id=tid)
        all_melted.append(melted)

    # Concatenate everything
    big_melted_df = pd.concat(all_melted, ignore_index=True)

    # Now compute stats: group by species, dataset, variable, frame
    summary_stats_df = compute_stats_by_species_dataset_variable_frame(big_melted_df)

    # Save both if you wish
    big_melted_df.to_csv("/path/to/Penguin_Rostock/unified_outputs/all_isi_chunks_melted.csv", index=False)
    summary_stats_df.to_csv("/path/to/Penguin_Rostock/unified_outputs/isi_chunk_summary_stats.csv", index=False)

    print("Done! Summary stats example:")
    print(summary_stats_df.head(20))
