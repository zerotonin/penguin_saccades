import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import scipy.stats
import SaccadeAnalysis
import seaborn as sns
import scipy.stats


def butter_lowpass_filter(data, cutoff_freq, sample_rate):
    """Applies a Butterworth low-pass filter to the data.

    Args:
        data (array-like): The data to be filtered.
        cutoff_freq (float): The cutoff frequency of the filter.
        sample_rate (float): The sampling rate of the data.

    Returns:
        array-like: The filtered data.
    """
    # Calculate the Nyquist frequency
    nyquist = 0.5 * sample_rate
    
    # Normalize the cutoff frequency by the Nyquist frequency
    normal_cutoff = cutoff_freq / nyquist
    
    # Design the filter
    b, a = butter(1, normal_cutoff, btype='low', analog=False)
    
    # Apply the filter
    y = filtfilt(b, a, data)
    return y

def calculate_vector_angle(df, point1, point2):
    """Calculates the angle of the vector formed by two points with respect to the x-axis.

    Args:
        df (pd.DataFrame): DataFrame containing the coordinates of the points.
        point1 (str): Name of the first point (e.g., 'neck').
        point2 (str): Name of the second point (e.g., 'beak').

    Returns:
        pd.Series: A Pandas Series containing the angles in radians.
    """
    
    # Calculate the vector components
    dx = df[f'{point2}_x'] - df[f'{point1}_x']
    dy = df[f'{point2}_y'] - df[f'{point1}_y']
    
    # Calculate the angle using arctan2
    angles = np.arctan2(dy, dx)
    
    # Unwrap the angles to avoid discontinuity
    angles = np.unwrap(angles)
    
    return angles

def calculate_translational_velocity(df, frame_rate=25, mean_body_len_m = 0.8):
    """Calculates the translational velocity based on neck movement, in units of body lengths per second.

    Args:
        df (pd.DataFrame): DataFrame containing x and y coordinates for neck and caudal, and other required columns.
        frame_rate (int): The frame rate (fps) to calculate the velocity. Default is 25 fps.

    Returns:
        pd.DataFrame: DataFrame with an added 'translational_velocity' column.
    """
    # Calculate body length as the sum of the neck-to-beak and neck-to-caudal vectors
    df['neck_caudal_length'] = np.sqrt((df['neck_x'] - df['caudal_x'])**2 + (df['neck_y'] - df['caudal_y'])**2)
    df['beak_neck_length'] = np.sqrt((df['beak_x'] - df['neck_x'])**2 + (df['beak_y'] - df['neck_y'])**2)
    df['body_length'] = df['neck_caudal_length'] + df['beak_neck_length']
    df['body_length'].interpolate(method='linear', inplace=True)
    df['body_length'] = butter_lowpass_filter(df['body_length'], 0.995, frame_rate)

    # Calculate displacement of the neck across frames
    df['neck_disp_x'] = df['neck_x'].diff()
    df['neck_disp_y'] = df['neck_y'].diff()
    df['neck_displacement'] = np.sqrt(df['neck_disp_x']**2 + df['neck_disp_y']**2)
    df.loc[0, 'neck_displacement'] = df.loc[1, 'neck_displacement'] # first value cannot be determined in time array

    # Calculate translational velocity in units of body lengths per second
    df['translational_velocity'] = (df['neck_displacement'] / df['body_length']) * frame_rate
    df['translational_velocity'].interpolate(method='linear', inplace=True)

    
    #df['translational_velocity'] = butter_lowpass_filter(df['translational_velocity'], 0.995, frame_rate)
    
    # Translational velocity in meters per second
    df['translational_velocity_mPs'] = df['translational_velocity'] * mean_body_len_m

    return df

def read_cvs_into_dataframe(target_folder,frame_rate=25):
    """Reads all CSV files in a target folder into a Pandas DataFrame and adds an 'Identifier' field.

    Args:
        target_folder (str): Path to the folder containing the CSV files.

    Returns:
        pd.DataFrame: DataFrame containing all data from the CSV files, with an added 'Identifier' column.
    """
    
    all_data = []
    
    for filename in os.listdir(target_folder):
        if filename.endswith('.csv'):
            filepath = os.path.join(target_folder, filename)
            
            # Extract the first two digits from the filename
            identifier = ''.join(filter(str.isdigit, filename))[:2]
            
            # Read the CSV file into a DataFrame, skipping the first row and combining the next two rows as header
            df = pd.read_csv(filepath, skiprows=[0], header=[0, 1])
            
            # Combine multi-level columns into a single level with underscore
            df.columns = ['_'.join(col).strip() for col in df.columns.values]

            df = calculate_translational_velocity(df)

            # Calculate the angle for the vector from neck to beak
            df['head_yaw_rad'] = calculate_vector_angle(df, 'neck', 'beak')
            
            # Calculate the angle for the vector from caudal to neck
            df['body_yaw_rad'] = calculate_vector_angle(df, 'caudal', 'neck')
            

            # Apply a simple moving average for smoothing (window size = 5)
            #df['head_yaw_rad_smoothed'] = df['head_yaw_rad'].rolling(window=5).mean()
            #df['body_yaw_rad_smoothed'] = df['body_yaw_rad'].rolling(window=5).mean()
            df['head_yaw_rad'] = butter_lowpass_filter(df['head_yaw_rad'], 0.995, 25)
            df['body_yaw_rad'] = butter_lowpass_filter(df['body_yaw_rad'], 0.995, 25)


            # Convert angles to degrees
            df['head_yaw_deg'] = np.degrees(df['head_yaw_rad'])
            df['body_yaw_deg'] = np.degrees(df['body_yaw_rad'])
            
            # Calculate the first derivative to get the rate of change of the angle
            df['head_yaw_speed'] = df['head_yaw_deg'].diff() * frame_rate
            df['body_yaw_speed'] = df['body_yaw_deg'].diff() * frame_rate
    
            # Add the identifier to a new column
            df['Identifier'] = identifier
            
            all_data.append(df)
    
    # Combine all individual DataFrames into one DataFrame
    combined_df = pd.concat(all_data, ignore_index=True)
    
    return combined_df

def create_saccade_dataframe(name, sacc_dir, sacc_type, sacc_trigger, data_type, triggered_data):
    """
    Creates a DataFrame for a single saccade or a group of saccades with meta data and triggered data.
    
    Args:
    - name (str): Identifier of the saccade group.
    - sacc_dir (str): Direction of the saccade ('positive' or 'negative').
    - sacc_type (str): Type of the saccade ('free' or 'associated').
    - sacc_trigger (str): On which saccade source was this analysis triggered: 'head' or 'body'.
    - data_type (str): Type of data ('body_angle', 'head_velocity', etc.).
    - triggered_data (list or np.array): Triggered data points for this saccade group.
    
    Returns:
    - pd.DataFrame: DataFrame containing the saccade information and triggered data or an empty DataFrame if no data is provided.
    """

    # Check if triggered_data is empty or None
    if triggered_data is None or len(triggered_data) == 0:
        return pd.DataFrame()

    # Create a dictionary for the DataFrame
    data_dict = {
        'id': name,
        'sacc_dir': sacc_dir,
        'sacc_type': sacc_type,
        'sacc_trigger': sacc_trigger,
        'data_type': data_type
    }

    # Add each data point to the dictionary
    for i, data_point in enumerate(triggered_data):
        data_dict[f'time_{i}'] = data_point

    # Convert the dictionary to a DataFrame
    return pd.DataFrame([data_dict])

def identify_saccades(group, sa, angle_vel_threshold, window_length):
    """
    Identifies saccades for head and body.

    Args:
        group (pd.DataFrame): DataFrame group associated with the identifier.
        sa (SaccadeAnalysis): Instance of SaccadeAnalysis class.
        angle_vel_threshold (float): Threshold for angular velocity to identify saccades.
        window_length (int): Length of the window used in saccade analysis.

    Returns:
        tuple: (head_saccades, body_saccades)
    """
    head_saccades, _, _, _, _ = sa.main(group['head_yaw_rad'].to_numpy(), angle_vel_threshold, window_length, False)
    body_saccades, _, _, _, _ = sa.main(group['body_yaw_rad'].to_numpy(), angle_vel_threshold, window_length, False)
    return head_saccades, body_saccades

def categorize_saccades(head_saccades, body_saccades):
    """
    Categorizes head saccades as 'free' or 'associated'.

    Args:
        head_saccades (list): List of head saccades.
        body_saccades (list): List of body saccades.

    Returns:
        list: List of tuples containing the category and time difference.
    """
    if not head_saccades or not body_saccades:
        return [('free', np.nan) for _ in head_saccades]

    head_df = pd.DataFrame(head_saccades)
    body_df = pd.DataFrame(body_saccades)

    sacc_type_timeDiff = categorise_saccade_type(head_df, body_df)
    return sacc_type_timeDiff

def collect_triggered_data_for_saccades(saccades, group, sa, window_length):
    """
    Collects triggered data for a list of saccades.

    Args:
        saccades (list): List of saccade dictionaries.
        group (pd.DataFrame): DataFrame group associated with the identifier.
        sa (SaccadeAnalysis): Instance of SaccadeAnalysis class.
        window_length (int): Length of the window used in saccade analysis.

    Returns:
        list: List of triggered saccade data.
    """
    trig_saccades = []
    for saccade in saccades:
        trig_data = collect_triggered_data_for_saccade(saccade, group, sa, window_length)
        trig_saccades.extend(trig_data)
    return trig_saccades

def identify_intersaccadic_intervals(head_saccades, group_length):
    """
    Identifies intersaccadic intervals based on saccade indices.

    Args:
        head_saccades (list): List of head saccade dictionaries.
        group_length (int): Length of the data group.

    Returns:
        list: List of intersaccadic intervals with start and stop indices.
    """
    if not head_saccades:
        # No saccades, entire duration is intersaccadic interval
        return [{'saccade_start_idx': 0, 'saccade_stop_idx': group_length - 1}]

    # Sort saccades by 'saccade_start_idx'
    head_saccades_sorted = sorted(head_saccades, key=lambda s: s['saccade_start_idx'])

    intersaccadic_intervals = []

    # First interval
    first_interval_start_idx = 0
    first_interval_end_idx = head_saccades_sorted[0]['saccade_start_idx']
    if first_interval_end_idx > first_interval_start_idx:
        intersaccadic_intervals.append({
            'saccade_start_idx': first_interval_start_idx,
            'saccade_stop_idx': first_interval_end_idx,
        })

    # Middle intervals
    for i in range(len(head_saccades_sorted) - 1):
        interval_start_idx = head_saccades_sorted[i]['saccade_stop_idx']
        interval_end_idx = head_saccades_sorted[i + 1]['saccade_start_idx']
        if interval_end_idx > interval_start_idx:
            intersaccadic_intervals.append({
                'saccade_start_idx': interval_start_idx,
                'saccade_stop_idx': interval_end_idx,
            })

    # Last interval
    last_interval_start_idx = head_saccades_sorted[-1]['saccade_stop_idx']
    last_interval_end_idx = group_length - 1
    if last_interval_end_idx > last_interval_start_idx:
        intersaccadic_intervals.append({
            'saccade_start_idx': last_interval_start_idx,
            'saccade_stop_idx': last_interval_end_idx,
        })

    return intersaccadic_intervals

def compute_interval_metrics(intersaccadic_intervals, group, frame_rate, name):
    """
    Computes metrics for intersaccadic intervals.

    Args:
        intersaccadic_intervals (list): List of intersaccadic intervals with indices.
        group (pd.DataFrame): DataFrame group associated with the identifier.
        frame_rate (float): Frame rate of the data.
        name (str): Identifier name.

    Returns:
        list: List of intersaccadic interval entries with metrics.
    """
    intersaccadic_df = []
    for interval in intersaccadic_intervals:
        start_idx = interval['saccade_start_idx']
        stop_idx = interval['saccade_stop_idx']
        duration_s = (stop_idx - start_idx) / frame_rate

        # Compute peak speed in the interval
        head_yaw_speed_interval = group['head_yaw_speed'].iloc[start_idx:stop_idx].to_numpy()
        if len(head_yaw_speed_interval) == 0:
            continue

        top_speed_degPs = np.max(np.abs(head_yaw_speed_interval))

        # Create intersaccadic entry with the same columns as saccades_df
        intersaccadic_entry = {
            'saccade_peak_s': (start_idx + stop_idx) / (2 * frame_rate),
            'saccade_start_idx': start_idx,
            'sacc_peak_idx': np.nan,  # No peak in intersaccadic interval
            'saccade_stop_idx': stop_idx,
            'amplitude_deg': np.nan,  # No amplitude
            'top_speed_degPs': top_speed_degPs,
            'saccade_duration_s': duration_s,
            'direction': np.nan,
            'id': name,
            'category': 'intersaccadic'
        }
        intersaccadic_df.append(intersaccadic_entry)
    return intersaccadic_df

def compute_mean_velocities(saccades_combined, combined_df, frame_rate):
    """
    Computes the mean absolute rotational velocity and mean translational velocity
    for each saccade and intersaccadic interval.

    Args:
        saccades_combined (pd.DataFrame): DataFrame containing saccade and intersaccadic data.
        combined_df (pd.DataFrame): Combined DataFrame of all data groups.
        frame_rate (float): Frame rate of the data.

    Returns:
        pd.DataFrame: Updated saccades_combined DataFrame with new columns.
    """
    # Create a mapping from 'Identifier' to their data groups
    id_to_group = {name: group for name, group in combined_df.groupby('Identifier')}

    mean_abs_rot_velocities = []
    mean_trans_velocities = []

    for idx, row in saccades_combined.iterrows():
        id_ = row['id']
        start_idx = int(row['saccade_start_idx'])
        stop_idx = int(row['saccade_stop_idx'])

        # Get the corresponding group DataFrame
        group = id_to_group[id_]

        # Ensure indices are within bounds
        start_idx = max(0, start_idx)
        stop_idx = min(len(group) - 1, stop_idx)

        # Extract head_yaw_speed and translational_velocity_mPs
        head_yaw_speed = group['head_yaw_speed'].iloc[start_idx:stop_idx + 1].to_numpy()
        translational_velocity = group['translational_velocity_mPs'].iloc[start_idx:stop_idx + 1].to_numpy()

        # Compute mean absolute rotational velocity
        mean_abs_rot_vel = np.median(np.abs(head_yaw_speed))
        mean_abs_rot_velocities.append(mean_abs_rot_vel)

        # Compute mean translational velocity
        mean_trans_vel = np.median(translational_velocity)
        mean_trans_velocities.append(mean_trans_vel)

    # Add new columns to saccades_combined DataFrame
    saccades_combined['median_abs_rot_vel_degPs'] = mean_abs_rot_velocities
    saccades_combined['median_translational_vel_mPs'] = mean_trans_velocities

    return saccades_combined


def process_identifier_group(name, group, sa, angle_vel_threshold, window_length, frame_rate):
    """
    Processes a group of data for a single identifier in the dataset.

    This function identifies saccades for both head and body, categorizes saccade types,
    collects triggered data for each saccade, and identifies intersaccadic intervals.

    Args:
        name (str): Identifier for the group of data.
        group (pd.DataFrame): DataFrame group associated with the identifier.
        sa (SaccadeAnalysis): Instance of SaccadeAnalysis class.
        angle_vel_threshold (float): Threshold for angular velocity to identify saccades.
        window_length (int): Length of the window used in saccade analysis.
        frame_rate (float): Frame rate of the data.

    Returns:
        dict: A dictionary containing lists - 'saccades', 'trig_saccades', 'body_saccades', and 'intersaccadic_intervals'.
    """
    # Step 1: Identify saccades
    head_saccades, body_saccades = identify_saccades(group, sa, angle_vel_threshold, window_length)

    saccades_accumulated = []
    trig_saccades = []
    intersaccadic_df = []

    if head_saccades:
        # Step 2: Categorize saccades
        sacc_type_timeDiff = categorize_saccades(head_saccades, body_saccades)

        # Assign categories and IDs to saccades
        for i, saccade in enumerate(head_saccades):
            saccade['id'] = name
            saccade['category'], _ = sacc_type_timeDiff[i]
            saccades_accumulated.append(saccade)

        # Step 3: Collect triggered data
        trig_saccades = collect_triggered_data_for_saccades(head_saccades, group, sa, window_length)

        # Step 4: Identify intersaccadic intervals
        intersaccadic_intervals = identify_intersaccadic_intervals(head_saccades, len(group))

        # Step 5: Compute interval metrics
        intersaccadic_df = compute_interval_metrics(intersaccadic_intervals, group, frame_rate, name)
    else:
        # No saccades, entire duration is intersaccadic interval
        intersaccadic_intervals = [{'saccade_start_idx': 0, 'saccade_stop_idx': len(group) - 1}]
        intersaccadic_df = compute_interval_metrics(intersaccadic_intervals, group, frame_rate, name)

    # Process body saccades
    body_saccade_df = pd.DataFrame(body_saccades)
    body_saccade_df['id'] = name
    body_saccade_df['category'] = 'body'

    return {
        'saccades': saccades_accumulated,
        'trig_saccades': trig_saccades,
        'body_saccades': body_saccade_df,
        'intersaccadic_intervals': intersaccadic_df
    }

def collect_triggered_data_for_saccade(saccade, group, sa, window_length):
    """
    Collects and organizes triggered data for a specific saccade from the provided data group.

    This function extracts the triggered data matrices for body and head angles and velocities,
    normalizes them, and creates DataFrames for each type.

    Args:
        saccade (dict): Dictionary containing details of a single saccade.
        group (pd.DataFrame): Grouped data related to the saccade.
        sa (SaccadeAnalysis): Instance of SaccadeAnalysis for data collection.
        window_length (int): Length of the window for collecting triggered data.

    Returns:
        list: List of DataFrames, each containing triggered data for a specific type.
    """
    trig_saccades = []
    
    b_angle_matrix = sa.collect_more_triggered_data([saccade], np.degrees(group['body_yaw_rad'].to_numpy()), window_length)
    b_velocity_matrix = sa.collect_more_triggered_data([saccade], group['body_yaw_speed'].to_numpy(), window_length)
    h_angle_matrix = sa.collect_more_triggered_data([saccade], np.degrees(group['head_yaw_rad'].to_numpy()), window_length)
    h_velocity_matrix = sa.collect_more_triggered_data([saccade], group['head_yaw_speed'].to_numpy(), window_length)

    # Select the correct matrices based on saccade direction
    b_angle_matrix = b_angle_matrix[0] if saccade['direction'] == 'right' else b_angle_matrix[1]
    b_velocity_matrix = b_velocity_matrix[0] if saccade['direction'] == 'right' else b_velocity_matrix[1]
    h_angle_matrix = h_angle_matrix[0] if saccade['direction'] == 'right' else h_angle_matrix[1]
    h_velocity_matrix = h_velocity_matrix[0] if saccade['direction'] == 'right' else h_velocity_matrix[1]

    # Normalize and create dataframes for each data type
    trig_saccades.append(create_saccade_dataframe(saccade['id'], saccade['direction'], saccade['category'], 'head', 'body_angle', sa.normalize_angle_data(b_angle_matrix[0])))
    trig_saccades.append(create_saccade_dataframe(saccade['id'], saccade['direction'], saccade['category'], 'head', 'body_velocity', b_velocity_matrix[0]))
    trig_saccades.append(create_saccade_dataframe(saccade['id'], saccade['direction'], saccade['category'], 'head', 'head_angle', sa.normalize_angle_data(h_angle_matrix[0])))
    trig_saccades.append(create_saccade_dataframe(saccade['id'], saccade['direction'], saccade['category'], 'head', 'head_velocity', h_velocity_matrix[0]))

    return trig_saccades

def analyze_grouped_data(grouped_df, sa, angle_vel_threshold, window_length,frame_rate):
    """
    Analyzes grouped data from a DataFrame, processing each group to identify saccades and 
    collect corresponding triggered data.

    This function iterates over each group in the DataFrame, processes the data for each 
    identifier, and accumulates saccade and triggered saccade data.

    Args:
        grouped_df (pd.DataFrameGroupBy): Grouped DataFrame to be analyzed.
        sa (SaccadeAnalysis): Instance of SaccadeAnalysis for identifying saccades.
        angle_vel_threshold (float): Threshold for angular velocity in saccade identification.
        window_length (int): Length of the window used in saccade analysis.

    Returns:
        tuple: A tuple containing two DataFrames - one for accumulated saccade data and 
               another for accumulated triggered saccade data.
    """
    saccades_accumulated = []
    trig_saccades = []
    body_saccades = []
    intersaccadic_intervals_accumulated = []


    for name, group in grouped_df:
        print(f"Processing Identifier: {name}")
        processed_data = process_identifier_group(name, group, sa, angle_vel_threshold, window_length,frame_rate)
        saccades_accumulated.extend(processed_data['saccades'])
        trig_saccades.extend(processed_data['trig_saccades'])
        body_saccades.append(processed_data['body_saccades'])
        intersaccadic_intervals_accumulated.extend(processed_data['intersaccadic_intervals'])


    all_trig_saccades_df = pd.concat(trig_saccades, ignore_index=True)
    saccades_df = pd.DataFrame(saccades_accumulated)
    intersaccadic_df = pd.DataFrame(intersaccadic_intervals_accumulated)
    return saccades_df, all_trig_saccades_df, body_saccades, intersaccadic_df

def categorise_saccade_type(head_saccades: pd.DataFrame, body_saccades: pd.DataFrame, time_threshold: float = 1.0) -> list:
    """
    Categorises each head saccade as 'free' or 'associated' based on its temporal proximity to a body saccade.

    This function iterates through each head saccade to identify the closest body saccade in time that shares the same direction. 
    If the closest body saccade occurs within a specified time threshold, the head saccade is categorised as 'associated'; otherwise, 
    it is categorised as 'free'.

    Args:
    head_saccades (pd.DataFrame): A DataFrame containing the head saccades with columns 'saccade_peak_s' and 'direction'.
    body_saccades (pd.DataFrame): A DataFrame containing the body saccades with columns 'saccade_peak_s' and 'direction'.
    time_threshold (float, optional): The time difference threshold to categorise a saccade as 'associated'. Default is 1.0 second.

    Returns:
    list: A list of tuples containing the categorised type ('free' or 'associated') and the time difference to the next body saccade for each head saccade.

    Example:
    >>> head_saccades = pd.DataFrame(...) # Your head saccade data here
    >>> body_saccades = pd.DataFrame(...) # Your body saccade data here
    >>> types = categorise_saccade_type(head_saccades, body_saccades)
    """
    
    saccade_type = []
    
    # Ensure the data is sorted by 'saccade_peak_s' if not already
    head_saccades = head_saccades.sort_values('saccade_peak_s')
    body_saccades = body_saccades.sort_values('saccade_peak_s')

    # Loop through each head saccade and find the closest body saccade in time
    for index, head_saccade in head_saccades.iterrows():
        turn_type = 'free'
        time_difference = np.nan
        head_time = head_saccade['saccade_peak_s']
        head_direction = head_saccade['direction']
        
        # Filter body saccades by matching direction
        body_saccades_dir = body_saccades[body_saccades['direction'] == head_direction]
        
        # Calculate the absolute time differences
        time_diffs = (body_saccades_dir['saccade_peak_s'] - head_time).abs()

        if not time_diffs.empty:
            # Find the closest body saccade in time
            closest_body_saccade_idx = time_diffs.idxmin()
            closest_body_saccade_time = body_saccades_dir.loc[closest_body_saccade_idx, 'saccade_peak_s']

            # Calculate the time difference
            time_difference = closest_body_saccade_time - head_time
            if abs(time_difference) < time_threshold:
                turn_type = 'associated'

        # Store the results
        saccade_type.append((turn_type,time_difference))
    
    return saccade_type

def flip_left_saccades(trig_average_df):
    """
    Flips the direction of left saccades to right in the provided DataFrame.

    This function modifies the DataFrame in-place. It negates the values of the time 
    columns for left saccades and changes their 'sacc_dir' value to 'right'.

    Args:
        trig_average_df (pd.DataFrame): DataFrame containing saccade data.

    Returns:
        pd.DataFrame: Modified DataFrame with left saccades flipped to right.
    """
    left_saccades = trig_average_df['sacc_dir'] == 'left'
    time_columns = [col for col in trig_average_df.columns if col.startswith('time_')]
    trig_average_df.loc[left_saccades, time_columns] = trig_average_df.loc[left_saccades, time_columns] * -1
    trig_average_df.loc[left_saccades, 'sacc_dir'] = 'right'
    return trig_average_df

def mean_confidence_interval(data, confidence=0.95):
    """
    Calculates the mean and the confidence interval for a given dataset.

    Args:
        data (array-like): Array-like object containing the dataset.
        confidence (float, optional): Confidence level for the interval. Defaults to 0.95.

    Returns:
        tuple: A tuple containing the mean, lower bound, and upper bound of the confidence interval.
    """
    n = len(data)
    mean = np.mean(data)
    sem = scipy.stats.sem(data)
    h = sem * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return mean, mean - h, mean + h

def calculate_mean_ci_for_all_saccades(trig_average_df):
    """
    Calculates the mean and confidence interval for each time column in the DataFrame.

    This function groups the data by 'data_type' and 'sacc_type', then calculates the mean 
    and confidence interval for each group's time columns.

    Args:
        trig_average_df (pd.DataFrame): DataFrame containing saccade data with time columns.

    Returns:
        pd.DataFrame: A new DataFrame with mean and confidence intervals for each time column.
    """
    time_columns = [col for col in trig_average_df.columns if col.startswith('time_')]
    # Group by data_type and sacc_type, and calculate mean and CI for each group
    grouped_data = trig_average_df.groupby(['data_type', 'sacc_type'])

    results = []
    for (data_type, sacc_type), group in grouped_data:
        for time_col in time_columns:
            mean, lower_ci, upper_ci = mean_confidence_interval(group[time_col].dropna())
            results.append({
                'source': data_type.split('_')[0],
                'data_type': data_type.split('_')[1],
                'sacc_type': sacc_type,
                'time_col': time_col,
                'mean': mean,
                '95%_CI_lower': lower_ci,
                '95%_CI_upper': upper_ci
            })

    return pd.DataFrame(results)

def plot_saccade_data(df):
    """
    Plots saccade data with subplots for different types and data sources.

    The function creates a 2x2 grid of subplots, with rows for 'associated' and 'free' saccades,
    and columns for 'angle' and 'velocity'. Each subplot includes data for both 'head' and 'body' sources.

    Args:
        df (pd.DataFrame): DataFrame containing the saccade data to plot.

    Returns:
        None: The function directly generates and shows a plot.
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Iterate over the subplot indices
    for i, sacc_type in enumerate(['associated', 'free']):
        for j, metadata in enumerate([('angle','angle, deg'), ('velocity','velocity, deg/s')]):
            ax = axs[i, j]
            data_type = metadata[0]
            y_label_str = metadata[1]
            # Filter the DataFrame for each subplot combination
            filtered_df = df[(df['sacc_type'] == sacc_type) & (df['data_type'] == data_type)]

            # Time axis
            time_axis = np.linspace(-500, 500, len(df.time_col.unique()))

            # Plotting data for both sources: 'head' and 'body'
            for source in ['head', 'body']:
                source_df = filtered_df[filtered_df['source'] == source]
                # Drop rows with NaN in 'mean' column
                source_df = source_df.dropna(subset=['mean'])
                means = source_df['mean'].values
                lower_ci = source_df['95%_CI_lower'].values
                upper_ci = source_df['95%_CI_upper'].values
                ax.plot(time_axis[:len(means)], means, label=f'{source}')
                ax.fill_between(time_axis[:len(means)], lower_ci, upper_ci, alpha=0.2)

            # Set labels and titles
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel(f'{data_type.capitalize()}')
            ax.set_title(f'{sacc_type.capitalize()} Saccades - {data_type.capitalize()}')
            ax.legend()

    plt.tight_layout()

def sacc_type_comparison_plots(df, data_col, category_col, category_order, log_flag=False):
    """
    Plots a boxplot for the given data column across different categories.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    data_col (str): The name of the column for data values.
    category_col (str): The name of the column for categories.
    category_order (list): The specific order of categories for the plot.
    log_flag (bool): Whether to use a logarithmic scale for the y-axis.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=category_col, y=data_col, data=df, order=category_order, notch=False)

    # Set the y-axis to logarithmic scale if required
    if log_flag:
        plt.yscale('log')

    # Additional customizations
    plt.title(f'Boxplot of {data_col} by {category_col}')
    plt.xlabel(category_col.capitalize())
    plt.ylabel(data_col.replace('_', ' ').capitalize())


# Define a function to compute statistics
def compute_stats(group, value_col):
    data = group[value_col]
    n = len(data)
    median = np.median(data)
    mean = np.mean(data)
    sem = scipy.stats.sem(data)
    if n > 1:
        ci_low, ci_high = scipy.stats.t.interval(0.95, n-1, loc=mean, scale=sem)
    else:
        ci_low, ci_high = mean - sem, mean + sem
    return pd.Series({
        'Median Velocity': median,
        'Lower 95% CI': ci_low,
        'Upper 95% CI': ci_high,
        'Mean Velocity': mean,
        'SEM': sem
    })

def make_statistic_df(saccades_combined):

    df_velocities = saccades_combined.copy()
    # Add 'Dataset' column
    df_velocities['Dataset'] = '2'  
    # Add 'Species' column
    df_velocities['Species'] = 'Gentoo'  # Assuming the species is Gentoo

    # Map 'category' to 'Movement Type'
    df_velocities['Movement Type'] = df_velocities['category']

    # Ensure necessary columns are present
    # For rotational velocity
    df_rotational = df_velocities[['Dataset', 'Species', 'Movement Type', 'Bodypart', 'median_abs_rot_vel_degPs']].copy()
    df_rotational.rename(columns={'median_abs_rot_vel_degPs': 'Rotational Velocity (deg/s)'}, inplace=True)

    # For translational velocity
    df_translational = df_velocities[['Dataset', 'Species', 'Movement Type', 'Bodypart', 'median_translational_vel_mPs']].copy()
    df_translational.rename(columns={'median_translational_vel_mPs': 'Translational Velocity (m/s)'}, inplace=True)
    # Group and compute statistics for rotational velocity
    group_cols = ['Dataset', 'Species', 'Movement Type', 'Bodypart']
    rotational_stats = df_rotational.groupby(group_cols).apply(lambda x: compute_stats(x.dropna(), 'Rotational Velocity (deg/s)')).reset_index()

    # Group and compute statistics for translational velocity
    translational_stats = df_translational.groupby(group_cols).apply(lambda x: compute_stats(x, 'Translational Velocity (m/s)')).reset_index()

    return rotational_stats, translational_stats
            
def main(target_folder, frame_rate=25, window_length=25, angle_vel_threshold=250):
    # Read data and group by 'Identifier'
    combined_df = read_cvs_into_dataframe(target_folder, frame_rate)
    grouped = combined_df.groupby('Identifier')

    # Initialize SaccadeAnalysis with the frame rate
    sa = SaccadeAnalysis.SaccadeAnalysis(frame_rate)

    # Perform saccade analysis and accumulate matrices
    saccades_df, trig_average_df, body_saccades, intersaccadic_df = analyze_grouped_data(
        grouped, sa, angle_vel_threshold, window_length, frame_rate)
    trig_average_df = flip_left_saccades(trig_average_df)
    mean_triggered_average = calculate_mean_ci_for_all_saccades(trig_average_df)
    
    # Combine saccades and intersaccadic intervals for comparison
    saccades_df['Bodypart'] = 'head'
    body_saccades =pd.concat(body_saccades, ignore_index=True)
    body_saccades['Bodypart'] = 'body'
    intersaccadic_df['Bodypart'] = 'head+body'

    saccades_combined = pd.concat([saccades_df, body_saccades,intersaccadic_df], ignore_index=True)
    saccades_combined['abs_speed_degPs'] = saccades_combined['top_speed_degPs'].abs()

    # Compute mean velocities
    saccades_combined = compute_mean_velocities(saccades_combined, combined_df, frame_rate)

    # Plotting including intersaccadic intervals
    sacc_type_comparison_plots(saccades_combined, 'saccade_duration_s', 'category', ['free', 'associated', 'body'], True)
    sacc_type_comparison_plots(saccades_combined, 'abs_speed_degPs', 'category', ['free', 'associated', 'body', 'intersaccadic'])

    # New plots for mean velocities
    sacc_type_comparison_plots(saccades_combined, 'median_abs_rot_vel_degPs', 'category', ['free', 'associated', 'body', 'intersaccadic'],True)
    sacc_type_comparison_plots(saccades_combined, 'median_translational_vel_mPs', 'category', ['free', 'associated', 'body', 'intersaccadic'], True)

    plot_saccade_data(mean_triggered_average)
    saccades_combined = saccades_combined.dropna(subset=['median_abs_rot_vel_degPs'])
    rotational_stats, translational_stats = make_statistic_df(saccades_combined)

    return saccades_combined, rotational_stats, translational_stats


if __name__ == "__main__":
    target_folder = '/path/to/Penguin_Rostock/pengu_head_movies'
    saccades_combined,rotational_stats, translational_stats = main(target_folder)
    saccades_combined.to_csv('/path/to/Penguin_Rostock/saccade_speeds_combined.csv')
    rotational_stats.to_csv('/path/to/Penguin_Rostock/rotational_velocity_statistics.csv', index=False)
    translational_stats.to_csv('/path/to/Penguin_Rostock/translational_velocity_statistics.csv', index=False)

    plt.show()


