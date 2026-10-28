import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import SaccadeAnalysis
import seaborn as sns

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

def plot_angles_over_time(df):
    """Plots the head and body yaw angles over time, indicating starts and stops of the identifier.

    Args:
        df (pd.DataFrame): DataFrame containing the angles and identifiers.

    Returns:
        None: Shows the plot.
    """
    
    plt.figure(figsize=(14, 8))
    
    # Plotting angles
    plt.subplot(2, 1, 1)
    plt.title("Head and Body Yaw Angles over Time")
    plt.plot(df['head_yaw_speed'], label='Head Yaw Angle (radians)')
    plt.plot(df['body_yaw_speed'], label='Body Yaw Angle (radians)')
    plt.xlabel("Time")
    plt.ylabel("Angle (radians)")
    plt.legend()

    # Indicating starts and stops of the identifier
    unique_identifiers = df['Identifier'].unique()
    for identifier in unique_identifiers:
        start_idx = df[df['Identifier'] == identifier].index[0]
        end_idx = df[df['Identifier'] == identifier].index[-1]
        
        plt.axvline(x=start_idx, color='r', linestyle='--', alpha=0.6)
        plt.text(start_idx, 0, f"Start {identifier}", rotation=90)
        
        plt.axvline(x=end_idx, color='g', linestyle='--', alpha=0.6)
        plt.text(end_idx, 0, f"End {identifier}", rotation=90)

    plt.tight_layout()
    plt.show()




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

def read_csvs_into_dataframe(target_folder,frame_rate=25):
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

def plot_triggered_avg(mean,error,ylabel,time_ax):
    """
    This function creates a line plot of the average (triggered) data over time. 
    It also includes a shaded region indicating the standard error of the mean (SEM) around the line.
    
    Parameters
    ----------
    mean : array-like
        1D array-like object (list, numpy array, etc.) containing the mean data points over time.
        
    error : array-like
        1D array-like object containing the standard error of the mean (SEM) for each data point over time.

    ylabel : str
        The label to be set for the y-axis of the plot.

    Returns
    -------
    None. 
    The function directly generates a plot which displays the mean data with its SEM over time.
    
    Example
    -------
    >>> plot_triggered_avg(mean_data, sem_data, 'Average Value')
    """
    # Create the line plot
    plt.figure(figsize=(10,6))
    plt.plot(time_ax, mean)

    # Add transparent error band
    plt.fill_between(time_ax, mean - error, mean + error, color='blue', alpha=0.2)
    plt.xlabel('time, ms')
    plt.ylabel(ylabel)

def plot_multi_triggered_avg(mean_list, error_list, ylabel, time_ax, labels):
    """
    This function creates multiple line plots of the average (triggered) data over time. 
    It also includes shaded regions indicating the standard error of the mean (SEM) around each line.
    
    Parameters
    ----------
    mean_list : list of array-like
        List of 1D array-like objects (lists, numpy arrays, etc.) containing the mean data points over time for each condition.
        
    error_list : list of array-like
        List of 1D array-like objects containing the standard error of the mean (SEM) for each data point over time for each condition.

    ylabel : str
        The label to be set for the y-axis of the plot.
    
    time_ax : array-like
        1D array-like object indicating the time axis.

    labels : list of str
        List of strings for the legend, corresponding to each line.

    Returns
    -------
    None. 
    The function directly generates a plot which displays the mean data with its SEM over time for multiple conditions.
    """
    plt.figure(figsize=(10, 6))

    for mean, error, label in zip(mean_list, error_list, labels):
        plt.plot(time_ax, mean, label=label)
        plt.fill_between(time_ax, mean - error, mean + error, alpha=0.2)

    plt.xlabel('Time (ms)')
    plt.ylabel(ylabel)
    plt.legend()

def append_if_non_empty(accumulated_list, array_to_append):
    if np.size(array_to_append) > 0:
        accumulated_list.append(array_to_append)

def accumulate_non_empty_matrices(accumulated_lists, arrays_to_append):
    """Appends non-empty numpy arrays to their corresponding accumulation lists.

    Parameters:
        accumulated_lists (list): List of lists where arrays are accumulated.
        arrays_to_append (list): List of numpy arrays to append to the accumulation lists.

    Returns:
        None: The function modifies the accumulated_lists in-place.
    """
    for acc_list, array in zip(accumulated_lists, arrays_to_append):
        append_if_non_empty(acc_list, array)


def normalize_angle_matrices(angle_matrices):
    """Applies angle normalization on each row of every matrix in a list.

    Parameters:
        angle_matrices (list): List of numpy matrices to be normalized.

    Returns:
        list: List of normalized numpy matrices.
    """
    return [np.apply_along_axis(sa.normalize_angle_data, 1, matrix) for matrix in angle_matrices]


def analyze_grouped_data(grouped_df, sa, angle_vel_threshold, window_length ,mode = 'head'):
    """Analyzes a DataFrame grouped by 'Identifier', accumulating saccade-related matrices.

    Parameters:
        grouped_df (DataFrameGroupBy): DataFrame grouped by 'Identifier'.
        sa (SaccadeAnalysis): An instance of the SaccadeAnalysis class.
        angle_vel_threshold (float): Threshold for angular velocity.
        window_length (int): The window length for the main analysis function.

    Returns:
        list: Lists containing accumulated saccade data and related matrices.
    """
    # Initialize lists to store saccade analysis results
    saccades_accumulated = []
    accumulated_lists = [[] for _ in range(8)]

    if mode == 'head':
        primary =  'head_yaw_rad'
        secondaries = ('body_yaw_rad','body_yaw_speed')
    elif mode == 'body':
        primary =  'body_yaw_rad'
        secondaries = ('head_yaw_rad','head_yaw_speed')
    else:
        raise ValueError(f'analyze_grouped_data: Unknown mode {mode}' )


    for name, group in grouped_df:
        print(f"Processing Identifier: {name}")
        
        # Identify saccades and other related parameters
        saccades, pos_angle_matrix, pos_velocity_matrix, neg_angle_matrix, neg_velocity_matrix = sa.main(group[primary].to_numpy(), angle_vel_threshold, window_length, False)
        saccades =  [{**d, 'id': name} for d in saccades]


        # Save output
        saccades_accumulated += saccades

        # Collect more triggered data for body angles and velocities
        pos_body_angle_matrix, neg_body_angle_matrix = sa.collect_more_triggered_data(saccades, np.degrees(group[secondaries[0]].to_numpy()), window_length)
        pos_body_velocity_matrix, neg_body_velocity_matrix = sa.collect_more_triggered_data(saccades, group[secondaries[1]].to_numpy(), window_length)
        # Normalise body angles
        if pos_body_angle_matrix.size>0:
            pos_body_angle_matrix =  np.apply_along_axis(sa.normalize_angle_data, 1, pos_body_angle_matrix)
        if neg_body_angle_matrix.size>0:
            neg_body_angle_matrix =  np.apply_along_axis(sa.normalize_angle_data, 1, neg_body_angle_matrix)

        # Append non-empty arrays
        matrices_to_append = [
            pos_angle_matrix,
            pos_velocity_matrix,
            neg_angle_matrix,
            neg_velocity_matrix,
            pos_body_angle_matrix,
            pos_body_velocity_matrix,
            neg_body_angle_matrix,
            neg_body_velocity_matrix
        ]
        
        accumulate_non_empty_matrices(accumulated_lists, matrices_to_append)

    return saccades_accumulated, accumulated_lists


def calculate_mean_and_sem(data_matrix):
    """Calculates the mean and standard error of the mean (SEM) along axis 0 for a given 2D array.

    Parameters:
        data_matrix (numpy.ndarray): 2D numpy array where each row is a sample.

    Returns:
        tuple: A tuple containing the mean and SEM arrays.
    """
    mean_data = np.nanmean(data_matrix, axis=0)
    sem_data = np.nanstd(data_matrix, axis=0) / np.sqrt(data_matrix.shape[0])
    return mean_data, sem_data

def concatenate_and_invert_matrices(pos_accumulated, neg_accumulated):
    """Concatenates positive and negative matrices, inverting the latter.

    Parameters:
        pos_accumulated (list of numpy.ndarray): List of 2D arrays with positive values.
        neg_accumulated (list of numpy.ndarray): List of 2D arrays with negative values.

    Returns:
        numpy.ndarray: A 2D array containing the concatenated and inverted matrices.
    """
    pos_concatenated = np.concatenate(pos_accumulated)
    neg_concatenated = np.concatenate(neg_accumulated) * -1
    return np.concatenate((pos_concatenated, neg_concatenated))

def save_last_three_figures(path, suffix):
    """
    Saves the last three Matplotlib figures to the specified path with the given suffix in SVG and PNG formats.

    Parameters:
    path (str): The directory where the figures will be saved.
    suffix (str): A suffix to append to the filenames of the saved figures.
    """
    # Define the extensions and descriptors for each figure
    extensions = ['svg', 'png']
    descriptors = ['saccade_angle_trigAve_', 'saccade_velocity_trigAve_', 'saccade_durationHist_']
    
    # Get the last three figure numbers
    last_three_fig_nums = plt.get_fignums()[-3:]
    
    # Loop over each figure number and descriptor
    for fig_num, descriptor in zip(last_three_fig_nums, descriptors):
        
        # Get the figure by its number
        fig = plt.figure(fig_num)
        
        # Loop over each extension to save the figure
        for ext in extensions:
            filename = os.path.join(path, f"{descriptor}{suffix}.{ext}")
            fig.savefig(filename)
            
def main(target_folder, mode, frame_rate=25, window_length=25, angle_vel_threshold=250):
    """Main function to execute the data analysis pipeline.

    Parameters:
        target_folder (str): Path to the folder containing the data files.
        frame_rate (int): Frame rate of the data, default is 25.
        window_length (int): The window length for the main analysis function, default is 25.
        angle_vel_threshold (float): Threshold for angular velocity, default is 250.

    Returns:
        None: Outputs are saved or plotted.
    """
    # Read data and group by 'Identifier'
    combined_df = read_csvs_into_dataframe(target_folder, frame_rate)
    grouped = combined_df.groupby('Identifier')

    # Initialize SaccadeAnalysis with a frame rate of 25
    sa = SaccadeAnalysis.SaccadeAnalysis(frame_rate)

    # Perform saccade analysis and accumulate matrices
    saccades_accumulated, accumulated_lists = analyze_grouped_data(grouped, sa, angle_vel_threshold, window_length,mode)

    # Concatenate and invert matrices for plotting
    pos_angle_matrix_accumulated, pos_velocity_matrix_accumulated, neg_angle_matrix_accumulated, neg_velocity_matrix_accumulated, \
    pos_body_angle_matrix_accumulated, pos_body_velocity_matrix_accumulated, neg_body_angle_matrix_accumulated, neg_body_velocity_matrix_accumulated = accumulated_lists

    concatenated_ang = concatenate_and_invert_matrices(pos_angle_matrix_accumulated, neg_angle_matrix_accumulated)
    concatenated_vel = concatenate_and_invert_matrices(pos_velocity_matrix_accumulated, neg_velocity_matrix_accumulated)
    concatenated_ang_body = concatenate_and_invert_matrices(pos_body_angle_matrix_accumulated, neg_body_angle_matrix_accumulated)
    concatenated_vel_body = concatenate_and_invert_matrices(pos_body_velocity_matrix_accumulated, neg_body_velocity_matrix_accumulated)

    # Calculate mean and SEM
    mean_ang, sem_ang = calculate_mean_and_sem(concatenated_ang)
    mean_vel, sem_vel = calculate_mean_and_sem(concatenated_vel)
    mean_ang_body, sem_ang_body = calculate_mean_and_sem(concatenated_ang_body)
    mean_vel_body, sem_vel_body = calculate_mean_and_sem(concatenated_vel_body)

    # Here, you can add plotting or other forms of output
    if mode is 'head':
        legend_list = ['head','body']
    else:
        legend_list = ['body', 'head']

    time_ax = np.linspace(-500, 500, window_length)
    plot_multi_triggered_avg([mean_ang, mean_ang_body], [sem_ang, sem_ang_body], 'angle, deg', time_ax, legend_list)
    plot_multi_triggered_avg([mean_vel, mean_vel_body], [sem_vel, sem_vel_body], 'velocity, deg/s', time_ax, legend_list)

    # Create a dataframe from the list of saccades for easier visualization
    saccades_df = pd.DataFrame(saccades_accumulated)
    # Convert saccade duration to milliseconds
    saccades_df['saccade_duration_ms'] = saccades_df.saccade_duration_s*1000
    # Plot histogram
    g = sns.displot(saccades_df, x="saccade_duration_ms", stat="probability", bins= np.linspace(10, 100, 10))
    g.fig.suptitle(f'{legend_list[0]} saccade durations (n = {len(saccades_df)})')
    save_last_three_figures(f'{target_folder}/figures/', legend_list[0])
    saccades_df.to_csv(f'{target_folder}/saccade_data/saccades_{mode}.csv', index=False)


if __name__ == "__main__":
    target_folder = '/path/to/Penguin_Rostock/pengu_head_movies/'
    main(target_folder,'head')
    main(target_folder,'body')
    plt.show()
