"""
Script for reanalyzing the saccadic behavior of zebra finches, based on the data set of Eckmeier et al. 2008.
This analysis is part of the 'Penguin Manuscript 2023'.

The script works as follows:
1. It collects .tra files from a given directory.
2. It then reads these files and processes the trajectories, calculating the body axis and yaw.
3. After applying a Gaussian filter to the yaw, it uses the SaccadeAnalysis module to identify saccades.
4. The identified saccades and other related data are then stored.
5. The script plots a histogram of saccade durations and triggered averages of the angles and velocities.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import SaccadeAnalysis
from tqdm import tqdm
import pandas as pd
import seaborn as sns

def apply_gaussian_filter(data, sigma=3):
    """
    Applies a Gaussian filter to a 1D numpy array.

    Parameters:
    data (np.array): The input 1D numpy array to be filtered.
    sigma (float): The standard deviation for Gaussian kernel. The standard deviations of the 
                   Gaussian filter are given for each axis as a sequence, or as a single number, 
                   in which case it is equal for all axes.

    Returns:
    filtered_data (np.array): The filtered 1D numpy array.
    """
    return gaussian_filter1d(data, sigma)


def collect_tra_files(root_dir):
    """
    Collects the file paths of all .tra files that are located in 'oben' subdirectory of the provided root directory.

    Parameters:
    root_dir (str): The root directory from where to start the search.

    Returns:
    tra_files (list): A list of file paths for all .tra files in 'oben' subdirectories.
    """
    desired_file_extension = '.tra'
    subdirectory_name = 'oben'
    tra_files = []

    # Iterate over the root directory and its subdirectories
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Check if the current subdirectory is 'oben'
        if os.path.split(dirpath)[-1] == subdirectory_name:
            # If yes, iterate over its files
            for filename in filenames:
                # Check if the current file has the .tra extension
                if filename.endswith(desired_file_extension):
                    # If yes, add its path to the list
                    tra_files.append(os.path.join(dirpath, filename))

    return tra_files
time_ax= np.linspace(0,100,50)

def plot_triggered_avg(mean,error,ylabel):
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

def load_tra_file(filepath):
    """
    Loads .tra file into a numpy matrix of shape mx5xp.

    Parameters:
    filepath (str): Path of the .tra file to load.

    Returns:
    data_matrix (np.array): A numpy array of shape mx5xp, where m is the number of frames,
                            5 is the number of data points (x, y, angle, size, eccentricity),
                            and p is the number of regions.
    """
    # Load the file as a numpy array
    data = np.loadtxt(filepath)

    # Calculate the number of frames and regions
    num_frames = data.shape[0]
    num_regions = data.shape[1] // 5

    # Reshape the data into the desired format (mx5xp)
    data_matrix = data[:, 1:].reshape(num_frames, num_regions, 5)
    data_matrix = np.swapaxes(data_matrix, 1, 2)

    return data_matrix


# Define the root directory where .tra files are stored
root_dir = '/path/to/zebraFinch_Eckmeier_Data/Rohdaten -Trajektorien/wildtypen/'

# Collect .tra files from the directory
tra_files = collect_tra_files(root_dir)

# Initialize SaccadeAnalysis with a threshold of 500
sa = SaccadeAnalysis.SaccadeAnalysis(500)

# Lists to store saccade analysis results
saccades_accumulated = []
pos_angle_matrix_accumulated = []
pos_velocity_matrix_accumulated = []
neg_angle_matrix_accumulated = []
neg_velocity_matrix_accumulated = []

# Analyze each .tra file
for file in tqdm(tra_files,desc='read trajectories'):
    # Load the data from the .tra file into a numpy matrix
    data_matrix = load_tra_file(file)
    
    # Compute the body axis and yaw
    body_axis = np.diff(data_matrix,axis=2)[:,:2,0]
    yaw = np.unwrap(np.arctan2(body_axis[:,1],body_axis[:,0]))
    
    # Apply Gaussian filter to the yaw
    yaw = apply_gaussian_filter(yaw)
    
    # Identify saccades and other related parameters using the SaccadeAnalysis object
    saccades, pos_angle_matrix, pos_velocity_matrix, neg_angle_matrix, neg_velocity_matrix = sa.main(yaw,600,50,False)
    
    # Save output
    saccades_accumulated += saccades
    pos_angle_matrix_accumulated.append(pos_angle_matrix)
    pos_velocity_matrix_accumulated.append(pos_velocity_matrix)
    neg_angle_matrix_accumulated.append(neg_angle_matrix)
    neg_velocity_matrix_accumulated.append(neg_velocity_matrix)

# Create a dataframe from the list of saccades for easier visualization
saccades_df = pd.DataFrame(saccades_accumulated)

# Convert saccade duration to milliseconds
saccades_df['saccade_duration_ms'] = saccades_df.saccade_duration_s*1000


# Concatenate matrices for plotting
ang = np.concatenate((np.concatenate(pos_angle_matrix_accumulated),np.concatenate(neg_angle_matrix_accumulated)*-1))
vel = np.concatenate((np.concatenate(pos_velocity_matrix_accumulated),np.concatenate(neg_velocity_matrix_accumulated)*-1))

# Calculate mean and SEM for plotting
mean_ang = np.nanmean(ang, axis=0)
sem_ang = np.nanstd(ang, axis=0) / np.sqrt(ang.shape[0])

mean_vel = np.nanmean(vel, axis=0)
sem_vel = np.nanstd(vel, axis=0) / np.sqrt(vel.shape[0])

# Plot the average angle and velocity with SEM as a transparent area around the line
plot_triggered_avg(mean_ang, sem_ang, 'angle, deg')
plot_triggered_avg(mean_vel, sem_vel, 'velocity, deg/s')
# Plot the distribution of saccade durations
sns.displot(saccades_df, x="saccade_duration_ms", stat="probability", bins= np.linspace(10,100,10))

plt.show()