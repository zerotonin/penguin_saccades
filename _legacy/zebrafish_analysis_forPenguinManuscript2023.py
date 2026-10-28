"""
Saccade Duration Analysis for Zebrafish Data.

This script was used for the 2023 Penguin Manuscript to analyze the saccade durations from our zebrafish data.
It uses the SaccadeAnalysis class to detect saccades, extract data in a window around each saccade,
and perform further analysis on these windows.

Authors: 
Date: 
"""

import pandas as pd 
import numpy as np
import SaccadeAnalysis
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Chapter 1: File Location and Initial Setup

root_dir = '/path/to/fishDataBase'  # Root directory containing the data
desired_file = 'trace_mm.csv'  # Specific file we are looking for in each subdirectory
matched_files = []  # List to store full paths of matched files

# Iterate through root directory to find files of interest
for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        # Only store files that are named 'trace_mm.csv' and are in a directory containing 'Unt'
        if filename == desired_file and 'Unt' in dirpath:
            matched_files.append(os.path.join(dirpath, filename))

# Chapter 2: Saccade Analysis

saccade_threshold =  500  # Threshold for peak angular velocity to define a saccade
window_size = 100  # Size of the window around each saccade peak for data extraction

# Lists to store saccade analysis results
saccades_accumulated = []
pos_angle_matrix_accumulated = []
pos_velocity_matrix_accumulated = []
neg_angle_matrix_accumulated = []
neg_velocity_matrix_accumulated = []

sa = SaccadeAnalysis.SaccadeAnalysis(0)  # Initialize SaccadeAnalysis object

# Perform saccade analysis for each matched file
for filename in tqdm(matched_files, desc='analyse saccades'):
    trace = pd.read_csv(filename)
    sa.frame_rate = len(trace)/30.0 # all traces are thirty seconds long
    saccades, pos_angle_matrix, pos_velocity_matrix, neg_angle_matrix, neg_velocity_matrix = sa.main(trace.yaw_rad, saccade_threshold, window_size)
    
    # Save output
    saccades_accumulated += saccades
    pos_angle_matrix_accumulated.append(pos_angle_matrix)
    pos_velocity_matrix_accumulated.append(pos_velocity_matrix)
    neg_angle_matrix_accumulated.append(neg_angle_matrix)
    neg_velocity_matrix_accumulated.append(neg_velocity_matrix)

# Chapter 3: Results Visualization

# Create a dataframe from the list of saccades for easier visualization
saccades_df = pd.DataFrame(saccades_accumulated)

# Plot the distribution of saccade durations
sns.displot(saccades_df, x="saccade_duration_s", stat="probability", bins= np.linspace(0.1,1.0,10))
plt.show()
