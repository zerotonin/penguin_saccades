import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from peguin_ana01 import TrajectoryProcessor
import glob
import os
import json

# Constants
pix2m = np.array([0.97/124.6, 0.3/66, 0.4/76.2]).mean()
im_height = 788
im_width = 1402
frame_rate = 30
path_to_penguins = "/path/to/Penguin_Rostock/penguins/sorted_and_filtered/"
file_extension = 'h5'

def load_files(path, extension):
    return glob.glob(os.path.join(path, f'**/*{extension}'), recursive=True)

def process_file(file_path, im_width, im_height, pix2m, frame_rate):
    df = pd.read_hdf(file_path)
    tp = TrajectoryProcessor(df, im_width, im_height, pix2m, frame_rate)
    a,b = tp.main()
    return a,b,tp

def categorize_data(file_path, df_saccades, angle, angle_vel, inter_saccs, saccade_list, angle_list, angle_vel_list, inter_sacc_list):
    if 'Gentoo' in file_path:
        saccade_list['gentoo'].append(df_saccades)
        angle_list['gentoo'].append(angle)
        angle_vel_list['gentoo'].append(angle_vel)
        inter_sacc_list['gentoo'].append(inter_saccs)
    elif 'Rockhopper' in file_path:
        saccade_list['rockhopper'].append(df_saccades)
        angle_list['rockhopper'].append(angle)
        angle_vel_list['rockhopper'].append(angle_vel)
        inter_sacc_list['rockhopper'].append(inter_saccs)

    saccade_list['pooled'].append(df_saccades)
    angle_list['pooled'].append(angle)
    angle_vel_list['pooled'].append(angle_vel)
    inter_sacc_list['pooled'].append(inter_saccs)

def combine_data(saccade_list, angle_list, angle_vel_list):
    for key in saccade_list:
        angle_list[key] = np.vstack(angle_list[key])
        angle_vel_list[key] = np.vstack(angle_vel_list[key])
        saccade_list[key] = pd.concat(saccade_list[key])

def plot_data(saccade_list, angle_list, angle_vel_list, frame_rate):
    fig, axes = plt.subplots(4, 1, figsize=(10, 10))
    colors = {'gentoo': 'red', 'rockhopper': 'blue', 'pooled': 'purple'}
    saccade_attrib = dict()

    for key in saccade_list:
        angle_mean = np.nanmean(angle_list[key], axis=0)
        angle_sem = stats.sem(angle_list[key], axis=0, nan_policy='omit')
        angle_vel_mean = np.nanmean(angle_vel_list[key], axis=0)
        angle_vel_sem = stats.sem(angle_vel_list[key], axis=0, nan_policy='omit')

        saccade_attrib[f'{key}_max_amplitude_angle'] = np.nanmax(angle_list[key])
        saccade_attrib[f'{key}_mean_amplitude_angle'] = np.nanmax(angle_mean)
        indices = np.where(angle_mean == saccade_attrib[f'{key}_mean_amplitude_angle'])
        saccade_attrib[f'{key}_sem_amplitude_angle'] = np.nanmax(angle_sem[indices])

        saccade_attrib[f'{key}_max_amplitude_angle_vel'] = np.nanmax(angle_vel_list[key])
        saccade_attrib[f'{key}_mean_amplitude_angle_vel'] = np.nanmax(angle_vel_mean)
        indices = np.where(angle_vel_mean == saccade_attrib[f'{key}_mean_amplitude_angle_vel'])
        saccade_attrib[f'{key}_sem_amplitude_angle_vel'] = np.nanmax(angle_vel_sem[indices])

        t = np.arange(len(angle_mean)) * (1000.0 / frame_rate)

        axes[0].plot(t, angle_mean, color=colors[key], label=key)
        axes[0].fill_between(t, angle_mean - angle_sem, angle_mean + angle_sem, color=colors[key], alpha=0.2)

        axes[1].plot(t, angle_vel_mean, color=colors[key], label=key)
        axes[1].fill_between(t, angle_vel_mean - angle_vel_sem, angle_vel_mean + angle_vel_sem, color=colors[key], alpha=0.2)

    bins = np.linspace(0, max(max(saccade_list[key].saccade_duration_s * 1000) for key in saccade_list), 10)
    ax2 = axes[2]

    for idx, key in enumerate(saccade_list):
        hist, _ = np.histogram(saccade_list[key].saccade_duration_s * 1000, bins=bins, density=True)
        hist = hist / sum(hist)
        ax2.bar(bins[:-1] + idx * 20, hist, width=20, color=colors[key], label=key, alpha=1)

    bins = np.linspace(0, max(max(saccade_list[key].amplitude_deg) for key in saccade_list), 10)
    ax3 = axes[3]

    for idx, key in enumerate(saccade_list):
        hist, _ = np.histogram(saccade_list[key].amplitude_deg, bins=bins, density=True)
        hist = hist / sum(hist)
        ax3.bar(bins[:-1] + idx * 0.8, hist, width=0.8, color=colors[key], label=key, alpha=1)

    for ax in axes:
        ax.legend()

    axes[0].set_title('Mean Angle')
    axes[0].set_ylabel('Yaw, deg')
    axes[1].set_title('Mean Angle Velocity')
    axes[1].set_ylabel('Yaw velocity, deg*s^-1')
    axes[1].set_xlabel('Time (ms)')
    axes[2].set_title('Saccade Duration Histogram')
    axes[2].set_ylabel('Probability Density')
    axes[2].set_xlabel('Duration, ms')
    axes[3].set_title('Amplitude Histogram')
    axes[3].set_ylabel('Probability Density')
    axes[3].set_xlabel('Amplitude, deg')

    plt.tight_layout()
    plt.show()

    return saccade_attrib

def save_attributes(saccade_attrib, file_path):
    with open(file_path, 'w') as file:
        json.dump(saccade_attrib, file, indent=4)

def main():
    file_list = load_files(path_to_penguins, file_extension)
    saccade_list = {'gentoo': [], 'rockhopper': [], 'pooled': []}
    angle_list = {'gentoo': [], 'rockhopper': [], 'pooled': []}
    angle_vel_list = {'gentoo': [], 'rockhopper': [], 'pooled': []}
    inter_sacc_list = {'gentoo': [], 'rockhopper': [], 'pooled': []}
    c = 0
    d = 0

    for file_path in file_list:
        df_interp, df_saccades,tp = process_file(file_path, im_width, im_height, pix2m, frame_rate)

        if df_saccades is not None:
            angle, angle_vel = tp.extract_saccades(df_interp, df_saccades, 5)
            inter_saccs = np.vstack((df_saccades.saccade_start_idx[1::].to_numpy(),df_saccades.saccade_stop_idx[0:-1].to_numpy())).T
            inter_saccs =inter_saccs[:,0]-inter_saccs[:,1]
            inter_saccs =inter_saccs[inter_saccs>0] / frame_rate 
            categorize_data(file_path, df_saccades, angle, angle_vel,inter_saccs, saccade_list, angle_list, angle_vel_list, inter_sacc_list)
            d += 1
        c += 1
        print(c, d, angle.shape)

    combine_data(saccade_list, angle_list, angle_vel_list)
    saccade_attrib = plot_data(saccade_list, angle_list, angle_vel_list, frame_rate)
    inter_saccades_gentoo = np.hstack(inter_sacc_list['gentoo'])
    inter_saccades_rock = np.hstack(inter_sacc_list['rockhopper'])
    species = ['gentoo' for i in inter_saccades_gentoo] + ['rockhopper' for i in inter_saccades_rock ]

    df_inter_saccs = pd.DataFrame() 
    df_inter_saccs['species'] =species
    df_inter_saccs['inter_sacc_dur_sec'] = np.hstack((inter_saccades_gentoo,inter_saccades_rock))
    df_inter_saccs.to_csv('/path/to/Penguin_Rostock/penguins/body_inter_saccade_dur.csv',index=False)




    saccade_dur_gentoo = saccade_list['gentoo'].saccade_duration_s.to_numpy()
    saccade_dur_rock = saccade_list['rockhopper'].saccade_duration_s.to_numpy()
    species = ['gentoo' for i in saccade_dur_gentoo] + ['rockhopper' for i in saccade_dur_rock ]

    df_saccdur = pd.DataFrame() 
    df_saccdur['species'] =species
    df_saccdur['sacc_dur_sec'] = np.hstack((saccade_dur_gentoo,saccade_dur_rock))
    df_saccdur.to_csv('/path/to/Penguin_Rostock/penguins/body_saccade_dur.csv',index=False)
    save_attributes(saccade_attrib, "/path/to/Penguin_Rostock/saccade_attributes.txt")

if __name__ == "__main__":
    main()
