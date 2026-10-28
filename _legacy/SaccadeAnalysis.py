from scipy.signal import find_peaks, peak_widths
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class SaccadeAnalysis: 
    """
    SaccadeAnalyser is a class for analyzing saccadic  movements.

    The class computes angular velocities from a given list of angles, then identifies saccades based on these velocities.
    It separates positive and negative saccades, normalizes the saccades, and transforms the data into matrices for further analysis.
    The class provides options for plotting the angular velocities with marks indicating saccade start, peak, and stop points.

    Parameters
    ----------
    frame_rate : int or float
        The frame rate of the data recording in Hz.
        
    max_duration_sec : float, optional
        Maximum allowed duration for a biologically relevant saccade in seconds. Default is 5 seconds.
        
    max_speed_degPsec : float, optional
        Maximum allowed amplitude for a biologically relevant saccade in degrees per second. Default is 10000 deg/sec.
        
    min_amplitude_deg : float, optional
        Minimum required amplitude for a biologically relevant saccade in degrees. Default is 10 degrees.

    Attributes
    ----------
    max_duration_sec : float
        Maximum allowed duration for a saccade.

    max_speed_degPsec : float
        Maximum allowed speed for a saccade.

    min_amplitude_deg : float
        Minimum required amplitude for a saccade.

    frame_rate : int or float
        The frame rate of the data recording.
    
    Methods
    -------
    compute_angular_velocity(self, angles)
        Computes angular velocity from the provided list of angles.
        
    find_saccades(self, angles, angular_velocity, threshold)
        Identifies saccades from the angles and angular velocities.

    filter_artifacts(self, saccades)
        Filters out saccades that are not biologically relevant based on specified duration and amplitude thresholds.

    get_saccade_window(self, angle_data, velocity_data, saccade, window_size)
        Extracts saccade data using an adjusted window size if the full window cannot be used.
        
    normalize_angle_data(self, angle_data)
        Normalizes the angle data in a saccade window.

    get_saccade_triggered_average(self, angles, velocities, window_size)
        Returns two matrices of angles and angular velocities for positive and negative saccades, respectively.
        
    plot_saccades(self, angles, angular_velocity, saccades)
        Plots the angular velocities with marks indicating saccade start, peak, and stop points.
        
    main(self, angles, threshold)
        Main function that takes a list of angles and a threshold, and outputs saccades and four matrices.
    
    """
    def __init__(self, frame_rate, max_duration_sec=5, max_speed_degPsec=10000, min_amplitude_deg=10):
        self.max_duration_sec = max_duration_sec
        self.max_speed_degPsec = max_speed_degPsec
        self.min_amplitude_deg = min_amplitude_deg
        self.frame_rate = frame_rate
    
    def continuous_angle(self, angles):
        """
        Convert an array of angles to continuous angles by removing discontinuities caused by wrapping around 2*pi.

        This function takes an array of angles (in radians) and returns a new array where the angles are continuous,
        meaning that if the input angles jump from a value close to 2*pi to a value close to 0, the output angles
        will continue to increase beyond 2*pi, maintaining the continuity.

        Args:
            angles (np.ndarray): A 1D numpy array containing angles in radians.

        Returns:
            np.ndarray: A 1D numpy array containing the continuous angles.

        Example:
            >>> angles = np.array([0, 1, 2, 3, 0, 1, 2, 3])
            >>> continuous_angles = continuous_angle(angles)
            >>> continuous_angles
            array([0., 1., 2., 3., 4., 5., 6., 7.])
        """
        continuous_angles = np.zeros_like(angles)
        continuous_angles[0] = angles[0]
        for i in range(1, len(angles)):
            diff = angles[i] - angles[i - 1]
            if diff > np.pi:
                diff -= 2 * np.pi
            elif diff < -np.pi:
                diff += 2 * np.pi
            continuous_angles[i] = continuous_angles[i - 1] + diff
        return continuous_angles
    

    def find_saccades(self, angles, angular_velocity, threshold):
        """
        Finds the peaks and valleys in the signal with a given prominence threshold.

        Parameters
        ----------
        angles : list
            List of angles.
        angular_velocity : list
            List of angular velocities.
        threshold : float
            The prominence threshold for detecting peaks.

        Returns
        -------
        saccades : list
            A list of dictionaries containing the saccade times, start and stop times, and amplitudes.
        """
        saccades = []

        # Find peaks (local maxima) and valleys (local minima)
        peak_indices, peak_data = find_peaks(angular_velocity, height=threshold, distance=self.frame_rate/10)
        valley_indices, valley_data = find_peaks(-1*angular_velocity, height=threshold, distance=self.frame_rate/10)
        
        # Combine and sort peak and valley indices
        indices = np.concatenate([peak_indices, valley_indices])
        data = np.concatenate([peak_data['peak_heights'], valley_data['peak_heights']*-1])
        sort_indices = np.argsort(indices)
        indices, data = indices[sort_indices], data[sort_indices]

        for i, idx in enumerate(indices):
            sacc_dur, _, sacc_start, sacc_stop = peak_widths(np.abs(angular_velocity), [idx], rel_height=0.95)
            
            start_idx = int(round(sacc_start[0]))
            stop_idx = int(round(sacc_stop[0]))
            peak_time = idx / self.frame_rate
            
            # Determine the direction based on the sign of the peak
            direction = 'right' if data[i] > 0 else 'left'
            
            saccade = {
                'saccade_peak_s': peak_time,
                'saccade_start_idx': start_idx,
                'sacc_peak_idx': idx,
                'saccade_stop_idx': stop_idx,
                'amplitude_deg': np.rad2deg(angles[stop_idx] - angles[start_idx]),
                'top_speed_degPs': data[i],
                'saccade_duration_s': sacc_dur[0] / self.frame_rate,
                'direction': direction,
            }
            saccades.append(saccade)

        # filter out saccades that are not biologically relevant
        saccades = self.filter_artifacts(saccades)
        self.saccades = saccades
        return saccades

        
    def filter_artifacts(self, saccades):
        """
        Filters out saccades that are not biologically relevant based on specified duration and amplitude thresholds.

        Parameters
        ----------
        saccades : list
            List of saccades.

        Returns
        -------
        filtered_saccades : list
            List of saccades after filtering out non-biologically relevant saccades.
        """
        filtered_saccades = [saccade for saccade in saccades
                            if (np.abs(saccade['top_speed_degPs']) <= self.max_speed_degPsec and 
                                saccade['saccade_duration_s'] <= self.max_duration_sec and 
                                np.abs(saccade['amplitude_deg']) >= self.min_amplitude_deg)]
        return filtered_saccades
    

    def plot_saccades(self, angles, angular_velocity, saccades):
        """
        Plot angular velocity and angles with markers for saccade starts, stops, and peaks.

        Parameters
        ----------
        angles : list
            List of angles.
        angular_velocity : list
            List of angular velocities.
        saccades : list
            List of dictionaries containing the saccade times, start and stop times, and amplitudes.
        """
        # Create subplots
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        # Plot angles
        axs[0].plot(np.arange(len(angles)) / self.frame_rate, angles, label='Angles')
        axs[0].set_ylabel('Angles (deg)')
        axs[0].set_title('Angles and Angular Velocity')
        
        # Plot angular velocity
        axs[1].plot(np.arange(len(angular_velocity)) / self.frame_rate, angular_velocity, label='Angular Velocity')
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Angular Velocity (deg/s)')

        # Add markers for saccade starts, stops, and peaks
        for saccade in saccades:
            # Saccade start
            axs[1].plot(saccade['saccade_start_idx'] / self.frame_rate, angular_velocity[saccade['saccade_start_idx']], 'g>', markersize=10)
            # Saccade stop
            axs[1].plot(saccade['saccade_stop_idx'] / self.frame_rate, angular_velocity[saccade['saccade_stop_idx']], 'r<', markersize=10)
            # Saccade peak
            if saccade['direction'] == 'right':
                axs[1].plot(saccade['saccade_peak_s'], angular_velocity[saccade['sacc_peak_idx']], 'b*', markersize=10)
            else:
                axs[1].plot(saccade['saccade_peak_s'], angular_velocity[saccade['sacc_peak_idx']], 'r*', markersize=10)

        plt.tight_layout()
        plt.show()

    def compute_angular_velocity(self, angles):
        """
        Converts angles to a continuous format and computes angular velocity.

        Parameters
        ----------
        angles : list
            List of angles.

        Returns
        -------
        angles : list
            List of angles converted to a continuous format.
        velocities : list
            List of computed angular velocities.
        """
        # Convert angles to a continuous format
        angles = self.continuous_angle(angles)

        # Compute angular velocities
        velocities = np.diff(angles)
        velocities = np.rad2deg(velocities) * self.frame_rate

        return angles, velocities
    
    def normalize_angle_data(self, angle_data):
        """
        Normalizes the angle data in a saccade window by subtracting the geometric mean of the first three non-NaN angle values from all angle values in the window.

        Parameters
        ----------
        angle_data : np.array
            Numpy array containing angle data in a saccade window.

        Returns
        -------
        normalized_angle_data : np.array
            Numpy array with normalized angle data in a saccade window.
        """
        # Find the first three non-NaN values
        first_three_values = angle_data[~np.isnan(angle_data)][:3]
        
        geometric_mean = np.mean(first_three_values)

        # Normalize angle data by subtracting geometric mean
        normalized_angle_data = np.array([angle - geometric_mean if not np.isnan(angle) else np.nan for angle in angle_data])

        return normalized_angle_data

    
    def get_saccade_triggered_average(self, angle_data, velocity_data, window_length):
        """
        Extracts saccade data using an adjusted window size if the full window cannot be used and normalizes the angle data.
        Differentiates between positive and negative saccades.

        Parameters
        ----------
        angle_data : np.array
            Numpy array containing angle data.
        velocity_data : np.array
            Numpy array containing velocity data.
        window_length : int
            Desired window length.

        Returns
        -------
        pos_angle_matrix : np.array
            3D numpy array with normalized angle data for each positive saccade window.
        pos_velocity_matrix : np.array
            3D numpy array with velocity data for each positive saccade window.
        neg_angle_matrix : np.array
            3D numpy array with normalized angle data for each negative saccade window.
        neg_velocity_matrix : np.array
            3D numpy array with velocity data for each negative saccade window.
        """
        half_window =window_length // 2

        positive_saccades = [saccade for saccade in self.saccades if saccade['top_speed_degPs'] > 0]
        negative_saccades = [saccade for saccade in self.saccades if saccade['top_speed_degPs'] < 0]

        pos_angle_matrix = np.full((len(positive_saccades), window_length), np.nan)
        pos_velocity_matrix = np.full((len(positive_saccades), window_length), np.nan)
        neg_angle_matrix = np.full((len(negative_saccades), window_length), np.nan)
        neg_velocity_matrix = np.full((len(negative_saccades), window_length), np.nan)

        for i, saccade in enumerate(positive_saccades):
            start_index = max(0, saccade['sacc_peak_idx'] - half_window)
            end_index = min(len(angle_data), saccade['sacc_peak_idx'] + half_window)

            # Extract windowed data and normalize angle data
            windowed_angle_data = self.normalize_angle_data(angle_data[start_index:end_index])
            windowed_velocity_data = velocity_data[start_index:end_index]

            # Fill matrices with windowed data
            pos_angle_matrix[i, :len(windowed_angle_data)] = windowed_angle_data
            pos_velocity_matrix[i, :len(windowed_velocity_data)] = windowed_velocity_data

        for i, saccade in enumerate(negative_saccades):
            start_index = max(0, saccade['sacc_peak_idx'] - half_window)
            end_index = min(len(angle_data), saccade['sacc_peak_idx'] + half_window)

            # Extract windowed data and normalize angle data
            windowed_angle_data = self.normalize_angle_data(angle_data[start_index:end_index])
            windowed_velocity_data = velocity_data[start_index:end_index]

            # Fill matrices with windowed data
            neg_angle_matrix[i, :len(windowed_angle_data)] = windowed_angle_data
            neg_velocity_matrix[i, :len(windowed_velocity_data)] = windowed_velocity_data

        return pos_angle_matrix, pos_velocity_matrix, neg_angle_matrix, neg_velocity_matrix


    def main(self,angles,threshold,window_size, plot_now = False):
        """
        This is the main function for analyzing saccades from given angles and a threshold for peak angular velocity. 
        
        It first computes the angular velocities from the angles, then finds saccades based on the computed velocities and provided threshold.
        It proceeds to extract saccade-triggered windows around each saccade peak, separates positive and negative saccades,
        normalizes the saccades, and transforms the data into matrices for further analysis.
        If the plot_now parameter is set to True, it plots the angular velocities with marks indicating saccade start, peak, and stop points.

        Parameters
        ----------
        angles : list
            List of angles in degrees.

        threshold : float
            Threshold for peak angular velocity (in degrees per second) to define a saccade.
        
        plot_now : bool, optional
            If set to True, the function plots the angular velocities with marks indicating saccade start, peak, and stop points. 
            Default is False.

        Returns
        -------
        saccades : list of dictionaries
            List containing information about each saccade, including start, peak, stop indices, amplitude, duration, and top speed.

        pos_angle_matrix : np.array
            A matrix of normalized angle data for each positive saccade (shape m x n x 2).

        pos_velocity_matrix : np.array
            A matrix of velocity data for each positive saccade (shape m x n x 2).

        neg_angle_matrix : np.array
            A matrix of normalized angle data for each negative saccade (shape m x n x 2).

        neg_velocity_matrix : np.array
            A matrix of velocity data for each negative saccade (shape m x n x 2).
        """

        angles, velocities = self.compute_angular_velocity(angles)
        angles = np.rad2deg(angles)
        saccades = self.find_saccades(angles,velocities,threshold=threshold)
        pos_angle_matrix, pos_velocity_matrix, neg_angle_matrix, neg_velocity_matrix = self.get_saccade_triggered_average(angles,velocities,window_size)
        if plot_now:
            self.plot_saccades(angles,velocities,saccades)

        return saccades,pos_angle_matrix,pos_velocity_matrix,neg_angle_matrix,neg_velocity_matrix
    

    def collect_more_triggered_data(self, saccades, data_array, window_length):
        """
        Collects windowed data around the saccade peaks from the given data_array, 
        separated by positive and negative amplitude saccades.

        Parameters:
        -----------
        saccades : list of dicts
            List containing information about each saccade, including the peak index and amplitude.
        
        data_array : np.ndarray
            1D array containing the data for which the triggered data is to be collected.
        
        window_length : int
            The length of the window around each peak for which the triggered data is collected.

        Returns:
        --------
        pos_triggered_matrix : np.ndarray
            2D array containing the collected windowed data for saccades with positive amplitude.
        
        neg_triggered_matrix : np.ndarray
            2D array containing the collected windowed data for saccades with negative amplitude.
        """
        half_window = window_length // 2
        pos_triggered_matrix = []
        neg_triggered_matrix = []

        for saccade in saccades:
            peak_idx = saccade['sacc_peak_idx']
            amplitude = saccade['amplitude_deg']
            start_idx = max(0, peak_idx - half_window)
            end_idx = min(len(data_array), peak_idx + half_window)

            # Extract the data around the peak
            window_data = data_array[start_idx:end_idx]

            # Pad window_data if it's shorter than window_length
            if len(window_data) < window_length:
                padding = window_length - len(window_data)
                window_data = np.pad(window_data, (0, padding), 'constant', constant_values=np.nan)

            if amplitude > 0:
                pos_triggered_matrix.append(window_data)
            else:
                neg_triggered_matrix.append(window_data)

        pos_triggered_matrix = np.array(pos_triggered_matrix)
        neg_triggered_matrix = np.array(neg_triggered_matrix)
        
        return pos_triggered_matrix, neg_triggered_matrix
