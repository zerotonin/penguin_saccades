import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats

def mean_and_ci(data, confidence=0.95):
    """
    Compute nanmean + 95% confidence interval (t-based), ignoring NaNs.
    Returns (mean, ci_lower, ci_upper).
    """
    arr = data[~np.isnan(data)]
    n = len(arr)
    if n < 2:
        if n == 1:
            return arr[0], arr[0], arr[0]
        else:
            return np.nan, np.nan, np.nan
    mean_ = np.mean(arr)
    sem_ = scipy.stats.sem(arr)
    t_val = scipy.stats.t.ppf((1 + confidence)/2.0, n - 1)
    ci_lower = mean_ - t_val * sem_
    ci_upper = mean_ + t_val * sem_
    return mean_, ci_lower, ci_upper

def plot_two_subplots(csv_path):
    """
    Reads a 'melted' CSV with columns:
      species, dataset, trajectory_id, isi_index, chunk_index, variable, frame, value
    Filters for gentoo, then plots:
      - Left subplot: velocity => (headYawVel=blue, bodyYawVel=red)
      - Right subplot: angle    => (headYawDeg=blue, bodyYawDeg=red)

    We return the figure handle, so the calling code can save it as SVG/PNG.
    """
    df = pd.read_csv(csv_path)
    
    # 1) Only gentoo
    df = df[df['species'] == 'gentoo'].copy()
    if df.empty:
        print("No gentoo data found. Returning None.")
        return None
    
    # 2) Time axis from -0.5..0.5 for frames 0..24
    time_axis = np.linspace(-0.5, 0.5, 25)
    
    # 3) Group by (variable, frame), compute mean & CI
    grouped = df.groupby(['variable','frame'])['value']
    stats_list = []
    for (var, frm), subgrp in grouped:
        m, c_lo, c_hi = mean_and_ci(subgrp.values, confidence=0.95)
        stats_list.append({
            'variable': var,
            'frame': frm,
            'mean': m,
            'ci_lower': c_lo,
            'ci_upper': c_hi
        })
    stats_df = pd.DataFrame(stats_list)

    # Helper to extract arrays sorted by frame
    def get_arrays(varname):
        sub = stats_df[stats_df['variable'] == varname].sort_values('frame')
        sub = sub.set_index('frame').reindex(range(25))  # ensure frames 0..24
        mean_ = sub['mean'].values
        low_  = sub['ci_lower'].values
        up_   = sub['ci_upper'].values
        return mean_, low_, up_

    # 4) Grab head vs body, velocity vs angle
    headVel_mean, headVel_low, headVel_up = get_arrays('headYawVel')
    bodyVel_mean, bodyVel_low, bodyVel_up = get_arrays('bodyYawVel')
    headDeg_mean, headDeg_low, headDeg_up = get_arrays('headYawDeg')
    bodyDeg_mean, bodyDeg_low, bodyDeg_up = get_arrays('bodyYawDeg')
    
    # 5) Create a figure with two subplots side by side
    #    Note: figsize is up to you. Here let's pick 10 x 5 for clarity.
    fig, (ax_vel, ax_deg) = plt.subplots(1, 2, figsize=(10, 5), sharex=True)

    
    # Left: velocity
    ax_vel.set_xlim(-0.5, 0.5)
    ax_vel.set_ylim(-50, 600)
    ax_vel.set_xlabel("Time (s)")
    ax_vel.set_ylabel("Rotation Velocity (deg/s)")

    # Head velocity (blue)
    ax_vel.plot(time_axis, headVel_mean, color='blue', label='Head Velocity')
    ax_vel.fill_between(time_axis, headVel_low, headVel_up, color='blue', alpha=0.2)
    # Body velocity (red)
    ax_vel.plot(time_axis, bodyVel_mean, color='red', label='Body Velocity')
    ax_vel.fill_between(time_axis, bodyVel_low, bodyVel_up, color='red', alpha=0.2)
    ax_vel.legend(loc='upper left')
    ax_vel.set_title("Velocity")

    # Right: angle
    ax_deg.set_xlim(-0.5, 0.5)
    ax_deg.set_ylim(-50, 250)
    ax_deg.set_xlabel("Time (s)")
    ax_deg.set_ylabel("Rotation Angle (deg)")

    # Head angle (blue)
    ax_deg.plot(time_axis, headDeg_mean, color='blue', label='Head Angle')
    ax_deg.fill_between(time_axis, headDeg_low, headDeg_up, color='blue', alpha=0.2)
    # Body angle (red)
    ax_deg.plot(time_axis, bodyDeg_mean, color='red', label='Body Angle')
    ax_deg.fill_between(time_axis, bodyDeg_low, bodyDeg_up, color='red', alpha=0.2)
    ax_deg.legend(loc='upper left')
    ax_deg.set_title("Angle")

    plt.suptitle("Gentoo Triggered Averages (Angles Zero-Offset)\nLeft: Velocity, Right: Angle")

    plt.tight_layout()
    return fig  # Return the figure handle

# ---------------------------------------------------------------------
# Usage example:
if __name__ == "__main__":
    csv_file = "/path/to/Penguin_Rostock/unified_outputs/all_isi_chunks_melted.csv"
    
    # Produce the figure, get the handle
    fig = plot_two_subplots(csv_file)

    if fig is not None:
        # Save as SVG and PNG outside the function
        fig.savefig("/path/to/Penguin_Rostock/unified_outputs/gentoo_triggered_angleVel.svg", format="svg")
        fig.savefig("/path/to/Penguin_Rostock/unified_outputs/gentoo_triggered_angleVel.png", format="png")
        print("Figure saved.")
    else:
        print("No gentoo data. No figure saved.")
    
    plt.show()
