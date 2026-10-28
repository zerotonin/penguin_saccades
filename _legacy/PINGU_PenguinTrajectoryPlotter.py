import pandas as pd
import numpy as np
import os
import cv2
import glob
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter1d

class PenguinTrajectoryPlotter:
    """
    Creates a 4×5 multi-subplot figure for a selected penguin segment, including:
      - A large trajectory plot with skeleton overlays (only markers and skeletons with likelihood >= 0.75)
      - Three stacked time-series subplots (yaw velocity, yaw angle, translational velocity)
      - Eight sample frames from the labeled video (arranged in a 2×4 grid)
      
    The time-series plots use all the data from the CSV (already cut down to the segment).
    If custom frame_times are provided, they are used only to extract the 8 video frames.
    """

    def __init__(
        self,
        segment_id=52,
        csv_path="/path/to/Penguin_Rostock/unified_outputs/unified_trajectories_head_body.csv",
        video_dir="/path/to/Penguin_Rostock/pengu_head_movies/",
        dataset_filter="closed",
        species_filter="gentoo",
        frame_rate=25,
        frame_times=None
    ):
        """
        Parameters
        ----------
        segment_id : int
            Which segment_id to filter from the CSV.
        csv_path : str
            Path to the unified CSV with required columns.
        video_dir : str
            Directory holding the DLC .h5 and labeled .mp4 for the segment.
        dataset_filter : str
            Filter for dataset (here, 'closed').
        species_filter : str
            Filter for species (default 'gentoo').
        frame_rate : float
            Frame rate (FPS) used to convert frame_index to time in ms.
        frame_times : list of float or None
            If provided, exactly 8 times (in seconds) at which to sample frames from the video.
            If None, 8 frames are sampled evenly from the video.
        """
        self.segment_id = segment_id
        self.csv_path = csv_path
        self.video_dir = video_dir
        self.dataset_filter = dataset_filter
        self.species_filter = species_filter
        self.frame_rate = frame_rate
        self.frame_times = frame_times
        if self.frame_times is not None and len(self.frame_times) != 8:
            raise ValueError("If providing frame_times, must have exactly 8 entries (in seconds).")

    def create_figure(self):
        """
        Produces the 4×5 subplot figure:
          - Trajectory plot with skeleton overlays (likelihood >= 0.75 only)
            [spanning rows 1–3, columns 1–2]: The y-axis is inverted and all axis elements are removed.
          - Yaw velocity (deg/s) vs time (head in blue, body in red) [row 1, columns 3–4]
          - Yaw angle (deg) vs time (head in blue, body in red) [row 2, columns 3–4]
          - Translational velocity (m/s, low-pass filtered) vs time [row 3, columns 3–4]
          - 8 video frames [rows 4–5, columns 1–4] arranged in a 2×4 grid.
        Returns
        -------
        fig : matplotlib.figure.Figure
            The generated figure.
        """
        # 1) Load data from CSV
        df = pd.read_csv(self.csv_path)
        df_sub = df[
            (df["species"] == self.species_filter) &
            (df["dataset"] == self.dataset_filter) &
            (df["segment_id"] == self.segment_id)
        ].copy()
        if df_sub.empty:
            raise RuntimeError(f"No data for segment_id={self.segment_id}, dataset='{self.dataset_filter}', "
                               f"species='{self.species_filter}' in {self.csv_path}")
        df_sub.sort_values(by="frame_index", inplace=True)
        time_ms = df_sub["frame_index"].to_numpy() / self.frame_rate * 1000.0

        # 2) Extract angles and velocities
        bodyYawRad = df_sub["body_yaw_rad"].to_numpy()
        headYawRad = df_sub["head_yaw_rad"].to_numpy()
        bodyVelDegPs = df_sub["body_yaw_speed_degPs"].to_numpy()  # already in deg/s
        headVelDegPs = df_sub["head_yaw_speed_degPs"].to_numpy()  # already in deg/s
        translVel = df_sub["translational_velocity_mPs"].to_numpy()  # m/s

        bodyYawDeg = bodyYawRad * (180.0 / np.pi)
        headYawDeg = headYawRad * (180.0 / np.pi)

        # 3) Find corresponding DLC .h5 and video
        h5_pattern = os.path.join(self.video_dir, f"*{self.segment_id}*.h5")
        h5_files = glob.glob(h5_pattern)
        if not h5_files:
            raise RuntimeError(f"No DLC .h5 found for segment {self.segment_id} in {self.video_dir}")
        h5_path = h5_files[0]
        if len(h5_files) > 1:
            for path in h5_files:
                base = os.path.basename(path)
                if f"_{self.segment_id}." in base or base.startswith(str(self.segment_id)):
                    h5_path = path
                    break
        dlc_df = pd.read_hdf(h5_path)
        if isinstance(dlc_df.columns, pd.MultiIndex):
            dlc_df.columns = [f"{bp}_{coord}" for (_, bp, coord) in dlc_df.columns]

        keypoints = ["beak", "neck", "left_fin", "right_fin", "caudal"]
        for kp in keypoints:
            if f"{kp}_x" not in dlc_df.columns or f"{kp}_y" not in dlc_df.columns or f"{kp}_likelihood" not in dlc_df.columns:
                raise RuntimeError(f"Missing keypoint columns for {kp} in {h5_path}")

        mp4_pattern = os.path.join(self.video_dir, f"*{self.segment_id}*DLC*.mp4")
        mp4_files = glob.glob(mp4_pattern)
        if not mp4_files:
            raise RuntimeError(f"No labeled .mp4 found for segment {self.segment_id} in {self.video_dir}")
        video_path = mp4_files[0]
        if len(mp4_files) > 1:
            for path in mp4_files:
                base = os.path.basename(path)
                if f"_{self.segment_id}." in base or base.startswith(str(self.segment_id)):
                    video_path = path
                    break

        # 4) Extract 8 frames from video (using frame_times if provided)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_vid = cap.get(cv2.CAP_PROP_FPS)
        if fps_vid <= 1.0:
            fps_vid = 30.0
        duration_ms = (total_frames / fps_vid) * 1000.0

        if self.frame_times is not None:
            frame_times_ms = [sec * 1000.0 for sec in self.frame_times]
        else:
            indices = np.linspace(0, total_frames - 1, 8, dtype=int)
            frame_times_ms = [(idx / fps_vid) * 1000.0 for idx in indices]

        frames_rgb = []
        for t_ms in frame_times_ms:
            cap.set(cv2.CAP_PROP_POS_MSEC, t_ms)
            ret, frame_img = cap.read()
            if not ret or frame_img is None:
                frame_img = np.zeros((100, 100, 3), dtype=np.uint8)
            frame_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
            frames_rgb.append(frame_rgb)
        cap.release()
        if len(frames_rgb) < 8:
            last_frame = frames_rgb[-1] if frames_rgb else np.zeros((100, 100, 3), dtype=np.uint8)
            while len(frames_rgb) < 8:
                frames_rgb.append(last_frame)
        elif len(frames_rgb) > 8:
            frames_rgb = frames_rgb[:8]

        # 5) Create figure with GridSpec (DIN A4 landscape: approx 11.7" x 8.3")
        fig = plt.figure(figsize=(11.7, 8.3))
        gs = GridSpec(nrows=5, ncols=4, figure=fig, wspace=0.05, hspace=0.05)

        ax_traj = fig.add_subplot(gs[0:3, 0:2])
        ax_traj.invert_yaxis()
        ax_traj.axis("off")

        ax_yaw_vel = fig.add_subplot(gs[0, 2:4])
        ax_yaw_ang = fig.add_subplot(gs[1, 2:4], sharex=ax_yaw_vel)
        ax_trans_vel = fig.add_subplot(gs[2, 2:4], sharex=ax_yaw_vel)

        img_axes = []
        for rr in range(3, 5):
            for cc in range(4):
                ax_img = fig.add_subplot(gs[rr, cc])
                ax_img.axis("off")
                img_axes.append(ax_img)

        # 6) Trajectory skeleton: Plot a 90% transparent skeleton from every frame
        color_map = {"beak": "C1", "neck": "C2", "left_fin": "C4", "right_fin": "C5", "caudal": "C6"}
        for idx_skel in range(len(dlc_df)):
            coords = {}
            for kp in keypoints:
                if dlc_df[f"{kp}_likelihood"].iloc[idx_skel] >= 0.75:
                    x_val = dlc_df[f"{kp}_x"].iloc[idx_skel]
                    y_val = dlc_df[f"{kp}_y"].iloc[idx_skel]
                    coords[kp] = (x_val, y_val)
                else:
                    coords[kp] = None
            for kp in keypoints:
                if coords[kp] is not None:
                    xx, yy = coords[kp]
                    ax_traj.scatter(xx, yy, color=color_map.get(kp, "gray"), s=12, alpha=0.1)
            def draw_line(k1, k2):
                if coords[k1] is not None and coords[k2] is not None:
                    ax_traj.plot([coords[k1][0], coords[k2][0]],
                                 [coords[k1][1], coords[k2][1]],
                                 color="k", linewidth=1, alpha=0.1)
            draw_line("beak", "neck")
            draw_line("neck", "caudal")
            draw_line("neck", "left_fin")
            draw_line("neck", "right_fin")
        # Also plot every 10th frame as a solid skeleton with markers and time labels
        for idx_skel in range(0, len(dlc_df), 10):
            coords = {}
            for kp in keypoints:
                if dlc_df[f"{kp}_likelihood"].iloc[idx_skel] >= 0.75:
                    x_val = dlc_df[f"{kp}_x"].iloc[idx_skel]
                    y_val = dlc_df[f"{kp}_y"].iloc[idx_skel]
                    coords[kp] = (x_val, y_val)
                else:
                    coords[kp] = None
            for kp in keypoints:
                if coords[kp] is not None:
                    xx, yy = coords[kp]
                    ax_traj.scatter(xx, yy, color=color_map.get(kp, "gray"), s=12)
            def draw_line(k1, k2):
                if coords[k1] is not None and coords[k2] is not None:
                    ax_traj.plot([coords[k1][0], coords[k2][0]],
                                 [coords[k1][1], coords[k2][1]],
                                 color="k", linewidth=1)
            draw_line("beak", "neck")
            draw_line("neck", "caudal")
            draw_line("neck", "left_fin")
            draw_line("neck", "right_fin")
            if coords["caudal"] is not None:
                cx, cy = coords["caudal"]
                t_label_ms = int(idx_skel / fps_vid * 1000.0)
                ax_traj.text(cx + 2, cy + 2, f"{t_label_ms} ms", fontsize=8, color="k")
        # Create a legend (dummy handles) for keypoints
        legend_handles = []
        for kp in keypoints:
            label_str = kp.replace("_", " ")
            col = color_map.get(kp, "gray")
            (ln,) = ax_traj.plot([], [], color=col, marker="o", linestyle="None", label=label_str)
            legend_handles.append(ln)
        ax_traj.legend(handles=legend_handles, loc="upper left")
        ax_traj.set_aspect("equal")

        # 7) Yaw velocity plot (body=red, head=blue)
        ax_yaw_vel.plot(time_ms, bodyVelDegPs, color="red", label="Body yaw vel")
        ax_yaw_vel.plot(time_ms, headVelDegPs, color="blue", label="Head yaw vel")
        ax_yaw_vel.set_ylabel("Yaw vel (deg/s)")
        ax_yaw_vel.legend(loc="upper right")
        ax_yaw_vel.tick_params(axis="x", labelbottom=False)

        # 8) Yaw angle plot (body=red, head=blue)
        ax_yaw_ang.plot(time_ms, bodyYawDeg, color="red", label="Body yaw angle")
        ax_yaw_ang.plot(time_ms, headYawDeg, color="blue", label="Head yaw angle")
        ax_yaw_ang.set_ylabel("Yaw (deg)")
        ax_yaw_ang.legend(loc="upper right")
        ax_yaw_ang.tick_params(axis="x", labelbottom=False)

        # 9) Translational velocity plot (black) with low-pass filtering
        translVel_filt = gaussian_filter1d(translVel, sigma=3)
        ax_trans_vel.plot(time_ms, translVel_filt, color="black")
        ax_trans_vel.set_ylabel("Transl. vel (m/s)")
        ax_trans_vel.set_xlabel("time in ms")

        # 10) Place the 8 frames in the bottom subplots
        for i, ax_img in enumerate(img_axes):
            ax_img.imshow(frames_rgb[i])

        plt.tight_layout()
        return fig

# -------------------------------------------------------------------------
# Run as main
# -------------------------------------------------------------------------
if __name__ == "__main__":
    plotter = PenguinTrajectoryPlotter(
        segment_id=52,
        dataset_filter="closed",
        species_filter="gentoo",
        frame_rate=25,
        frame_times = [5.8, 6.1, 6.6, 7, 7.5, 7.8, 8.3, 8.72]
    )
    fig = plotter.create_figure()
    outdir = "/path/to/Penguin_Rostock/unified_outputs/"
    png_path = os.path.join(outdir, "segment52_figure.png")
    svg_path = os.path.join(outdir, "segment52_figure.svg")
    fig.savefig(png_path, dpi=300)
    fig.savefig(svg_path)
    print(f"Figure saved:\n  {png_path}\n  {svg_path}")
    plt.show()
