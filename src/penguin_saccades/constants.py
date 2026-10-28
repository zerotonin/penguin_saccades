# ╔══════════════════════════════════════════════════════════════════╗
# ║  penguin_saccades — constants                                    ║
# ║  « one source of truth for numbers, names and colours »          ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Every frame rate, threshold, calibration constant, smoothing    ║
# ║  width, column name and colour used anywhere in the pipeline     ║
# ║  lives here. Values marked TO RATIFY must be confirmed against   ║
# ║  the raw videos and stated in the Methods before the revision    ║
# ║  is submitted.                                                   ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Shared constants: data set specs, detection thresholds, column names, colours."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

# ┌────────────────────────────────────────────────────────────┐
# │ Type aliases  « array shorthands used in signatures »      │
# └────────────────────────────────────────────────────────────┘
FloatArray: TypeAlias = np.ndarray
IndexArray: TypeAlias = np.ndarray


# ┌────────────────────────────────────────────────────────────┐
# │ Data sets  « frame rates, image geometry, calibration »    │
# └────────────────────────────────────────────────────────────┘
@dataclass(frozen=True)
class DatasetSpec:
    """Recording properties of one data set.

    Attributes:
        name:            Short key used in the ``dataset`` column.
        label:           Human-readable description for figures and tables.
        frame_rate_hz:   Video frame rate.
        image_width_px:  Frame width, needed to undo relative coordinates.
        image_height_px: Frame height.
        m_per_px:        Metres per pixel on the water surface, or ``None``
                         when scale comes from body length instead.
    """

    name: str
    label: str
    frame_rate_hz: float
    image_width_px: int | None
    image_height_px: int | None
    m_per_px: float | None


# Three tape-measure calibrations (metres, pixels) of the overhead camera.
CALIBRATION_SAMPLES_M_PX: tuple[tuple[float, float], ...] = (
    (0.97, 124.6),
    (0.30, 66.0),
    (0.40, 76.2),
)
OPEN_M_PER_PX: float = float(np.mean([m / px for m, px in CALIBRATION_SAMPLES_M_PX]))

# TO RATIFY: the legacy code used 30 fps for data set 1 while the manuscript
# states 60 fps.  Velocities scale linearly and durations inversely with
# this number, so it must be read from the video files, not assumed.
OPEN_FRAME_RATE_HZ: float = 30.0
CLOSED_FRAME_RATE_HZ: float = 25.0

DATASET_OPEN = DatasetSpec(
    name="open",
    label="Data set 1: overhead camera, Faster-RCNN body detection",
    frame_rate_hz=OPEN_FRAME_RATE_HZ,
    image_width_px=1402,
    image_height_px=788,
    m_per_px=OPEN_M_PER_PX,
)
DATASET_CLOSED = DatasetSpec(
    name="closed",
    label="Data set 2: DeepLabCut head and body pose",
    frame_rate_hz=CLOSED_FRAME_RATE_HZ,
    image_width_px=None,
    image_height_px=None,
    m_per_px=None,
)

# Scale for data set 2 comes from the animal itself.  Source: assumed
# adult gentoo beak-to-tail length; TO RATIFY with a citation.
GENTOO_BODY_LENGTH_M: float = 0.80

# Species inferred from file names of data set 1; data set 2 is gentoo only.
SPECIES_FROM_FILENAME: dict[str, str] = {"Rockhopper": "rockhopper", "Gentoo": "gentoo"}
CLOSED_SPECIES: str = "gentoo"

# Region of the overhead frame that is water, in pixel coordinates
# (x, y).  Detections outside it are penguins on land.
OPEN_WATER_ROI_PX: tuple[tuple[float, float], ...] = (
    (0, 0), (660, 0), (660, 272), (1402, 272), (1402, 788), (0, 788),
)


# ┌────────────────────────────────────────────────────────────┐
# │ DeepLabCut  « keypoints, vectors and likelihood gating »   │
# └────────────────────────────────────────────────────────────┘
DLC_KEYPOINTS: tuple[str, ...] = ("beak", "neck", "left_fin", "right_fin", "caudal")
HEAD_VECTOR: tuple[str, str] = ("neck", "beak")     # tail of vector, tip of vector
BODY_VECTOR: tuple[str, str] = ("caudal", "neck")
DLC_LIKELIHOOD_THRESHOLD: float = 0.75
# Gaps shorter than this (frames) are linearly interpolated; longer gaps
# split the recording into separate segments.
MAX_INTERPOLATION_GAP_FRAMES: int = 5


# ┌──────────────────────────────────────────────────────────────────┐
# │ Trajectory processing  « segments, smoothing, heading validity » │
# └──────────────────────────────────────────────────────────────────┘
# Data set 1 detections may drop out; a gap up to this many frames is
# bridged by interpolation, a longer one starts a new segment.
MAX_GAP_FRAMES: int = 10
# A segment must be at least this long to host a saccade window.
MIN_SEGMENT_S: float = 1.0
# One Gaussian smoothing width in seconds, applied identically to both
# data sets (sigma in frames = SMOOTHING_SIGMA_S * frame rate).  The
# manuscript describes a 3-frame Gaussian at 60 fps, i.e. 0.05 s.  Wider
# kernels attenuate the peak velocity of 200 to 300 ms saccades.
SMOOTHING_SIGMA_S: float = 0.05
# Heading derived from displacement is undefined when the animal is
# (nearly) stationary.  Frames slower than this get NaN heading.
MIN_HEADING_SPEED_MPS: float = 0.10


# ┌────────────────────────────────────────────────────────────┐
# │ Saccade detection  « one detector, one artefact filter »   │
# └────────────────────────────────────────────────────────────┘
SACCADE_VELOCITY_THRESHOLD_DEGPS: float = 200.0
# Two peaks closer than this are one saccade.
MIN_PEAK_SEPARATION_S: float = 0.10
# Saccade boundaries are taken where |velocity| falls to this fraction
# of the peak prominence (scipy peak_widths rel_height).
BOUNDARY_REL_HEIGHT: float = 0.95
# Amplitude is the heading change between the mean heading in a settle
# window before the start and after the stop of the event, so a
# one-frame tracking glitch (flip and return) has zero amplitude.
AMPLITUDE_SETTLE_S: float = 0.10
MAX_SACCADE_DURATION_S: float = 1.0
MIN_SACCADE_AMPLITUDE_DEG: float = 10.0
MAX_SACCADE_SPEED_DEGPS: float = 3000.0
# Positive yaw velocity in image coordinates (y down, camera overhead)
# is a clockwise turn, i.e. a right turn for the animal.
DIRECTION_OF_POSITIVE_VELOCITY: str = "right"
DIRECTION_OF_NEGATIVE_VELOCITY: str = "left"


# ┌────────────────────────────────────────────────────────────┐
# │ Taxonomy  « head and body saccade association »            │
# └────────────────────────────────────────────────────────────┘
# A head saccade within this window of a same-direction body saccade is
# "associated"; otherwise it is a free head saccade.  TO RATIFY: 1.0 s is
# the legacy value and is generous for 50 to 300 ms events.
ASSOCIATION_WINDOW_S: float = 1.0
SACCADE_TYPES: tuple[str, ...] = ("associated", "free_head", "free_body")
INTERVAL_CLASSES: tuple[str, ...] = ("body", "head", "any")


# ┌────────────────────────────────────────────────────────────┐
# │ Triggered averages and ISI chunks  « window geometry »     │
# └────────────────────────────────────────────────────────────┘
TRIGGER_WINDOW_S: float = 1.0
ISI_CHUNK_S: float = 1.0
# Minimum frames an interval must have for head-body correlation.
MIN_COMPENSATION_FRAMES: int = 5


# ┌────────────────────────────────────────────────────────────┐
# │ Statistics  « reproducibility settings »                   │
# └────────────────────────────────────────────────────────────┘
N_PERMUTATIONS: int = 10_000
N_BOOTSTRAP: int = 10_000
RANDOM_SEED: int = 20251028
ALPHA: float = 0.05
CONFIDENCE_LEVEL: float = 0.95


# ┌────────────────────────────────────────────────────────────┐
# │ Column names  « the schema shared by every stage »         │
# └────────────────────────────────────────────────────────────┘
COL_DATASET = "dataset"
COL_SPECIES = "species"
COL_TRAJECTORY = "trajectory_id"
COL_SEGMENT = "segment_id"
COL_FRAME = "frame_index"
COL_TIME = "time_s"
COL_X = "x_m"
COL_Y = "y_m"
COL_BODY_YAW = "body_yaw_rad"
COL_BODY_VEL = "body_yaw_speed_degps"
COL_HEAD_YAW = "head_yaw_rad"
COL_HEAD_VEL = "head_yaw_speed_degps"
COL_TRANS = "translational_speed_mps"
COL_SOURCE = "source"            # "body" or "head"
COL_SACCADE_TYPE = "saccade_type"
COL_INTERVAL_CLASS = "interval_class"

TRAJECTORY_COLUMNS: tuple[str, ...] = (
    COL_DATASET, COL_SPECIES, COL_TRAJECTORY, COL_SEGMENT, COL_FRAME, COL_TIME,
    COL_X, COL_Y, COL_BODY_YAW, COL_BODY_VEL, COL_HEAD_YAW, COL_HEAD_VEL, COL_TRANS,
)


# ┌────────────────────────────────────────────────────────────┐
# │ Wong (2011) palette  « colourblind-safe base colours »     │
# └────────────────────────────────────────────────────────────┘
WONG: dict[str, str] = {
    "black":          "#000000",
    "orange":         "#E69F00",
    "sky_blue":       "#56B4E9",
    "bluish_green":   "#009E73",
    "yellow":         "#F0E442",
    "blue":           "#0072B2",
    "vermilion":      "#D55E00",
    "reddish_purple": "#CC79A7",
}

# Species and body part use disjoint palettes so a reader never has to
# guess whether red means gentoo or body (revision note, section 3).
SPECIES_COLOURS: dict[str, str] = {
    "gentoo":     WONG["blue"],
    "rockhopper": WONG["orange"],
}
BODYPART_COLOURS: dict[str, str] = {
    "head": WONG["bluish_green"],
    "body": WONG["vermilion"],
}
SACCADE_TYPE_COLOURS: dict[str, str] = {
    "associated": WONG["reddish_purple"],
    "free_head":  WONG["bluish_green"],
    "free_body":  WONG["vermilion"],
    "interval":   "#7F7F7F",
}

FIGURE_DPI: int = 300
FIGURE_SIZE_SINGLE: tuple[float, float] = (3.5, 3.0)
FIGURE_SIZE_DOUBLE: tuple[float, float] = (7.0, 3.0)
FIGURE_SIZE_A4_LANDSCAPE: tuple[float, float] = (11.7, 8.3)
SVG_PARAMS: dict[str, object] = {"svg.fonttype": "none"}
FONT_SIZE: int = 9
