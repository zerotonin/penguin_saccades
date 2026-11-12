# penguin_saccades

[![tests](https://github.com/zerotonin/penguin_saccades/actions/workflows/tests.yml/badge.svg)](https://github.com/zerotonin/penguin_saccades/actions/workflows/tests.yml)
[![docs](https://github.com/zerotonin/penguin_saccades/actions/workflows/docs.yml/badge.svg)](https://github.com/zerotonin/penguin_saccades/actions/workflows/docs.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Saccadic gaze strategy analysis for swimming penguins. One detector,
one smoothing rule and one set of constants for both data sets of the
Oepen, Geurten, Hanke and Sørensen manuscript: body saccades from
overhead trajectories (data set 1, open water) and head plus body
saccades from DeepLabCut poses (data set 2, closed pool).

The package replaces the analysis scripts kept on the `Penguin_ana`
branch of the DALLAS repository. Verbatim copies of those scripts live
in `_legacy/` for provenance; see `_legacy/README.md` for what was
wrong with them and why every number in the manuscript must be
regenerated with this package.

## Installation

```bash
git clone https://github.com/zerotonin/penguin_saccades.git
cd penguin_saccades
conda env create -f environment.yml
conda activate penguin_saccades
pip install -e ".[dev]"
```

**Step one, before running anything:** copy the path template and fill
in where your data live.

```bash
cp local_paths.template.json local_paths.json
```

Keys that matter in `local_paths.json`:

| key                     | content                                                   |
|-------------------------|-----------------------------------------------------------|
| `open_trajectories_dir` | folder of `*.h5` trajectories from charon (data set 1)    |
| `closed_dlc_dir`        | folder of DeepLabCut `*DLC*.csv` or `.h5` files (data set 2) |
| `closed_sleap_dir`      | SLEAP `*.analysis.h5` for data set 2; when present it replaces `closed_dlc_dir` |
| `open_sleap_dir`        | SLEAP `*.analysis.h5` for data set 1; when present it replaces the charon trajectories |
| `output_dir`            | where CSV intermediates, tables and figures are written   |

`local_paths.json` is gitignored. The environment variable
`PENGUIN_SACCADES_DATA_ROOT` overrides `data_root`, and
`PENGUIN_SACCADES_PROFILE` selects a profile (`local` or `hpc`). A
missing paths file stops every entry point with a message naming the
template.

## Quickstart

```bash
penguin-saccades all            # ingest -> saccades -> qc -> isi -> stats -> plots -> manuscript
penguin-saccades saccades       # rerun one stage after changing a constant
```

Each stage reads the CSV written by its predecessor in `output_dir`:

| stage      | writes                                                                  |
|------------|-------------------------------------------------------------------------|
| `ingest`   | `trajectories.csv` (one row per frame, both data sets, shared schema)   |
| `saccades` | `saccades.csv` (body and head, typed), `intervals.csv` (body/head/any)  |
| `qc`       | `qc/qc_frame_budget` (frames per movie through every gate of the pose route), `qc_keypoint_scores`, `qc_yaw_jitter.csv`, `qc_noise_floor` (yaw velocity under the thresholds), `qc_saccade_rejections` (which rule drops each raw peak), `qc_detection_timeline` (detected, usable and bridged frames per movie) |
| `isi`      | `saccade_content.csv`, `interval_content.csv`, `saccade_budget.csv`, `head_body_compensation.csv`, `isi_chunks.csv` |
| `stats`    | `summary_tables.csv` (n, median, bootstrap CI), `pairwise_tests.csv` (Fisher resampling, BH-FDR) |
| `plots`    | `figures/*.svg|png|csv`                                                 |
| `manuscript` | `manuscript/table_1.csv`, `manuscript/fig{1e,2,3,4,5}_*` and `supp1_*` as SVG, PNG and CSV; every figure is species-based and pools both cameras (marker shape = camera, 25 fps traces resampled onto the 60 fps grid, one pair of y-limits for all triggered averages) |

Every analysis constant lives in `src/penguin_saccades/constants.py`.

To see what the detector and the gates did on a stretch of movie, render
an overlay clip (needs the `video` extra, `pip install -e ".[video]"`):

```bash
penguin-saccades-overlay <movie.mp4> <open_sleap_dir>/<movie>.analysis.h5 \
    --start 4500 --end 4760 --out overlays/<movie>_f4500-4760.mp4
```

The clip follows the bird at a third of real time, draws every SLEAP
instance (the one the ingest keeps in colour, low-score keypoints
hollow), and badges each frame with the pipeline's verdict from the
profile's `output_dir`: usable, bridged or rejected, plus the saccade it
lies in. A PNG contact sheet of eight frames is written next to the mp4.

## Parameters to ratify before the rerun

These values were inconsistent between the legacy scripts and the
manuscript. Each is a single constant; change it there and rerun.

| constant                             | current value | note                                             |
|--------------------------------------|---------------|--------------------------------------------------|
| `DATASET_OPEN.frame_rate_hz`         | 59.94         | read from the GoPro files; legacy code used 30   |
| `DATASET_OPEN.smoothing_sigma_s`     | 0.10 s        | from the smoothing sweep (results/qc)            |
| `OPEN_POSE_SMOOTHING_SIGMA_S`        | 0.05 s        | data set 1 from SLEAP poses; head saccade count is 343 / 144 / 22 at 0.03 / 0.05 / 0.10 s, so this needs a decision |
| `DATASET_CLOSED.smoothing_sigma_s`   | 0.04 s        | one frame at 25 fps; 0.10 s erases head saccades |
| `MAX_HEAD_BODY_ANGLE_DEG`            | 60            | skeleton gate against beak-tail swaps and misplaced beaks; gentoo open-water frames have p95 at 50 |
| `MAX_BRIDGED_WINDOW_SHARE`           | 0.5           | a saccade whose triggered window is more than half bridged is dropped |
| `MAX_SACCADE_SPEED_DEGPS`            | 500           | no gentoo peak exceeds 475 deg/s; faster rockhopper peaks are whole-skeleton reversals (legacy 3000) |
| `MAX_GAP_FRAMES`                     | 10            | ratified: keypoint and detection gaps up to 10 frames are bridged, longer ones split segments |
| `SACCADE_VELOCITY_THRESHOLDS_DEGPS`  | head 200, body 100 | ratified: the body yaw velocity has no saccade mode, 100 deg/s is where it leaves the head distribution (legacy 180/200/250 for both) |
| `ASSOCIATION_WINDOW_S`               | 1.0 s         | head-body pairing window, not stated in legacy   |
| `GENTOO_BODY_LENGTH_M`               | 0.80 m        | scale for data set 2                             |
| `DLC_LIKELIHOOD_THRESHOLD`           | 0.75          | legacy code had no gating                        |
| `SLEAP_SCORE_THRESHOLD` | 0.3 | SLEAP point scores are peak heights (median 0.65 on the re-tracked clips), not probabilities |

## Development

```bash
ruff check .
pytest
sphinx-build -b html docs/ docs/_build/html
```

The test suite runs on synthetic penguins with known headings, speeds
and saccade amplitudes, so a regression in any kinematic step fails a
test rather than silently changing a table.

## Citation

See `CITATION.cff`. Statistics use
[reRandomStats](https://github.com/zerotonin/reRandomStats).
