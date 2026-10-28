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
| `output_dir`            | where CSV intermediates, tables and figures are written   |

`local_paths.json` is gitignored. The environment variable
`PENGUIN_SACCADES_DATA_ROOT` overrides `data_root`, and
`PENGUIN_SACCADES_PROFILE` selects a profile (`local` or `hpc`). A
missing paths file stops every entry point with a message naming the
template.

## Quickstart

```bash
penguin-saccades all            # ingest -> saccades -> isi -> stats -> plots
penguin-saccades saccades       # rerun one stage after changing a constant
```

Each stage reads the CSV written by its predecessor in `output_dir`:

| stage      | writes                                                                  |
|------------|-------------------------------------------------------------------------|
| `ingest`   | `trajectories.csv` (one row per frame, both data sets, shared schema)   |
| `saccades` | `saccades.csv` (body and head, typed), `intervals.csv` (body/head/any)  |
| `isi`      | `saccade_content.csv`, `interval_content.csv`, `saccade_budget.csv`, `head_body_compensation.csv`, `isi_chunks.csv` |
| `stats`    | `summary_tables.csv` (n, median, bootstrap CI), `pairwise_tests.csv` (Fisher resampling, BH-FDR) |
| `plots`    | `figures/*.svg|png|csv`                                                 |

Every analysis constant lives in `src/penguin_saccades/constants.py`.

## Parameters to ratify before the rerun

These values were inconsistent between the legacy scripts and the
manuscript. Each is a single constant; change it there and rerun.

| constant                             | current value | note                                             |
|--------------------------------------|---------------|--------------------------------------------------|
| `DATASET_OPEN.frame_rate_hz`         | 30            | legacy code used 30, manuscript says 60          |
| `SMOOTHING_SIGMA_S`                  | 0.05 s        | manuscript: 3-frame Gaussian at 60 fps           |
| `SACCADE_VELOCITY_THRESHOLD_DEGPS`   | 200           | legacy scripts used 180, 200 and 250             |
| `ASSOCIATION_WINDOW_S`               | 1.0 s         | head-body pairing window, not stated in legacy   |
| `GENTOO_BODY_LENGTH_M`               | 0.80 m        | scale for data set 2                             |
| `DLC_LIKELIHOOD_THRESHOLD`           | 0.75          | legacy code had no gating                        |

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
