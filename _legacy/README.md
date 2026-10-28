# Legacy scripts (parked, not imported)

Verbatim copies of the penguin analysis scripts from the DALLAS repository
(`zerotonin/dallas-dlc-seperate-multi-animal-analysis`, branch `Penguin_ana`,
commit 470d688, 2025-03-24).  They produced the numbers in the June 2025
manuscript draft and are kept for provenance only.  They are excluded from
linting and are not installed with the package.

Known defects that the rewrite in `src/penguin_saccades` corrects:

- `peguin_ana01.py` smoothed with `gaussian_filter1d(order=1)`, i.e. a
  derivative, not a low-pass; translational speed used dx twice.
- `PINGU_unified_saccade_analysis.py` read `*_yaw_deg` columns that the
  ingestion never wrote and detected no saccades.
- `SaccadeAnalysis.py` converted amplitude from radians to degrees twice,
  disabling the minimum-amplitude artefact filter.
- Three detectors with thresholds 180, 200 and 250 deg/s coexisted.
- Intersaccadic intervals were computed across movie and segment boundaries.
- Data set 2 angles were low-passed at 1 Hz without DeepLabCut likelihood gating.

Delete this directory once the rerun has been reconciled with the draft.

Absolute data paths in these scripts were replaced by `/path/to/...`
placeholders when the copies were made; nothing else was edited.
