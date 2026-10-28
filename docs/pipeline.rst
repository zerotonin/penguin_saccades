Pipeline
========

The analysis is a chain of independently re-runnable stages with CSV
intermediates in the configured ``output_dir``::

    penguin-saccades ingest     # trajectories.csv   (both data sets, one row per frame)
    penguin-saccades saccades   # saccades.csv, intervals.csv
    penguin-saccades isi        # isi_content.csv, saccade_budget.csv, head_body_compensation.csv, isi_chunks.csv
    penguin-saccades stats      # summary_*.csv, pairwise_*.csv
    penguin-saccades plots      # figures/ (SVG + PNG + CSV companions)

``penguin-saccades all`` runs every stage in order.

Every threshold, frame rate, calibration constant and smoothing width
lives in :mod:`penguin_saccades.constants`.  Machine-specific paths live
in the gitignored ``local_paths.json`` (copy ``local_paths.template.json``).
