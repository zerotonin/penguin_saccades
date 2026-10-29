# Re-tracking data set 2 with SLEAP

The 2023 DeepLabCut model was single-animal, but two or three penguins
share the closed pool, so its keypoints jumped between birds. This
folder re-tracks the 19 clips with a top-down SLEAP model (centroid
stage separates animals, instance stage poses one bird per crop).

Steps, in the SLEAP conda environment (`sleap=1.4.1` from the `sleap`
conda channel; then `pip install "pynwb>=2.3.3,<2.4"`, and call the
tools as `python -m sleap.nn.training` etc. because the console scripts
abort on a packaging conflict):

```bash
python sleap/build_training_package.py <dlc_project> <video_dir> <closed project dir>/penguins_train.pkg.slp
python sleap/make_configs.py <closed project dir> --preset closed
bash sleap/submit.sh closed     # on the cluster, hpc profile of local_paths.json
```

`train_track.sh` trains both models, tracks every clip and writes
`predictions/<rec>.analysis.h5` into the project directory
(`closed_sleap_project_dir` of the hpc profile, which also names
`conda_sh`, `sleap_env` and `gpu_partitions`). Point `closed_sleap_dir`
in `local_paths.json` at that folder and `penguin-saccades ingest` uses
the longest track per clip as the focal bird. Proofread the
`*.predictions.slp` files in the SLEAP GUI before trusting them.

What is and is not in git: the configs and scripts are tracked; the
training package (embeds labelled frames), trained models and
predictions are data and belong in the Zenodo data deposit, not here.

After the job, pull `predictions/` back and build the proofreading project:

```bash
python sleap/collect_predictions.py <predictions_dir> <video_dir> penguins_proofread.slp --labels sleap/penguins_train.pkg.slp
python sleap/gui.py penguins_proofread.slp
```

## Lessons from the first training run

- The raw clips have broken seek indices: OpenCV 4.2 (SLEAP's reader)
  decodes 32 frames fewer than newer builds and cannot seek past them,
  which silently dropped 47 labels and crashed tracking.  Re-encode the
  clips intra-only before anything else (`ffmpeg -vsync passthrough
  -c:v libx264 -crf 15 -x264-params keyint=1:min-keyint=1`); label-to-frame
  alignment was verified pixel-wise on the re-encoded clips.
- A centred-instance UNet with max stride 16 never learned the pose: a
  gentoo spans ~200 px, beyond the receptive field, so beak and tail
  were indistinguishable and only small, distant birds were detected.
  Stride 32, a 320 px crop, 32 filters and sigma 5 fix that.
- No contrast augmentation.  SLEAP 1.4.1 hands its gamma range (0.7 to
  1.3) to albumentations, which expects percent, so every crop is bleached
  to near white and only the crop centre (the anchor node) is learnable.
  Diagnosed by dumping crops and targets from `trainer.training_pipeline`
  with each augmentation switched on alone.
- No horizontal flip augmentation: the skeleton defines no fin symmetry,
  so flips contradict the left/right fin labels.
- `sleap-train`/`sleap-track` abort on a pkg_resources conflict in the
  1.4.1 conda package, and `python -m sleap.nn.inference` has no main
  guard.  Train with `python -m sleap.nn.training`, track with
  `track_videos.py`.

## Open-water movies (data set 1): a second SLEAP project

The closed-pool model does not transfer (camera above the pool, birds
150 to 250 px long, low contrast), so the open-water movies get their own
labels.  The curated charon trajectories seed the suggested frames, which
keeps the labeller on frames that hold a bird:

```bash
python sleap/export_open_boxes.py <sorted_and_filtered dir> open_boxes.csv          # analysis env
python sleap/build_open_labelling_project.py <open movies dir> open_water_labels.slp \
    --n-frames 200 --exclude 04-03 --boxes-csv open_boxes.csv                        # sleap env
python sleap/gui.py open_water_labels.slp
```

The 04-03-2021 avi is a damaged MPEG-4 file (decode errors, empty frames on
seek) and the raw `.tra` detections are mostly false positives at the pool
edge; both are left out on purpose.  Label beak, neck, left fin, right fin
and tail; the skeleton carries the fin symmetry so flips are allowed here.

Once labelled (190 frames, 162 with all five points, were enough), export
the package, write the open-water configs and submit the same job script
for the second data set:

```bash
python sleap/export_training_package.py open_water_labels.slp <open project dir>/penguins_train.pkg.slp
python sleap/make_configs.py <open project dir> --preset open
bash sleap/submit.sh open       # tracks Gentoo_*.mp4 with at most three tracks
```

The `open` preset keeps the half-resolution centroid stage (a bird is
~100 px at that scale in the 2704 x 1520 frame) and widens the instance
crop to 384 px because neck to tail reaches 180 px.  The track cap keeps
the analysis export small on 45 000-frame movies in which the bird dives
and resurfaces; the ingest takes the best-scoring instance per frame
anyway.  Point `open_sleap_dir` at the predictions and
`penguin-saccades ingest` derives body and head yaw from the skeleton
(`process_open_pose_trajectory`), rescaling movie pixels to the calibrated
1402 x 788 frame and dropping birds outside the water ROI, instead of
heading from displacement.
