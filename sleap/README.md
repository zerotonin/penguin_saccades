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
python sleap/build_training_package.py <dlc_project> <video_dir> sleap/penguins_train.pkg.slp
python sleap/make_configs.py sleap/
bash sleap/submit.sh            # on the cluster, hpc profile of local_paths.json
```

`train_track.sh` trains both models, tracks every clip and writes
`predictions/<rec>.analysis.h5`. Point `closed_sleap_dir` in
`local_paths.json` at that folder and `penguin-saccades ingest` uses
the longest track per clip as the focal bird. Proofread the
`*.predictions.slp` files in the SLEAP GUI before trusting them.

What is and is not in git: the configs and scripts are tracked; the
training package (embeds labelled frames), trained models and
predictions are data and belong in the Zenodo data deposit, not here.

After the job, pull `predictions/` back and build the proofreading project:

```bash
python sleap/collect_predictions.py <predictions_dir> <video_dir> penguins_proofread.slp --labels sleap/penguins_train.pkg.slp
python -m sleap.gui.app penguins_proofread.slp
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
