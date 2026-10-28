# Re-tracking data set 2 with SLEAP

The 2023 DeepLabCut model was single-animal, but two or three penguins
share the closed pool, so its keypoints jumped between birds. This
folder re-tracks the 19 clips with a top-down SLEAP model (centroid
stage separates animals, instance stage poses one bird per crop).

Steps, in the SLEAP conda environment (`sleap=1.4.1` from the `sleap`
conda channel):

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
