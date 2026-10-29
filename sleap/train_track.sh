#!/bin/bash
# Train the two SLEAP models and track every movie of one data set.
# submit.sh exports the project folder (package, configs, track_videos.py),
# the movie folder and the conda setup; the same script serves the
# closed-pool clips and the open-water movies.
# The sleap-* console scripts are avoided on purpose: the 1.4.1 conda
# package ships a pkg_resources dependency conflict that aborts them at
# import, while python -m skips that check.
#SBATCH --job-name=pengu_sleap
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00

PROJ=${PENGUIN_SLEAP_PROJECT_DIR:?run via sleap/submit.sh so the paths are exported}
VIDEOS=${PENGUIN_SLEAP_VIDEO_DIR:?}
PATTERN=${PENGUIN_SLEAP_VIDEO_PATTERN:-[0-9]*.mp4}
TRACK_ARGS=${PENGUIN_SLEAP_TRACK_ARGS:-}
sleep 5
source "${PENGUIN_SLEAP_CONDA_SH:?}"
conda activate "${PENGUIN_SLEAP_ENV:?}"
cd "$PROJ" || exit 1
echo "job $SLURM_JOB_ID on $(hostname) partition $SLURM_JOB_PARTITION start $(date)"
echo "project $PROJ movies $VIDEOS/$PATTERN track args: $TRACK_ARGS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import tensorflow as tf; g = tf.config.list_physical_devices('GPU'); print('GPUs:', g); assert g, 'NO GPU VISIBLE'" || exit 1

T0=$(date +%s)
[ -f models/centroid/best_model.h5 ] || python -m sleap.nn.training centroid.json penguins_train.pkg.slp || exit 1
echo "centroid training took $(( $(date +%s) - T0 )) s"
T0=$(date +%s)
[ -f models/centered_instance/best_model.h5 ] || python -m sleap.nn.training centered_instance.json penguins_train.pkg.slp || exit 1
echo "centered_instance training took $(( $(date +%s) - T0 )) s"

T0=$(date +%s)
# shellcheck disable=SC2086
python track_videos.py "$VIDEOS" models predictions --pattern "$PATTERN" $TRACK_ARGS || echo "TRACK FAILED"
echo "tracking took $(( $(date +%s) - T0 )) s"
echo "end $(date)"
