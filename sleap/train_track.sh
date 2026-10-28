#!/bin/bash
# Train the two SLEAP models and track every closed-pool clip on the cluster.
# All machine-specific values come from local_paths.json (hpc profile) via
# penguin_saccades.config --export; submit with sleap/submit.sh.
#SBATCH --job-name=pengu_sleap
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00

PROJ=${PENGUIN_SACCADES_SLEAP_DIR:?run via sleap/submit.sh so the paths are exported}
VIDEOS=${PENGUIN_SACCADES_CLOSED_VIDEO_DIR:?}
sleep 5
source "${PENGUIN_SACCADES_CONDA_SH:?}"
conda activate "${PENGUIN_SACCADES_SLEAP_ENV:?}"
cd "$PROJ" || exit 1
echo "job $SLURM_JOB_ID on $(hostname) partition $SLURM_JOB_PARTITION start $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import tensorflow as tf; g = tf.config.list_physical_devices('GPU'); print('GPUs:', g); assert g, 'NO GPU VISIBLE'" || exit 1

T0=$(date +%s)
sleap-train centroid.json penguins_train.pkg.slp || exit 1
echo "centroid training took $(( $(date +%s) - T0 )) s"
T0=$(date +%s)
sleap-train centered_instance.json penguins_train.pkg.slp || exit 1
echo "centered_instance training took $(( $(date +%s) - T0 )) s"

mkdir -p predictions
T0=$(date +%s)
for v in "$VIDEOS"/[0-9]*.mp4; do
    rec=$(basename "$v" .mp4)
    case $rec in *DLC*) continue;; esac
    sleap-track "$v" -m models/centroid -m models/centered_instance \
        --tracking.tracker simple --tracking.similarity centroid --tracking.match hungarian \
        --tracking.track_window 5 --tracking.post_connect_single_breaks 1 \
        --peak_threshold 0.2 -o "predictions/${rec}.predictions.slp" || echo "FAILED $rec"
    sleap-convert --format analysis -o "predictions/${rec}.analysis.h5" "predictions/${rec}.predictions.slp" \
        || echo "CONVERT FAILED $rec"
done
echo "tracking took $(( $(date +%s) - T0 )) s"
echo "end $(date)"
