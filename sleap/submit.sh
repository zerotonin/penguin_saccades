#!/bin/bash
# Export the hpc profile of local_paths.json and submit train_track.sh for one data set.
# Usage: bash sleap/submit.sh closed|open [extra sbatch args]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DATASET=${1:?usage: submit.sh closed|open [sbatch args]}
shift
eval "$(python -m penguin_saccades.config --export --profile hpc)"
case "$DATASET" in
    closed)
        PROJECT=$PENGUIN_SACCADES_CLOSED_SLEAP_PROJECT_DIR
        VIDEOS=$PENGUIN_SACCADES_CLOSED_VIDEO_DIR
        PATTERN='[0-9]*.mp4'
        TRACK_ARGS=''
        ;;
    open)
        PROJECT=$PENGUIN_SACCADES_OPEN_SLEAP_PROJECT_DIR
        VIDEOS=$PENGUIN_SACCADES_OPEN_VIDEO_DIR
        PATTERN='Gentoo_*.mp4'
        TRACK_ARGS='--max-tracks 3'    # one bird, sometimes two; dives break tracks
        ;;
    *)
        echo "unknown data set: $DATASET (closed|open)" >&2
        exit 1
        ;;
esac
mkdir -p "$PROJECT/logs"
cp "$HERE/train_track.sh" "$HERE/track_videos.py" "$PROJECT/"
sbatch --account="$PENGUIN_SACCADES_ACCOUNT" --partition="$PENGUIN_SACCADES_GPU_PARTITIONS" \
    --job-name="pengu_sleap_$DATASET" \
    --output="$PROJECT/logs/train_track_%j.out" \
    --export=ALL,PENGUIN_SLEAP_PROJECT_DIR="$PROJECT",PENGUIN_SLEAP_VIDEO_DIR="$VIDEOS",PENGUIN_SLEAP_VIDEO_PATTERN="$PATTERN",PENGUIN_SLEAP_TRACK_ARGS="$TRACK_ARGS",PENGUIN_SLEAP_CONDA_SH="$PENGUIN_SACCADES_CONDA_SH",PENGUIN_SLEAP_ENV="$PENGUIN_SACCADES_SLEAP_ENV" \
    "$@" "$PROJECT/train_track.sh"
