#!/bin/bash
# Export the hpc profile of local_paths.json and submit train_track.sh.
# Usage: bash sleap/submit.sh [extra sbatch args]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
eval "$(python -m penguin_saccades.config --export --profile hpc)"
mkdir -p "$PENGUIN_SACCADES_SLEAP_DIR/logs"
sbatch --account="$PENGUIN_SACCADES_ACCOUNT" --partition="$PENGUIN_SACCADES_GPU_PARTITIONS" \
    --output="$PENGUIN_SACCADES_SLEAP_DIR/logs/train_track_%j.out" --export=ALL "$@" "$HERE/train_track.sh"
