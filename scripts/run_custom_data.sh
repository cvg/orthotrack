#!/bin/bash
# run_custom_data.sh
# 
# Helper script to run OrthoTrack on custom UAV footage and custom GeoTIFF maps.
#
# Usage:
#   bash scripts/run_custom_data.sh <path_to_video> <path_to_dop_tif> <path_to_dsm_tif> <output_dir>
#

if [ "$#" -lt 4 ]; then
    echo "Usage: bash scripts/run_custom_data.sh <video> <dop_tif> <dsm_tif> <output_dir>"
    echo "Example: bash scripts/run_custom_data.sh my_flight.mp4 ortho.tif dem.tif outputs/my_flight"
    exit 1
fi

VIDEO=$1
DOP=$2
DSM=$3
OUTPUT=$4

echo "============================================================"
echo "Running OrthoTrack on Custom Data"
echo "============================================================"
echo "Video: $VIDEO"
echo "DOP:   $DOP"
echo "DSM:   $DSM"
echo "Out:   $OUTPUT"
echo "============================================================"

# Ensure PYTHONPATH is set so local modules resolve correctly
export PYTHONPATH=.

# Default matcher is turbo (~8 GB VRAM). Override with e.g. --fine_matcher precise
python scripts/run_tracking.py \
    --footage "$VIDEO" \
    --dop "$DOP" \
    --dsm "$DSM" \
    --output "$OUTPUT" \
    --fine_matcher turbo \
    --save_keyframe_vis \
    "${@:5}"
