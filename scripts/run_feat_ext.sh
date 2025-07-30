#!/bin/bash
# Script to run all feature encoding models for Algonauts 2025

set -e

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/.."

# Configuration
LOG_DIR="$PROJECT_ROOT/scripts/logs"

# Create log directory
mkdir -p "$LOG_DIR"

# Define scripts and their corresponding configs
declare -a scripts=(
    "feature_extraction/internvl3_video_features.py"
    "feature_extraction/qwen_video_audio_features.py"
    "feature_extraction/vjepa2_video_features.py"
    "feature_extraction/whisper_audio_features.py"
)

declare -a configs=(
    "config/default_internvl3_features.yaml"
    "config/default_qwen_features.yaml"
    "config/default_vjepa2_features.yaml"
    "config/default_whisper_features.yaml"
)

echo "Found ${#scripts[@]} feature extraction scripts to run."

# Process each script and config pair
for i in "${!scripts[@]}"; do
    script="${PROJECT_ROOT}/${scripts[$i]}"
    config_file="${PROJECT_ROOT}/${configs[$i]}"
    config_name=$(basename "$config_file" .yaml)

    echo "[$((i+1))/${#scripts[@]}] Processing: $config_name"

    # Extract out_dir from the config file. It might be relative or absolute.
    out_dir_raw=$(grep "out_dir:" "$config_file" | awk '{print $2}')

    # If out_dir is not an absolute path, prepend the project root
    if [[ "$out_dir_raw" != /* ]]; then
        out_dir="$PROJECT_ROOT/$out_dir_raw"
    else
        out_dir="$out_dir_raw"
    fi

    if [[ -d $out_dir ]]; then
        echo "Output dir: ${out_dir} exists; skipping"
        continue
    fi

    # Run the feature extraction script and save logs
    uv run python "$script" \
        --cfg-path "$config_file" \
        | tee "$LOG_DIR/${config_name}.log"

    echo "Completed: $config_name"
done

echo "All feature extractions processed. Logs saved to: $LOG_DIR"