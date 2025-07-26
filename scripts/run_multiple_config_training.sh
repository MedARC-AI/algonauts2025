#!/bin/bash
# Simplified script to run feature encoding model for Algonauts 2025

set -e

# Configuration
CONFIGS_PATH="config_ensemble"
LOG_DIR="config_ensemble/logs"

# Create log directory
mkdir -p "$LOG_DIR"

# Find all config files
config_files=($(find "$CONFIGS_PATH" -name "*.yaml"))

echo "Found ${#config_files[@]} configuration files"

# Process each config
for i in "${!config_files[@]}"; do
    config_file="${config_files[$i]}"
    config_name=$(basename "$config_file" .yaml)

    echo "[$((i+1))/${#config_files[@]}] Processing: $config_name"
    out_dir=$(cat $config_file | grep out_dir | awk '{print $2}')

    if [[ -d $out_dir ]]; then
        echo "Output dir: ${out_dir} exists; skipping"
        continue
    fi

    # Run training and save logs
    uv run python train_feature_encoder.py \
        --cfg-path "$config_file" \
        | tee "$LOG_DIR/${config_name}.log"

    uv run python submit_feature_encoder.py \
        --overrides \
        checkpoint_dir="${out_dir}" \
        test_set_name=ood \
        | tee -a "$LOG_DIR/${config_name}.log"

    echo "Completed: $config_name"
done

echo "All configs processed. Logs saved to: $LOG_DIR"
