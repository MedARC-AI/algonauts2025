#!/bin/bash
# Simplified script to run feature encoding model for Algonauts 2025

set -e

# Configuration
VENV_PATH="/home/cesar/algo_sub/algonauts2025/.venv/bin/activate"
CONFIGS_PATH="/home/cesar/algo_sub/algonauts2025/config_ensemble"
LOG_DIR="/home/cesar/algo_sub/algonauts2025/config_ensemble/logs"

# Activate virtual environment
source "$VENV_PATH"

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

    # Run training and save logs
    uv run python train_and_inference_ood.py \
        --cfg-path "$config_file" \
        --run_ood_prediction \
        > "$LOG_DIR/${config_name}.log" 2> "$LOG_DIR/${config_name}.error"

    echo "Completed: $config_name"
done

echo "All configs processed. Logs saved to: $LOG_DIR"
