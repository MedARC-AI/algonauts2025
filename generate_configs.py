#!/usr/bin/env python3
"""
Generate multiple config files with different hyperparameter combinations
for the Algonauts 2025 feature encoding model.
"""

import yaml
import os
import itertools


def load_base_config(config_path):
    """Load the base configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_config_variations():
    """Define all the variations you want to test."""

    # Define hyperparameter variations
    variations = {
        "lr": [1e-4, 3e-4, 1e-3],
        "weight_decay": [0.01, 0.1, 0.3],
        "encoder_kernel_size": [11, 33, 45, 65],
        "num_samples": [1000, 2000, 4000],
        "sample_length": [32, 64, 128],
        "batch_size": [8, 16, 32],
        "epochs": [10, 15, 20],
        "embed_dim": [128, 192, 256],
        "transformer_depth": [4, 6, 8],
        "pool_num_heads": [2, 3, 4],
    }

    # Define feature combinations to test
    feature_combinations = [
        # Single features
        ["llama_3.2_3B/layers.11"],
        ["whisper/layers.12"],
        ["qwen-2-5-omni-3b/layers.20"],
        ["internvl3_8b/layers.20"],
        ["vjepa2/encoder.layernorm_avg"],
        ["qwen-2-5-omni-7b/layers.20"],
        # Pairs
        ["llama_3.2_3B/layers.11", "whisper/layers.12"],
        ["qwen-2-5-omni-3b/layers.20", "internvl3_8b/layers.20"],
        ["whisper/layers.12", "vjepa2/encoder.layernorm_avg"],
        # Triples
        ["llama_3.2_3B/layers.11", "whisper/layers.12", "qwen-2-5-omni-3b/layers.20"],
        ["internvl3_8b/layers.20", "vjepa2/encoder.layernorm_avg", "whisper/layers.12"],
        # All features (original)
        [
            "llama_3.2_3B/layers.11",
            "whisper/layers.12",
            "qwen-2-5-omni-3b/layers.20",
            "internvl3_8b/layers.20",
            "vjepa2/encoder.layernorm_avg",
        ],
        # All features (new)
        [
            "llama_3.2_3B/layers.11",
            "whisper/layers.12",
            "qwen-2-5-omni-3b/layers.20",
            "qwen-2-5-omni-7b/layers.20",
            "internvl3_8b/layers.20",
            "vjepa2/encoder.layernorm_avg",
        ],
    ]

    return variations, feature_combinations


def create_config_name(base_name, params, feature_suffix):
    """Create a descriptive config name."""
    param_str = "_".join([f"{k}{v}" for k, v in params.items()])
    return f"{base_name}_{param_str}_{feature_suffix}"


def modify_config(base_config, params, features):
    """Modify the base config with new parameters."""
    config = base_config.copy()

    # Update hyperparameters
    for key, value in params.items():
        if key == "transformer_depth":
            config["transformer"]["depth"] = value
        elif key in ["lr", "weight_decay", "batch_size", "epochs"]:
            config[key] = value
        elif key == "num_samples":
            config["datasets"]["train"]["num_samples"] = value
        elif key == "sample_length":
            config["datasets"]["train"]["sample_length"] = value
        elif key == "encoder_kernel_size":
            config["model"]["encoder_kernel_size"] = value
        elif key == "embed_dim":
            config["model"]["embed_dim"] = value
        elif key == "pool_num_heads":
            config["model"]["pool_num_heads"] = value

    # Update features
    config["include_features"] = features

    # Update output directory to be unique
    feature_suffix = "_".join([f.split("/")[-1] for f in features])
    param_suffix = "_".join([f"{k}{v}" for k, v in params.items()])
    config["out_dir"] = (
        f"output_ensemble/feature_encoding_{param_suffix}_{feature_suffix}"
    )

    return config


def generate_all_configs(base_config_path, output_dir, max_configs=None):
    """Generate all config combinations."""

    # Load base config
    base_config = load_base_config(base_config_path)

    # Get variations
    variations, feature_combinations = generate_config_variations()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate configs
    config_count = 0
    generated_configs = []

    # You can choose to do full grid search or random sampling
    # For demonstration, let's do a subset of combinations

    # Method 1: Full grid search (warning: can generate A LOT of configs)
    if max_configs is None:
        # Generate all possible combinations
        param_keys = list(variations.keys())
        param_values = list(variations.values())

        for param_combo in itertools.product(*param_values):
            params = dict(zip(param_keys, param_combo))

            for features in feature_combinations:
                config = modify_config(base_config, params, features)

                # Create feature suffix for naming
                feature_suffix = "_".join([f.split("/")[-1] for f in features])
                config_name = create_config_name("feat_enc", params, feature_suffix)

                # Save config
                config_path = os.path.join(output_dir, f"{config_name}.yaml")
                with open(config_path, "w") as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

                generated_configs.append(config_path)
                config_count += 1

                print(f"Generated config {config_count}: {config_name}")

    else:
        # Method 2: Random sampling of combinations
        import random

        param_keys = list(variations.keys())
        param_values = list(variations.values())

        all_combinations = list(itertools.product(*param_values))

        # Sample random combinations
        sampled_combinations = random.sample(
            all_combinations,
            min(max_configs // len(feature_combinations), len(all_combinations)),
        )

        for param_combo in sampled_combinations:
            params = dict(zip(param_keys, param_combo))

            for features in feature_combinations:
                if config_count >= max_configs:
                    break

                config = modify_config(base_config, params, features)

                # Create feature suffix for naming
                feature_suffix = "_".join([f.split("/")[-1] for f in features])
                config_name = create_config_name("feat_enc", params, feature_suffix)

                # Save config
                config_path = os.path.join(output_dir, f"{config_name}.yaml")
                with open(config_path, "w") as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

                generated_configs.append(config_path)
                config_count += 1

                print(f"Generated config {config_count}: {config_name}")

            if config_count >= max_configs:
                break

    print(f"\nTotal configs generated: {config_count}")
    return generated_configs


def main():
    base_config_path = (
        "/home/cesar/algo_sub/algonauts2025/config/default_feature_encoding.yaml"
    )
    output_dir = "config_ensemble"

    # Generate configs - set max_configs to limit the number
    # Remove max_configs parameter for full grid search (warning: many configs!)
    configs = generate_all_configs(base_config_path, output_dir, max_configs=140)

    print(f"\nConfigs saved to: {output_dir}")
    print(f"Total configs: {len(configs)}")


if __name__ == "__main__":
    main()
