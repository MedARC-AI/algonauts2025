#!/bin/bash

export CUDA_VISIBLE_DEVICES=4

now="2507221407"
out_prefix="output/feature_encoding_ablations_${now}"

kernel_names=(
    default
    causal
    positive
    blockwise
)
causal_kernel=(
    false
    true
    false
    false
)
positive_kernel=(
    false
    false
    true
    false
)
blockwise_kernel=(
    false
    false
    false
    true
)
for ii in 0 1 2 3; do
    name=${kernel_names[ii]}
    echo "ablate kernel type: ${name}"
    out_dir="${out_prefix}/kernel_type/${name}"
    uv run python train_feature_encoder.py \
        --overrides \
        "model.encoder_causal=${causal_kernel[ii]}" \
        "model.encoder_positive=${positive_kernel[ii]}" \
        "model.encoder_blockwise=${blockwise_kernel[ii]}" \
        "out_dir=${out_dir}"
done
