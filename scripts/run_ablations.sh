#!/bin/bash

export CUDA_VISIBLE_DEVICES=4

now="2507221407"
out_prefix="output/feature_encoding_ablations_${now}"

# kernel size: [65, 45, 17, 9, 0]
# embedding dim: [256, 192, 128, 64, 32]
# decoders:
#   - shared=true, subject=true
#   - shared=false, subject=true
#   - shared=true, subject=false
# features:
#   - [llama_3.2_3b/layers.11,whisper/layers.12,qwen-2-5-omni-3b/layers.20,internvl3_14b/layers.30,vjepa2/norm]
#   - [whisper/layers.12,qwen-2-5-omni-3b/layers.20,internvl3_14b/layers.30,vjepa2/norm]
#   - [llama_3.2_3b/layers.11,qwen-2-5-omni-3b/layers.20,internvl3_14b/layers.30,vjepa2/norm]
#   - [llama_3.2_3b/layers.11,whisper/layers.12,internvl3_14b/layers.30,vjepa2/norm]
#   - [llama_3.2_3b/layers.11,whisper/layers.12,qwen-2-5-omni-3b/layers.20,vjepa2/norm]
#   - [llama_3.2_3b/layers.11,whisper/layers.12,qwen-2-5-omni-3b/layers.20,internvl3_14b/layers.30]
#   - [llama_3.2_3b/layers.11,whisper/layers.12,vjepa2/norm]
#   - [llama_3.2_3b/layers.11]
#   - [whisper/layers.12]
#   - [vjepa2/norm]
# subjects: [all,single subject]

kernel_sizes="65 45 17 9 0"
for val in $kernel_sizes; do
    echo "ablate kernel size: $val"
    out_dir="${out_prefix}/kernel_size/kernel-${val}"
    uv run python train_feature_encoder.py \
        --overrides \
        "model.encoder_kernel_size=${val}" \
        "out_dir=${out_dir}"
done

embed_dims="256 192 128 64 32"
for val in $embed_dims; do
    echo "ablate embed dim: $val"
    out_dir="${out_prefix}/embed_dim/dim-${val}"
    uv run python train_feature_encoder.py \
        --overrides \
        "model.embed_dim=${val}" \
        "out_dir=${out_dir}"
done


with_shared_decoder=( true true false )
with_subject_decoders=( true false true )
for ii in 0 1 2; do
    val1=${with_shared_decoder[ii]}
    val2=${with_subject_decoders[ii]}
    echo "ablate arch: shared=${val1} subject=${val2}"
    out_dir="${out_prefix}/arch/shared-${val1}_subject-${val2}"
    uv run python train_feature_encoder.py \
        --overrides \
        "model.with_shared_decoder=${val1}" \
        "model.with_subject_decoders=${val2}" \
        "out_dir=${out_dir}"
done


feature_group_names=(
    # all features
    llama-1_whisper-1_qwen-1_internvl-1_vjepa-1
    # leave one out
    llama-0_whisper-1_qwen-1_internvl-1_vjepa-1
    llama-1_whisper-0_qwen-1_internvl-1_vjepa-1
    llama-1_whisper-1_qwen-0_internvl-1_vjepa-1
    llama-1_whisper-1_qwen-1_internvl-0_vjepa-1
    llama-1_whisper-1_qwen-1_internvl-1_vjepa-0
    # leave all but one out
    llama-1_whisper-0_qwen-0_internvl-0_vjepa-0
    llama-0_whisper-1_qwen-0_internvl-0_vjepa-0
    llama-0_whisper-0_qwen-1_internvl-0_vjepa-0
    llama-0_whisper-0_qwen-0_internvl-1_vjepa-0
    llama-0_whisper-0_qwen-0_internvl-0_vjepa-1
)
llama="llama_3.2_3b/layers.11"
whisper="whisper/layers.12"
qwen="qwen-2-5-omni-3b/layers.20"
internvl="internvl3_14b/layers.30"
vjepa="vjepa2/norm"
feature_groups=(
    # all features
    "[${llama},${whisper},${qwen},${internvl},${vjepa}]"
    # leave one out
    "[${whisper},${qwen},${internvl},${vjepa}]"
    "[${llama},${qwen},${internvl},${vjepa}]"
    "[${llama},${whisper},${internvl},${vjepa}]"
    "[${llama},${whisper},${qwen},${vjepa}]"
    "[${llama},${whisper},${qwen},${internvl}]"
    # leave all but one out
    "[${llama}]"
    "[${whisper}]"
    "[${qwen}]"
    "[${internvl}]"
    "[${vjepa}]"
)

for ii in {0..10}; do
    feat_group_name=${feature_group_names[ii]}
    feat_group=${feature_groups[ii]}
    echo "ablate features: features=${feat_group_name}"
    out_dir="${out_prefix}/features/${feat_group_name}"
    uv run python train_feature_encoder.py \
        --overrides \
        "include_features=${feat_group}" \
        "out_dir=${out_dir}"
done

subjects=(
    "1,2,3,5"
    "1"
    "2"
    "3"
    "5"
)
for ii in {0..4}; do
    subs="${subjects[ii]}"
    echo "ablate multi-sub: subs=${subs}"
    subs_fmt=$(echo "$subs" | tr "," "_")
    out_dir="${out_prefix}/multi_sub/subs-${subs_fmt}"
    uv run python train_feature_encoder.py \
        --overrides \
        "subjects=[${subs}]" \
        "out_dir=${out_dir}"
done

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
