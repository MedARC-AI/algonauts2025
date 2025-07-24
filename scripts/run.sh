export OMP_NUM_THREADS=8

export CUDA_VISIBLE_DEVICES=7

now=$(TZ=America/New_York date '+%y%m%d%H%M')

# uv run python train_feature_encoder.py \
#     --overrides \
#     out_dir=output/feature_encoding \
#     overwrite=true \
#     checkpoint=output/cross_encoding/ckpt.pt \
#     freeze_decoder=true \

# uv run python train_feature_encoder.py \
#     --overrides \
#     out_dir=output/feature_encoding_dim-192_hidden-0 \
#     overwrite=true

# uv run python train_feature_encoder.py \
#     --overrides \
#     out_dir=output/feature_encoding_dim-192_hidden-0 \
#     model.encoder_kernel_size=0 \
#     model.with_hidden_model=false \
#     overwrite=true


# uv run python train_cross_encoder.py \
#     --overrides \
#     out_dir="output/cross_encoding_default_${now}" \
#     datasets_root=/dev/shm/algonauts2025 \
#     overwrite=true

# uv run python train_cross_encoder.py \
#     --overrides \
#     out_dir="output/cross_encoding_dim-192_${now}" \
#     datasets_root=/dev/shm/algonauts2025 \
#     model.embed_dim=192 \
#     overwrite=true

# uv run python train_feature_encoder.py \
#     --overrides \
#     out_dir="output/feature_encoding_default_${now}" \
#     datasets_root=/dev/shm/algonauts2025 \
#     overwrite=true

# uv run python train_feature_encoder.py \
#     --overrides \
#     out_dir="output/feature_encoding_no-norm_${now}" \
#     model.encoder_normalize=false \
#     datasets_root=/dev/shm/algonauts2025 \
#     overwrite=true

# uv run python train_feature_encoder.py \
#     --overrides \
#     out_dir="output/feature_encoding_no-causal_${now}" \
#     model.encoder_causal=false \
#     datasets_root=/dev/shm/algonauts2025

# uv run python train_feature_encoder.py \
#     --overrides \
#     out_dir="output/feature_encoding_pos-kernel_${now}" \
#     model.encoder_positive=true \
#     datasets_root=/dev/shm/algonauts2025

# uv run python train_feature_encoder.py \
#     --overrides \
#     out_dir="output/feature_encoding_dec-kernel-7_${now}" \
#     model.decoder_kernel_size=7 \
#     datasets_root=/dev/shm/algonauts2025


# uv run python train_feature_encoder.py \
#     --overrides \
#     out_dir="output/feature_encoding_hidden-conv1dnext_depth-2_kernel-11_${now}" \
#     model.hidden_model=conv1dnext \
#     conv1dnext.depth=2 \
#     conv1dnext.kernel_size=11 \
#     datasets_root=/dev/shm/algonauts2025


# uv run python train_feature_encoder.py \
#     --overrides \
#     out_dir="output/feature_encoding_hidden-null_${now}" \
#     model.hidden_model=null \
#     datasets_root=/dev/shm/algonauts2025

# uv run python train_feature_encoder.py \
#     --overrides \
#     out_dir="output/feature_encoding_hidden-conv1dnext_depth-6_kernel-11_${now}" \
#     model.hidden_model=conv1dnext \
#     conv1dnext.depth=6 \
#     conv1dnext.kernel_size=11 \
#     datasets_root=/dev/shm/algonauts2025

# uv run python train_feature_encoder.py \
#     --overrides \
#     out_dir="output/feature_encoding_len-64_n-2k_bs-16_${now}" \
#     datasets.train.sample_length=64 \
#     datasets.train.num_samples=2000 \
#     batch_size=16 \
#     datasets_root=/dev/shm/algonauts2025

# uv run python train_feature_encoder.py \
#     --overrides \
#     out_dir="output/feature_encoding_len-64_n-2k_bs-32_${now}" \
#     datasets.train.sample_length=64 \
#     datasets.train.num_samples=2000 \
#     batch_size=32 \
#     datasets_root=/dev/shm/algonauts2025

# uv run python train_feature_encoder.py \
#     --overrides \
#     out_dir="output/feature_encoding_default_${now}" \
#     datasets_root=/dev/shm/algonauts2025

# uv run python submit_feature_encoder.py \
#     --overrides \
#     out_dir=output/feature_encoding_default_2506271614 \
#     datasets_root=/dev/shm/algonauts2025

# uv run python train_feature_encoder.py \
#     --cfg-path config/feature_encoding_no_llama.yaml \
#     --overrides \
#     out_dir="output/feature_encoding_no_llama_${now}" \
#     datasets_root=/dev/shm/algonauts2025

# "include_features=[qwen-2-5-omni-7b/layers.20, whisper/layers.12, llama_3.2_1b/layers.7]" \

# uv run python train_feature_encoder.py \
#     --overrides \
#     "include_features=[internvl3_8b/layers.20, qwen-2-5-omni-7b/layers.20, whisper/layers.12, llama_3.2_1b/layers.7]" \
#     out_dir="output/test" \
#     datasets_root=/dev/shm/algonauts2025

# uv run python train_feature_encoder.py \
#     --overrides \
#     out_dir="output/feature_encoding_default_${now}" \
#     datasets_root=/dev/shm/algonauts2025

# uv run python train_feature_encoder.py \
#     --overrides \
#     out_dir="output/feature_encoding_blockwise_${now}" \
#     model.encoder_blockwise=true \
#     datasets_root=/dev/shm/algonauts2025

# uv run python train_feature_encoder.py \
#     --overrides \
#     out_dir="output/feature_encoding_no-subj_${now}" \
#     model.with_subject_decoders=false \
#     datasets_root=/dev/shm/algonauts2025

# now=2506292006

# uv run python train_feature_encoder.py \
#     --overrides \
#     out_dir="output/feature_encoding_dinov2_${now}/embedding_only" \
#     "include_features=[dinov2/embedding]" \
#     datasets_root=/dev/shm/algonauts2025

# uv run python train_feature_encoder.py \
#     --overrides \
#     out_dir="output/feature_encoding_dinov2_${now}/blocks_5_15_25_linear" \
#     "include_features=[dinov2/blocks.5, dinov2/blocks.15, dinov2/blocks.25]" \
#     model.global_pool=linear \
#     datasets_root=/dev/shm/algonauts2025


# now=2506300839

# uv run python train_feature_encoder.py \
#     --overrides \
#     out_dir="output/feature_encoding_default_${now}" \
#     datasets_root=/dev/shm/algonauts2025

# uv run python train_feature_encoder.py \
#     --overrides \
#     out_dir="output/feature_encoding_add_dinov2_embedding_${now}" \
#     "include_features=[internvl3_8b/layers.20, whisper/layers.12, llama_3.2_1b/layers.7, dinov2/embedding]" \
#     datasets_root=/dev/shm/algonauts2025

now=2507111531

export CUDA_VISIBLE_DEVICES=2

# uv run python train_feature_encoder.py \
#     --overrides \
#     out_dir="output/feature_encoding_default_${now}"

# uv run python extract_text_features.py \
#     --overrides \
#     model=meta-llama/Llama-3.2-3B \
#     layers=[model.layers.*5]

# uv run python train_feature_encoder.py \
#     --overrides \
#     out_dir="output/feature_encoding_llama_3b_15_${now}" \
#     "include_features=[internvl3_8b/layers.20, whisper/layers.12, llama_3.2_3b/layers.15]"

# uv run python train_feature_encoder.py \
#     --overrides \
#     datasets.train.filter.seasons=[1,2,3] \
#     out_dir="output/feature_encoding_default_less_friends_${now}"

# uv run python train_feature_encoder.py \
#     --overrides \
#     datasets.train.filter.seasons=[1,2,3,4,5,6] \
#     datasets.train.filter.movies=[bourne,wolf,figures] \
#     val_set_name=val_life \
#     out_dir="output/feature_encoding_leave_out_life_${now}"

# uv run python train_feature_encoder.py \
#     --overrides \
#     datasets.train.filter.seasons=[1,2,3,4,5,6] \
#     datasets.train.filter.movies=[bourne,wolf,figures] \
#     val_set_name=val_life \
#     "include_features=[internvl3_8b/layers.20, whisper/layers.12, llama_3.2_1b/layers.7, vjepa2/layernorm]" \
#     out_dir="output/feature_encoding_leave_out_life_plus_vjepa2_${now}"

# uv run python train_feature_encoder.py \
#     --overrides \
#     datasets.train.filter.seasons=[1,2,3,4,5,6] \
#     datasets.train.filter.movies=[bourne,wolf,life] \
#     val_set_name=val_figures \
#     out_dir="output/feature_encoding_leave_out_figures_${now}"

# uv run python train_feature_encoder.py \
#     --overrides \
#     datasets.train.filter.seasons=[1,2,3,4,5,6] \
#     datasets.train.filter.movies=[bourne,wolf,life] \
#     val_set_name=val_figures \
#     "include_features=[internvl3_8b/layers.20, whisper/layers.12, llama_3.2_1b/layers.7, vjepa2/layernorm]" \
#     out_dir="output/feature_encoding_leave_out_figures_plus_vjepa2_${now}"

# uv run python train_feature_encoder.py \
#     --overrides \
#     "include_features=[internvl3_8b/layers.20, whisper/layers.12, llama_3.2_1b/layers.7, vjepa2/layernorm]" \
#     val_set_name=val_life \
#     out_dir="output/feature_encoding_default_plus_vjepa2_val_on_life_${now}"


# uv run python train_feature_encoder.py \
#     --overrides \
#     "include_features=[internvl3_8b/layers.20, whisper/layers.12, llama_3.2_1b/layers.7, vjepa2/layernorm]" \
#     val_set_name=val_life \
#     out_dir="output/feature_encoding_default_plus_vjepa2_val_on_life_${now}"

# uv run python train_feature_encoder.py \
#     --overrides \
#     "include_features=[internvl3_8b/layers.20, whisper/layers.12, llama_3.2_1b/layers.7, vjepa2/layernorm]" \
#     datasets.train.filter.seasons=[1,2,3,4,5] \
#     datasets.train.filter.movies=[bourne,wolf,figures,life] \
#     val_set_name=val_s6 \
#     out_dir="output/feature_encoding_default_leave_out_s6_plus_vjepa2_${now}"


# uv run python extract_text_features.py \
#     --overrides \
#     model=meta-llama/Llama-3.2-3B \
#     layers=[model.layers.7,model.layers.11,model.layers.15,model.layers.19,model.layers.23]


# "--overrides",
# "out_dir=tmp/test_attn_pool_v2",
# "model_name=multi_sub_conv_attn_linear",
# "model.global_pool=attn",
# "overwrite=true",

now=2507120759

export CUDA_VISIBLE_DEVICES=0

# uv run python train_feature_encoder.py \
#     --overrides \
#     model_name=multi_sub_conv_attn_linear \
#     model.global_pool=attn \
#     model.pool_num_heads=6 \
#     model.encoder_blockwise=true \
#     out_dir="output/feature_encoding_multi_attn_pool_blockwise_heads-6_${now}"


now="2507182054"

export CUDA_VISIBLE_DEVICES=5

# uv run python train_feature_encoder.py \
#     --overrides \
#     out_dir="output/feature_encoding_default_${now}"

export CUDA_VISIBLE_DEVICES=7

now="2507211453"

# uv run python train_feature_encoder.py \
#     --overrides \
#     out_dir="output/feature_encoding_default_${now}"

uv run python train_cross_encoder.py \
    --overrides \
    out_dir="output/cross_encoding_default_${now}"
