# source /home/connor/algonauts2025.clean/.venv/bin/activate # This

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=7

now=$(TZ=America/New_York date '+%y%m%d%H%M')

# Model 1
# uv run python train_feature_encoder.py \
#     --overrides \
#     out_dir=output/feature_encoding_remove_llama_remove_whisper_add_qwen3b_vjepa2_enc-kernel-45_${now} \
#     "include_features=[qwen-2-5-omni-3b/layers.20,internvl3_8b/layers.20, vjepa2/encoder.layernorm_avg]" \
#     model.encoder_kernel_size=45\
#     datasets_root=./datasets/ #/dev/shm/algonauts2025

# uv run python submit_feature_encoder.py \
#     --overrides \
#     out_dir=output/feature_encoding_remove_llama_remove_whisper_add_qwen3b_vjepa2_enc-kernel-45_${now} \
#     datasets_root=./datasets/

# Model 2
uv run python submit_feature_encoder.py \
    --overrides \
    out_dir=output/feature_encoding_remove_llama_add_qwen3b_vjepa2_enc-kernel-45_2507092108 \
    datasets_root=./datasets/ \
    test_set_name=ood-nochaplin


# Model 3
# ...

# Ensemble prediction (can also be run using existing model predictions at any time)
# foldernames= None #"feature_encoding_enc_kernel-45_2507061610,feature_encoding_enc_kernel-42_2507061637"
# python3 ensemble_predictions.py --target_filename "fmri_predictions_friends_s7.npy" --foldernames $foldernames

 #Issues: 
 # 1. I edited my version of ./config/default_feature_encoding.yaml so that we have more features available but still some are not available in Connor's home directory
 # 2. The submit_feature_encoder.py model construction part still broke if other features are added even though it is supposed to load the config file and create the same model
 # 3. Currently the best one is 'feature_encoding_enc_kernel-45_2507061610' but it looks like Connor tried encoding kernel with 49 size and it's worse than 33
