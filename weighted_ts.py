from collections import defaultdict
import os
import h5py
import numpy as np
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.model_summary import ModelSummary
from lightning.pytorch.utilities.combined_loader import CombinedLoader
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import zipfile
import sys
from functools import partial
import math
from utils import load_fmri, align_features_and_fmri_samples, align_features_and_fmri_samples_friends_s7, CosineLRSchedulerWithWarmup, calculate_metrics, normalize, normalize_across_episodes, check_fmri_centering, CosineAnnealingWarmDecayedRestarts

class AlgonautsDataset(Dataset):
    def __init__(self, features_dir, fmri_dir, movies, subject, excluded_samples_start=5, excluded_samples_end=5, hrf_delay=3, stimulus_window=5):
        self.features_dir = features_dir
        self.fmri_dir = fmri_dir
        self.movies = movies
        self.subject = subject
        self.excluded_samples_start = excluded_samples_start
        self.excluded_samples_end = excluded_samples_end
        self.hrf_delay = hrf_delay
        self.stimulus_window = stimulus_window
        self.partition_indices = defaultdict(list)
        
        # First load all raw features
        stimuli_features = {"visual": {}, "audio": {}, "language": {}}
        # Load audio and video features first
        for movie in self.movies:
            if 'friends' in movie:
                season = movie.split('-')[1]
                dir_list = sorted(os.listdir(self.features_dir['audio']  / 'audio')) #Just to get a list of all directories
                for episode in dir_list:
                    if f"{season}e" in episode and '_features_' in episode:
                        episode_base = episode.split('_features_')[0] # friends_s01e01 and so on....
                        
                        for modality in ['audio', 'visual']:
                            with h5py.File(self.features_dir[modality] / modality / f"{episode_base}_features_{modality}.h5", 'r') as f:
                                try:
                                    stimuli_features[modality][episode_base.split('_')[1]] = f['language_model.model.layers.20.post_attention_layernorm'][:] #(453, 3584)
                                    # stimuli_features[modality][episode_base.split('_')[1]] = f[episode_base.split('_')[1]][modality][:]
                                except:
                                    try:
                                        stimuli_features[modality][episode_base.split('_')[1]] = f['layers.31.fc2'][:]
                                    except:
                                        print("error friends vid/aud")
                                        f.visit(lambda x: print(x))
                                        # sys.exit()
                # lang_dir_list = sorted(os.listdir(self.features_dir['language'] / 'friends' / 'meta-llama__Llama-3.2-1B'))
                with h5py.File(self.features_dir['language'] / 'friends' / 'meta-llama__Llama-3.2-1B' / 'context-long_summary-0.h5', 'r') as f:
                    lang_dir_list = f.keys()
                # for episode in lang_dir_list:
                #     if f"{season}e" in episode and '_features_' in episode:
                #         episode_base = episode.split('_features_')[0]
                    for ep in lang_dir_list:
                        try:
                            stimuli_features['language'][ep] = f[ep]['model.layers.11'][:]
                        except:
                            print("error friends lang")
                            f.visit(lambda x: print(x))
            else:
                movie_name = movie.replace('movie10-', '')
                partitions = sorted([f for f in os.listdir(self.features_dir['audio']  / 'audio') if movie_name in f and '_features_' in f])

                for partition in partitions:
                    partition_base = partition.split('_features_')[0]
                    
                    for modality in ['audio', 'visual']:
                        with h5py.File(self.features_dir[modality] / modality / f"{partition_base}_features_{modality}.h5", 'r') as f:
                            try:
                                stimuli_features[modality][partition_base] = f['language_model.model.layers.20.post_attention_layernorm'][:]
                                # stimuli_features[modality][partition_base] = f[partition_base][modality][:]
                            except:
                                try:
                                    stimuli_features[modality][partition_base] = f['layers.31.fc2'][:]
                                except:
                                    print("error movie vid/aud")
                                    f.visit(lambda x: print(x))

                # lang_partitions = sorted([f for f in os.listdir(self.features_dir['language'] / 'language') if movie_name in f and '_features_' in f])
                with h5py.File(self.features_dir['language'] / 'movie10' / 'meta-llama__Llama-3.2-1B' / 'context-long_summary-0.h5', 'r') as f:
                    lang_dir_list = f.keys()
                    for base in lang_dir_list:
                        try:
                            stimuli_features['language'][base] = f[base]['model.layers.11'][:]
                        except:
                            print("error movie lang")
                            f.visit(lambda x: print(x))
                # for partition in lang_partitions:
                #     partition_base = partition.split('_features_')[0]
                    
                #     with h5py.File(self.features_dir['language'] / 'language' / f"{partition_base}_features_language.h5", 'r') as f:
                #         try:
                #             stimuli_features['language'][partition_base] = f[partition_base]['language_pooler_output'][:]
                #         except:
                #             print("error movie lang")
                #             f.visit(lambda x: print(x))
                #             sys.exit()

        fmri_data = load_fmri(self.fmri_dir, self.subject)
        # fmri = []
        # for key in fmri_data.keys():
        #     data = fmri_data[key]
        #     fmri.append(data)
        # fmri = np.concatenate(fmri, axis=0)
        # print('fMRI data shape: ', fmri.shape)
        # check_fmri_centering(fmri_data=fmri)
        # fmri_data = normalize_across_episodes(fmri_data)
        self.raw_stimuli = stimuli_features

        self.aligned_features, self.aligned_fmri = align_features_and_fmri_samples(
            stimuli_features, 
            fmri_data, 
            self.excluded_samples_start, 
            self.excluded_samples_end, 
            self.hrf_delay, 
            self.stimulus_window, 
            self.movies
        )

    def __len__(self):
        return self.aligned_features['audio'].shape[0]

    def __getitem__(self, idx):
        return {
            'audio': self.aligned_features['audio'][idx],
            'video': self.aligned_features['visual'][idx],
            'language': self.aligned_features['language'][idx],
            'fmri': self.aligned_fmri[idx],
        }
    
    def get_raw_stimuli(self):
        return self.raw_stimuli
# subjects_train = [1, 2, 3, 5] # Example: Use subjects 1, 2, 3, 5 for training
# subjects_val = 1   # Example: Validate on the same subjects (adjust if needed)

# batch_size = 8 # Per subject batch size
# num_workers = 1

# # --- Create individual Datasets and DataLoaders ---
# train_dataloaders = {}
# print("\n--- Creating Training DataLoaders ---")
# for subj_num in subjects_train:
#     subject_id_str = f"sub-{subj_num:02d}"
#     dataset = AlgonautsDataset(
#         features_dir, fmri_dir, movies=movies_train, subject=subj_num,
#         excluded_samples_start=excluded_samples_start, excluded_samples_end=excluded_samples_end,
#         hrf_delay=hrf_delay, stimulus_window=stimulus_window
#     )
#     # Important: Drop last batch if it's smaller than batch_size and might cause issues with dimensions
#     # Or handle variable batch sizes carefully in the model/loss
#     train_dataloaders[subject_id_str] = DataLoader(
#         dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
#         pin_memory=True, persistent_workers=True if num_workers > 0 else False, drop_last=True # Consider drop_last=True
#     )

# val_dataloaders = {}
# print("\n--- Creating Validation DataLoaders ---")
# for subj_num in subjects_val:
#     subject_id_str = f"sub-{subj_num:02d}"
#     dataset = AlgonautsDataset(
#         features_dir, fmri_dir, movies=movies_val, subject=subj_num,
#         excluded_samples_start=excluded_samples_start, excluded_samples_end=excluded_samples_end,
#         hrf_delay=hrf_delay, stimulus_window=stimulus_window
#     )
#     val_dataloaders[subject_id_str] = DataLoader(
#         dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
#         pin_memory=True, persistent_workers=True if num_workers > 0 else False, drop_last=False # Usually keep last batch in validation
    # )

# --- Create CombinedLoaders ---
# Mode 'max_size_cycle' iterates through the loaders sequentially based on the longest one,
# ensuring each batch is from a single subject's dataloader.
# combined_train_loader = CombinedLoader(train_dataloaders, mode="max_size_cycle")
# combined_val_loader = CombinedLoader(val_dataloaders, mode="max_size_cycle")

# print("\nCreated CombinedLoaders for training and validation.")


# print(train_dataloaders)
# print(val_dataloaders)

# print(len(combined_train_loader))
# print(len(combined_val_loader))





# InternVL3 bs, sw, 3584
# Whisper bs, sw, 1, 1280
class MMDTemporal(L.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.latent_dim = config['latent_dim']
        # self.codebook_size = config['codebook_size'] # Not used in this version
        self.learning_rate = config['learning_rate']
        self.dropout_prob = config['dropout_prob']
        self.encoder_dropout_prob = config['encoder_dropout_prob']
        # self.num_layers = config['num_layers'] # Original transformer layers, now using num_time_mlp_blocks
        self.stimulus_window = config['stimulus_window']
        self.weight_decay = config['weight_decay']
        self.alpha = config['alpha']
        self.vision_proj_dim = config['vision_proj_dim']
        self.audio_proj_dim = config['audio_proj_dim']
        self.language_proj_dim = config['vision_proj_dim'] 
        self.num_attn_heads = config['num_attn_heads']
        # self.subjects = config['subjects'] # Not directly used in model architecture
        self.decay_factor = config['decay_factor']
        self.warmup_epochs = config['warmup_epochs']
        self.minlr_mult = config['minlr_mult']

        # New config parameters for DynaDiff-style temporal processing
        self.time_specific_hidden_dim = config['latent_dim']

        # --- Input Projection Layers ---
        self.vision_proj = nn.Sequential(
            nn.Linear(3584, self.vision_proj_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.vision_proj_dim, self.latent_dim)
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(1280, self.audio_proj_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.audio_proj_dim, self.latent_dim)
        )
        self.language_proj = nn.Sequential(
            nn.Linear(2048, self.language_proj_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.language_proj_dim, self.latent_dim)
        )

        # --- Multimodal Fusion (Cross-Attention) ---
        self.audio_vis_cross_attn = nn.MultiheadAttention(
            embed_dim=self.latent_dim, num_heads=self.num_attn_heads,
            dropout=self.encoder_dropout_prob, batch_first=True
        )
        self.vision_aud_cross_attn = nn.MultiheadAttention( # Note: This seems to be a duplicate or a typo in original if it does the same as above.
                                                           # Assuming it's intended for vision query, audio key/value
            embed_dim=self.latent_dim, num_heads=self.num_attn_heads,
            dropout=self.encoder_dropout_prob, batch_first=True
        )
        self.av_fusion_norm = nn.LayerNorm(self.latent_dim)
        self.text_condition_attn = nn.MultiheadAttention(
            embed_dim=self.latent_dim, num_heads=self.num_attn_heads,
            dropout=self.encoder_dropout_prob, batch_first=True
        )
        self.text_condition_norm = nn.LayerNorm(self.latent_dim)

        # self.pos_embedding = nn.Parameter(torch.zeros(1, self.stimulus_window, self.latent_dim))

        # --- DynaDiff-style Temporal Processing ---
        # 1. Time-Specific Linear Layers (using Conv1d)
        # Input: (B, T, H_in=latent_dim)
        # Conv1d expects (B, C_in=T, L_in=H_in) if kernel_size=H_in
        # Output of Conv1d: (B, C_out = T * H_out, 1) if L_out = 1
        # We want output (B, T, H_out=time_specific_hidden_dim)
        
        # DynaDiff: self.lin0 = nn.Conv1d(in_channels=T, out_channels=T*hidden, kernel_size=in_dim, groups=T)
        # Input to this conv1d is (B, T, in_dim), kernel_size is in_dim.
        # This means L_in = in_dim.
        # self.time_specific_conv = nn.Conv1d(
        #     in_channels=self.stimulus_window,             # T (number of time steps)
        #     out_channels=self.stimulus_window * self.time_specific_hidden_dim, # T * H_out
        #     kernel_size=self.latent_dim,                 # H_in (feature dim per time step)
        #     groups=self.stimulus_window,                 # T groups for T independent layers
        #     bias=True
        # )
        self.time_specific_weights = nn.Parameter(
            torch.Tensor(self.stimulus_window, self.latent_dim, self.time_specific_hidden_dim)
        )
        self.time_specific_biases = nn.Parameter(
            torch.Tensor(self.stimulus_window, self.time_specific_hidden_dim)
        )
        nn.init.kaiming_uniform_(self.time_specific_weights, a=math.sqrt(5))
        if self.latent_dim > 0: # Avoid error if latent_dim is 0, though unlikely
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.time_specific_weights[0])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.time_specific_biases, -bound, bound)
        else:
            nn.init.zeros_(self.time_specific_biases)
        # 2. Post-Convolution Normalization, Activation, Dropout
        # Similar to DynaDiff's post_lin0
        # Input will be (B, T, time_specific_hidden_dim)
        self.post_tsp = nn.Sequential(
            nn.LayerNorm(self.time_specific_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.5)
        )

        self.temporal_agg = nn.Linear(self.stimulus_window, 1)

        # --- Final Projection to fMRI space ---
        # Input dim will be self.time_specific_hidden_dim
        self.fmri_proj = nn.Linear(self.time_specific_hidden_dim, 1000) 
        
        self.save_hyperparameters()

    def forward(self, video, audio, text):
        batch_size = video.shape[0]

        # --- Input Projections ---
        # vision: (B, T, 3584) -> vision_p: (B, T, latent_dim)
        vision_p = self.vision_proj(video)
        # audio: (B, T, 1, 1280) -> audio.squeeze(): (B, T, 1280) -> audio_p: (B, T, latent_dim)
        audio_p = self.audio_proj(audio.squeeze(2)) # Squeeze the channel dim for audio
        # text: (B, 2048) -> lang_p_proj: (B, latent_dim) -> lang_p: (B, 1, latent_dim)
        lang_p = self.language_proj(text).unsqueeze(1)

        # --- Multimodal Fusion ---
        # Query: vision_p (B,T,L), Key/Value: audio_p (B,T,L) -> ctx_vis (B,T,L)
        ctx_vis, _ = self.vision_aud_cross_attn(query=vision_p, key=audio_p, value=vision_p)
        vis_enh = vision_p + ctx_vis 
        # Query: audio_p (B,T,L), Key/Value: vision_p (B,T,L) -> ctx_aud (B,T,L)
        ctx_aud, _ = self.audio_vis_cross_attn(query=audio_p, key=vision_p, value=audio_p) # Corrected: was vision_aud_cross_attn
        aud_enh = audio_p + ctx_aud

        fused_av_stack = self.av_fusion_norm(vis_enh + aud_enh) # (B, T, L)

        # Condition with language features
        # Query: fused_av_stack (B,T,L), Key/Value: lang_p (B,1,L) -> txt_cxt (B,T,L)
        txt_cxt, _ = self.text_condition_attn(query=fused_av_stack, key=lang_p, value=lang_p)
        cond_seq = self.text_condition_norm(fused_av_stack + txt_cxt) # (B, T, L)

        # --- DynaDiff-style Temporal Processing ---
        # Input cond_seq: (B, T, H_in=latent_dim)
        
        # 1. Time-Specific Linear Layers
        # Conv1d expects (B, C_in=T, L_in=H_in)
        # Output of Conv1d: (B, C_out = T * H_out, 1)
        # Reshape to (B, T, H_out=time_specific_hidden_dim)
        # Note: DynaDiff applies lin0 directly to (B,T,H_in)
        # x = self.lin0(x).reshape(B, T, -1)
        # This implies Conv1d's kernel_size applies to the last dimension of (B, T, H_in)
        # if kernel_size = H_in.
        
        # Current cond_seq is (B, T, latent_dim). This matches (N, C_in, L_in) if T=C_in, latent_dim=L_in
        # So, this is the input to time_specific_conv
        # time_processed_seq = self.time_specific_conv(cond_seq) # (B, T*H_out, 1)
        # time_processed_seq = time_processed_seq.view(batch_size, self.stimulus_window, self.time_specific_hidden_dim) # (B, T, H_out)

        time_processed_seq = torch.einsum('bti,tio->bto', cond_seq, self.time_specific_weights)
        time_processed_seq += self.time_specific_biases

        # 2. Post-Convolution Norm, Activation, Dropout
        # Input (B, T, H_out), Output (B, T, H_out)
        time_processed_seq = self.post_tsp(time_processed_seq)
        
        # 3. Residual MLP Blocks
        # Input (B, T, H_out), Output (B, T, H_out)

        # 4. Temporal Aggregation
        # Input (B, T, H_out) -> Permute to (B, H_out, T) for aggregation layer
        aggregated_features = time_processed_seq.permute(0, 2, 1) # (B, H_out, T)
        
        aggregated_features = self.temporal_agg(aggregated_features).squeeze(-1) # (B, H_out)
        
        # aggregated_features is now (B, H_out = time_specific_hidden_dim)

        # 5. Final Projection to fMRI
        fmri_recon = self.fmri_proj(aggregated_features) # (B, 1000)
        
        return fmri_recon

    def training_step(self, batch, batch_idx):
        vision, audio, text, fmri = batch['video'], batch['audio'], batch['language'], batch['fmri']
        recon_fmri = self(vision, audio, text)

        mae, mse, _, pearson_r, _ = calculate_metrics(pred=recon_fmri, target=fmri)
        cosine_loss = (1 - F.cosine_similarity(recon_fmri, fmri, dim=1)).mean()
        loss = self.alpha * mse + ((1 - self.alpha) * cosine_loss)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_mse", mse, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train_mae", mae, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train_pearson_r", pearson_r, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train/cosine_loss", cosine_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        vision, audio, text, fmri= batch['video'], batch['audio'], batch['language'], batch['fmri']
        # audio = audio.squeeze() # Already squeezed in forward
        recon_fmri = self(vision, audio, text)

        mae, mse, _, pearson_r, _ = calculate_metrics(pred=recon_fmri, target=fmri)
        cosine_loss = (1 - F.cosine_similarity(recon_fmri, fmri, dim=1)).mean()
        loss = (self.alpha * mse) + ((1 - self.alpha) * cosine_loss)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True) # Corrected on_step for val
        self.log("val_mse", mse, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val_mae", mae, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val_pearson_r", pearson_r, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val/cosine_loss", cosine_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss # Important for some callbacks like ModelCheckpoint

    def on_train_epoch_end(self):
        optimizer = self.optimizers()
        if optimizer:
            current_lr = optimizer.param_groups[0]['lr']
            self.log('learning_rate', current_lr, on_step=False, on_epoch=True, prog_bar=False)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay,
            betas=(0.9, 0.95), eps=1e-8,
        )
        
        # warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        #     optimizer,
        #     start_factor=0.1,
        #     end_factor=1.0,
        #     total_iters=self.warmup_epochs
        # )

        scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.trainer.max_epochs,
            # T_max=self.trainer.max_epochs - self.warmup_epochs,
            eta_min=self.learning_rate * self.minlr_mult
        )

        # combined_scheduler = torch.optim.lr_scheduler.SequentialLR(
        #     optimizer,
        #     schedulers=[warmup_scheduler, scheduler],
        #     milestones=[self.warmup_epochs]
        # )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler, "interval": "epoch", "frequency": 1
            },
        }

if __name__ == '__main__':

    root_dir = Path('/home/mihirneal/Developer/')

    vision_dir = root_dir / 'algonauts/internvl3_8b_8bit/'
    # vision_dir = root_dir / 'algonauts/qwen_2.5omni-7B/'
    # vision_dir = '/home/pranav/mihir/algonauts_challenge/AlgonautsDS-features/developer_kit/stimulus_features/raw/'
    # audio_dir = '/home/pranav/mihir/algonauts_challenge/AlgonautsDS-features/developer_kit/stimulus_features/raw/'
    audio_dir = root_dir / 'algonauts/whisper/'
    # lang_dir = root_dir / 'algonauts/AlgonautsDS-features/developer_kit/stimulus_features/raw/'
    # lang_dir = root_dir / 'algonauts/meta-llama__Llama-3.2-1B'
    lang_dir = root_dir / 'algonauts/clane9_feat/algonauts2025/features'
    features_dir = {
        "visual": vision_dir,
        "audio": audio_dir,
        "language": lang_dir
    }
    fmri_dir = root_dir / 'algonauts/algonauts_2025.competitors/fmri/'
    # movies_train = ["friends-s01"]
    # movies_train = ["movie10-bourne", "movie10-figures", "movie10-life", "movie10-wolf"]
    movies_train = ["friends-s01", "friends-s02", "friends-s03", "friends-s04", "friends-s05", "movie10-bourne", "movie10-bourne", "movie10-figures", "movie10-life", "movie10-wolf"]
    movies_val = ["friends-s06"]
    modality = "all"  #@param ["visual", "audio", "language", "all"]

    excluded_samples_start = 5  #@param {type:"slider", min:0, max:20, step:1}
    excluded_samples_end = 5  #@param {type:"slider", min:0, max:20, step:1}
    hrf_delay = 0  #default: 3
    stimulus_window = 15

    subject = 1 #@param ["1", "2", "3", "5"] {type:"raw", allow-input: true}this
    train_ds = AlgonautsDataset(features_dir, fmri_dir, movies=movies_train, subject=subject, excluded_samples_start=excluded_samples_start, excluded_samples_end=excluded_samples_end, hrf_delay=hrf_delay, stimulus_window=stimulus_window)
    val_ds = AlgonautsDataset(features_dir, fmri_dir, movies=movies_val, subject=subject, excluded_samples_start=excluded_samples_start, excluded_samples_end=excluded_samples_end, hrf_delay=hrf_delay, stimulus_window=stimulus_window)
    train_loader = DataLoader(train_ds,
                            batch_size=32, 
                            num_workers=4,
                            pin_memory=True, 
                            prefetch_factor=2,
                            persistent_workers=True
                        )

    val_loader = DataLoader(val_ds, 
                            batch_size=32, 
                            num_workers=4, 
                            pin_memory=True, 
                            prefetch_factor=2, 
                            persistent_workers=True
                        )


    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")

    for i, batch in enumerate(train_loader):
        vision, audio, lang, fmri = batch['video'], batch['audio'], batch['language'], batch['fmri']
        print(f"Vision embeds: {vision.shape}")
        print(f"Audio embeds: {audio.shape}")
        print(f"Language embeds: {lang.shape}")
        print(f"fMRI: {fmri.shape}")
        break



    project = "algonauts-temporal"
    # run_name = "test"
    # run_name = "cos1NoCentFus_1024emb_15sw_5lr_drop1"
    run_name = "baseline_coslr1e5"
    wandb_logger = WandbLogger(
        project=project,
        name=run_name,
        dir="/home/mihirneal/Developer/algonauts/algonauts2025/wandb_logs"
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'/home/mihirneal/Developer/algonauts/algonauts2025/checkpoints/{project}/{run_name}',
        filename='{step:04d}-{val_pearson_r:.4f}',
        monitor='val_pearson_r',
        mode='max',
        save_top_k=1,
        save_last=True,
    )

    early_stopping = EarlyStopping(
        monitor='val_pearson_r',
        patience=10,
        mode='max',
        verbose=True,
        min_delta=1e-4
    )
    epochs = 15
    config = {
        'latent_dim': 1024,
        'codebook_size': 1000,
        'vision_proj_dim': 1024,
        'audio_proj_dim': 1024,
        'learning_rate': 1e-5,
        'minlr_mult': 0.001, 
        'dropout_prob': 0.3, #def 0.1
        'encoder_dropout_prob': 0.1,
        'num_layers': 6, #def: 6
        'num_attn_heads': 8,
        'stimulus_window': stimulus_window,
        'weight_decay': 0.01,
        'alpha': 1.0,
        'subjects': subject,
        'hrf_delay': hrf_delay,
        'decay_factor': 0.2,
        'epochs': epochs,
        'warmup_epochs': 3,
    }

    model = MMDTemporal(config)
    # model = torch.compile(model)

    # vis = torch.randn(32, 15, 3584)
    # aud = torch.randn(32, 15, 1, 1280)
    # lang = torch.randn(32, 2048)

    # recon_fmri = model(video=vis, audio=aud, text=lang)
    # print(recon_fmri.shape)
    # import sys; sys.exit()

    torch.set_float32_matmul_precision('high')
    summary = ModelSummary(model, max_depth=2)
    print(summary)

    # model = torch.compile(model)
    debug = False
    trainer = L.Trainer(
        accelerator='auto',
        devices=1,
        max_epochs=epochs,
        # callbacks=[early_stopping],
        callbacks=[checkpoint_callback],
        logger=wandb_logger if not debug else None,
        precision='bf16-mixed',
        log_every_n_steps=1,
    )
    # ckpt_path = "/home/pranav/mihir/algonauts_challenge/algonauts2025/checkpoints/fus_5l_12sw_5lr_drop1/step=97152-val_pearson_r=0.2384.ckpt"
    ckpt_path = None
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path if ckpt_path else None)
    wandb.finish()

