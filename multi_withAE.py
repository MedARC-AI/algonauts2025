from collections import defaultdict
import os
import h5py
import numpy as np
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
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
import random
import sys
import math
from fmri_vae import fMRILinearAE
from utils import load_fmri, align_features_and_fmri_samples, align_features_and_fmri_samples_friends_s7, CosineLRSchedulerWithWarmup, calculate_metrics, normalize, normalize_across_episodes, check_fmri_centering

class AlgonautsMultiSubjectDataset(Dataset):
    """
    A PyTorch Dataset to load and combine fMRI and stimulus data from multiple subjects.
    This class uses the functions defined in your 'utils.py' file.

    Args:
        features_dir (dict): Dictionary with paths to the feature directories for each modality.
        fmri_dir (str): Path to the root directory containing fMRI data.
        movies (list): List of movies to include (e.g., ['friends-s01', 'movie10-bourne']).
        subjects (list): List of subject identifiers (e.g., [1, 2, 3, 5]).
        excluded_samples_start (int): Number of initial TRs to exclude.
        excluded_samples_end (int): Number of final TRs to exclude.
        hrf_delay (int): Hemodynamic response function delay in TRs.
        stimulus_window (int): Number of feature chunks to use for modeling each TR.
    """
    def __init__(self, features_dir, fmri_dir, movies, subjects, excluded_samples_start=5, excluded_samples_end=5, hrf_delay=3, stimulus_window=5):
        self.features_dir = features_dir
        self.fmri_dir = fmri_dir
        self.movies = movies
        self.subjects = subjects
        self.excluded_samples_start = excluded_samples_start
        self.excluded_samples_end = excluded_samples_end
        self.hrf_delay = hrf_delay
        self.stimulus_window = stimulus_window

        # --- 1. Load Stimulus Features (Once) ---
        # Stimulus features are the same across all subjects, so we only need to load them once.
        print("Loading stimulus features...")
        stimuli_features = self._load_stimuli_features()

        # --- 2. Load and Align Data for Each Subject ---
        all_aligned_fmri = []
        base_aligned_features = None # To be set by the first subject
        self.subject_ids = [] # To track which subject each sample belongs to
        self.sample_indices = [] # To track original sample index for each subject

        for sub_id in self.subjects:
            print(f"\n--- Processing Subject {sub_id} ---")
            
            # Load this subject's fMRI data using the function from utils.py
            fmri_data = load_fmri(self.fmri_dir, sub_id)
            
            # Align this subject's fMRI data with the common stimulus features
            aligned_features_sub, aligned_fmri_sub = align_features_and_fmri_samples(
                stimuli_features,
                fmri_data,
                self.excluded_samples_start,
                self.excluded_samples_end,
                self.hrf_delay,
                self.stimulus_window,
                self.movies
            )

            # Store the aligned data
            all_aligned_fmri.append(aligned_fmri_sub)
            
            # Keep track of the subject ID and original index for each sample
            num_samples = aligned_fmri_sub.shape[0]
            self.subject_ids.extend([sub_id] * num_samples)
            self.sample_indices.extend(range(num_samples))

            if base_aligned_features is None:
                base_aligned_features = aligned_features_sub

        # --- 3. Concatenate Data from All Subjects ---
        print("\nConcatenating data from all subjects...")
        
        self.aligned_fmri = np.concatenate(all_aligned_fmri, axis=0)
        
        # The aligned features should be the same for all subjects. We use the one
        # from the first subject as the canonical version.
        self.aligned_features = base_aligned_features
        
        # Convert tracking lists to numpy arrays
        self.subject_ids = np.array(self.subject_ids)
        self.sample_indices = np.array(self.sample_indices)
        
        print(f"Dataset created successfully!")
        print(f"Total samples from {len(self.subjects)} subjects: {self.__len__()}")
        print(f"Shape of final aligned fMRI: {self.aligned_fmri.shape}")
        for mod, features in self.aligned_features.items():
             # The number of samples in the base aligned features should match the per-subject sample count
            print(f"Shape of shared aligned {mod} features: {features.shape}")


    def _load_stimuli_features(self):
        """
        Helper function to load all stimulus features using the logic from your
        original AlgonautsDataset class.
        """
        stimuli_features = defaultdict(dict)
        for movie in self.movies:
            if 'friends' in movie:
                season = movie.split('-')[1]
                # Visual and Audio
                dir_list = sorted(os.listdir(self.features_dir['audio'] / 'audio'))
                for episode in dir_list:
                    if f"{season}e" in episode and '_features_' in episode:
                        episode_base = episode.split('_features_')[0]
                        for modality in ['audio', 'visual']:
                            file_path = self.features_dir[modality] / modality / f"{episode_base}_features_{modality}.h5"
                            with h5py.File(file_path, 'r') as f:
                                # This key access logic is from your original script
                                try:
                                    stimuli_features[modality][episode_base.split('_')[1]] = f['language_model.model.layers.20.post_attention_layernorm'][:]
                                except KeyError:
                                    stimuli_features[modality][episode_base.split('_')[1]] = f['layers.12.fc2'][:]
                # Language
                lang_file = self.features_dir['language'] / 'friends' / 'meta-llama__Llama-3.2-1B' / 'context-long_summary-0.h5'
                with h5py.File(lang_file, 'r') as f:
                    for ep in f.keys():
                         if ep.startswith(season):
                            stimuli_features['language'][ep] = f[ep]['model.layers.7'][:]
            else: # movie10
                movie_name = movie.replace('movie10-', '')
                # Visual and Audio
                partitions = sorted([f for f in os.listdir(self.features_dir['audio'] / 'audio') if movie_name in f and '_features_' in f])
                for partition in partitions:
                    partition_base = partition.split('_features_')[0]
                    for modality in ['audio', 'visual']:
                         file_path = self.features_dir[modality] / modality / f"{partition_base}_features_{modality}.h5"
                         with h5py.File(file_path, 'r') as f:
                            try:
                                stimuli_features[modality][partition_base] = f['language_model.model.layers.20.post_attention_layernorm'][:]
                            except KeyError:
                                stimuli_features[modality][partition_base] = f['layers.12.fc2'][:]
                # Language
                lang_file = self.features_dir['language'] / 'movie10' / 'meta-llama__Llama-3.2-1B' / 'context-long_summary-0.h5'
                with h5py.File(lang_file, 'r') as f:
                     for base in f.keys():
                         if base.startswith(movie_name):
                            stimuli_features['language'][base] = f[base]['model.layers.7'][:]
                            
        return stimuli_features


    def __len__(self):
        """Returns the total number of samples across all subjects."""
        return self.aligned_fmri.shape[0]

    def __getitem__(self, idx):
        """
        Returns a single sample (stimulus-fMRI pair) from the combined dataset.
        """
        # Get the original sample index for the subject
        original_idx = self.sample_indices[idx]
        
        return {
            'audio': self.aligned_features['audio'][original_idx],
            'video': self.aligned_features['visual'][original_idx],
            'language': self.aligned_features['language'][original_idx],
            'fmri': self.aligned_fmri[idx], # fMRI is from the concatenated array
            'subject_id': self.subject_ids[idx]
        }


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


class MMDTemporal(L.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.latent_dim = config['latent_dim']
        # self.codebook_size = config['codebook_size'] # Not used in this version
        self.learning_rate = config['learning_rate']
        self.dropout_prob = config['dropout_prob']
        self.encoder_dropout_prob = config['encoder_dropout_prob']
        self.num_layers = config['num_layers'] # Original transformer layers, now using num_time_mlp_blocks
        self.stimulus_window = config['stimulus_window']
        self.weight_decay = config['weight_decay']
        self.alpha = config['alpha']
        self.vision_proj_dim = config['vision_proj_dim']
        self.audio_proj_dim = config['audio_proj_dim']
        self.language_proj_dim = config['vision_proj_dim'] 
        self.num_attn_heads = config['num_attn_heads']
        self.subjects = config['subjects'] # Not directly used in model architecture
        self.decay_factor = config['decay_factor']
        self.warmup_epochs = config['warmup_epochs']
        self.minlr_mult = config['minlr_mult']

        # New config parameters for DynaDiff-style temporal processing
        self.time_specific_hidden_dim = config['latent_dim']

        #Frozen AE
        self.ae = fMRILinearAE.load_from_checkpoint("/home/mihirneal/Developer/algonauts/algonauts2025/checkpoints/fmri_linearAE/baseline_multi768dim/epoch=39_val_pearson_r=0.955.ckpt")
        self.ae.eval()
        for param in self.ae.autoencoder.parameters():
            param.requires_grad = False


        # --- Input Projection Layers ---
        self.vision_proj = nn.Sequential(
            nn.Linear(3584, self.vision_proj_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.vision_proj_dim, self.latent_dim)
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(1280, self.audio_proj_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.audio_proj_dim, self.latent_dim)
        )
        self.language_proj = nn.Sequential(
            nn.Linear(2048, self.language_proj_dim),
            nn.GELU(),
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

        self.av_ffn = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.latent_dim, self.latent_dim)
        )
        self.text_ffn = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.latent_dim, self.latent_dim)
        )

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
        self.ae_map = nn.Linear(self.latent_dim, 768)
        # self.subject_map = {sub_id: i for i, sub_id in enumerate(self.subjects)}
        # self.subject_embedding = nn.Embedding(len(self.subjects), 32)

        # --- Final Projection to fMRI space ---
        # Input dim will be self.time_specific_hidden_dim
        # self.fmri_proj = nn.Linear(self.time_specific_hidden_dim, 1000) 
        # self.fmri_proj = nn.Sequential(
        #     nn.Linear(self.time_specific_hidden_dim + 32, self.latent_dim),
        #     nn.GELU(),
        #     nn.Dropout(self.dropout_prob),
        #     nn.Linear(self.latent_dim, 1000)
        # )

        self.save_hyperparameters()

    def forward(self, video, audio, text, subID):
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
        ffn_av = self.av_ffn(fused_av_stack)
        # Condition with language features
        # Query: fused_av_stack (B,T,L), Key/Value: lang_p (B,1,L) -> txt_cxt (B,T,L)
        txt_cxt, _ = self.text_condition_attn(query=fused_av_stack, key=lang_p, value=lang_p)
        ffn_txt = self.text_ffn(txt_cxt)
        # cond_seq = self.text_condition_norm(ffn_av + ffn_txt) # (B, T, L)
        cond_seq = self.text_condition_norm(fused_av_stack + txt_cxt) # (B, T, L)

        # cond_seq += self.pos_embedding

        # cond_seq = self.temporal_encoder(cond_seq)


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

        red_agg = self.ae_map(aggregated_features)
        fmri_recon = self.ae.autoencoder.decoder(red_agg)

        # sub_ind = torch.tensor([self.subject_map[s.item()] for s in subID]).to(aggregated_features.device)
        # subj_emb = self.subject_embedding(sub_ind)

        # combined_feat = torch.cat([aggregated_features, subj_emb], dim=1)
        
        # # aggregated_features is now (B, H_out = time_specific_hidden_dim)

        # # 5. Final Projection to fMRI
        # fmri_recon = self.fmri_proj(combined_feat) # (B, 1000)
        
        return fmri_recon
    
    def training_step(self, batch, batch_idx):
        # 1. Unpack the batch of data
        vision, audio, text, fmri, subID = batch['video'], batch['audio'], batch['language'], batch['fmri'], batch['subject_id']

        recon_fmri = self(vision, audio, text, subID)

        # 2. Get the aggregated stimulus features from the shared model
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
        # This step mirrors the training step but now calculates and logs metrics per subject.
        
        # 1. Unpack the batch of data
        vision, audio, text, fmri, subID = batch['video'], batch['audio'], batch['language'], batch['fmri'], batch['subject_id']

        # 2. Get the aggregated stimulus features from the shared model
        recon_fmri = self(vision, audio, text, subID)
        mae, mse, _, pearson_r, _ = calculate_metrics(pred=recon_fmri, target=fmri)
        cosine_loss = (1 - F.cosine_similarity(recon_fmri, fmri, dim=1)).mean()
        loss = (self.alpha * mse) + ((1 - self.alpha) * cosine_loss)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True) # Corrected on_step for val
        self.log("val_mse", mse, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val_mae", mae, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val_pearson_r", pearson_r, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val/cosine_loss", cosine_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        # # print("agg feat: ", agg_feat.dtype)

        # # 3. Efficiently apply subject-specific projection layers for the whole batch
        # recon_fmri = torch.zeros_like(fmri, dtype=recon_fmri.dtype)
        # # print("fmri placeholder: ", recon_fmri.dtype)
        # for sub in torch.unique(subID):
        #     mask = (subID == sub)
        #     # print("mask: ", mask.dtype)
        #     # print("sub: ", sub.dtype)
        #     proj_layer = self.fmri_proj[f"sub0{str(sub.item())}"]
        #     subject_specific_output = proj_layer(agg_feat[mask])
        #     # print("out: ", subject_specific_output.dtype)
        #     recon_fmri[mask] = subject_specific_output

        # # 4. Calculate overall loss for the batch (for logging and returning)
        # mae, mse, _, pearson_r, _ = calculate_metrics(pred=recon_fmri, target=fmri)
        # cosine_loss = (1 - F.cosine_similarity(recon_fmri, fmri, dim=1)).mean()
        # loss = self.alpha * mse + (1 - self.alpha) * cosine_loss

        # # Log overall validation metrics
        # # self.log("val/avg_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        # # self.log("val/avg_mse", mse, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        # # self.log("val/avg_pearson_r", pearson_r, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        # # --- Calculate and Log Metrics Per Subject ---
        # subject_pearsons = []
        # unique_subjects = torch.unique(subID)

        # for sub in unique_subjects:
        #     mask = (subID == sub)
            
        #     if mask.sum() == 0:
        #         continue

        #     recon_fmri_sub = recon_fmri[mask]
        #     fmri_sub = fmri[mask]

        #     mae_sub, mse_sub, _, pearson_r_sub, _ = calculate_metrics(pred=recon_fmri_sub, target=fmri_sub)
        #     cos_sub = (1 - F.cosine_similarity(recon_fmri_sub, fmri_sub, dim=1)).mean() 
        #     # Log metrics with the requested subject-first naming convention
        #     self.log(f"sub{sub.item()}/val_mse", mse_sub, on_step=False, on_epoch=True, sync_dist=True)
        #     self.log(f"sub{sub.item()}/val_mae", mae_sub, on_step=False, on_epoch=True, sync_dist=True)
        #     self.log(f"sub{sub.item()}/val_pearson_r", pearson_r_sub, on_step=False, on_epoch=True, sync_dist=True)
        #     self.log(f"sub{sub.item()}/val_cosine_loss", cos_sub, on_step=False, on_epoch=True, sync_dist=True)

        #     subject_pearsons.append(pearson_r_sub)

        # # --- Calculate and Log Average Pearson R Across Subjects ---
        # if subject_pearsons:
        #     avg_pearson_r = sum(subject_pearsons) / len(unique_subjects)
        #     self.log("val_avgpearson_r", avg_pearson_r, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)



    # def training_step(self, batch, batch_idx):
    #     vision, audio, text, fmri = batch['video'], batch['audio'], batch['language'], batch['fmri']
    #     agg_feat = self(vision, audio, text) # Assuming this is used by self.fmri_proj
        
    #     subjects = ['02', '03', '05']
    #     total_loss = 0.0
        
    #     # Loop through each subject to calculate and log metrics
    #     for sub_id in subjects:
    #         # Get the target fMRI data for the current subject
    #         fmri_target = fmri[f'sub{sub_id}']
            
    #         # Perform the subject-specific projection/reconstruction
    #         rec = self.fmri_proj[f'sub{sub_id}'](agg_feat)

    #         # Calculate metrics and cosine loss
    #         mae, mse, _, pearson_r, _ = calculate_metrics(pred=rec, target=fmri_target)
    #         cos_l = (1 - F.cosine_similarity(rec, fmri_target, dim=1)).mean()

    #         # Calculate and accumulate the total loss
    #         subject_loss = self.alpha * mse + ((1 - self.alpha) * cos_l)
    #         total_loss += subject_loss

    #         # Log subject-specific metrics
    #         self.log(f"sub{sub_id}/train_mse", mse, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
    #         self.log(f"sub{sub_id}/train_mae", mae, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
    #         self.log(f"sub{sub_id}/train_pearson_r", pearson_r, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
    #         self.log(f"sub{sub_id}/train_cosine_loss", cos_l, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

    #     # Log the overall training loss
    #     self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

    #     return total_loss

    # def validation_step(self, batch, batch_idx):
    #     vision, audio, text, fmri = batch['video'], batch['audio'], batch['language'], batch['fmri']
        
    #     # This assumes the model's forward pass returns a dictionary of predictions
    #     # keyed by subject ID, just like the input `fmri` data.
    #     agg_feat = self(vision, audio, text)

    #     subjects = ['02', '03', '05']
    #     avg_pearson = 0.0
    #     total_loss = 0.0

    #     for sub_id in subjects:
    #         # Get the prediction and target for the current subject
    #         pred = self.fmri_proj[f'sub{sub_id}'](agg_feat)
    #         target = fmri[f'sub{sub_id}']

    #         # Calculate metrics and cosine loss
    #         mae, mse, _, pearson_r, _ = calculate_metrics(pred=pred, target=target)
    #         cos_l = (1 - F.cosine_similarity(pred, target, dim=1)).mean()

    #         # Calculate and accumulate the total loss
    #         subject_loss = self.alpha * mse + ((1 - self.alpha) * cos_l)
    #         total_loss += subject_loss
    #         avg_pearson += pearson_r

    #         # Log subject-specific validation metrics
    #         self.log(f"sub{sub_id}/val_mse", mse, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
    #         self.log(f"sub{sub_id}/val_mae", mae, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
    #         self.log(f"sub{sub_id}/val_pearson_r", pearson_r, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
    #         self.log(f"sub{sub_id}/val_cosine_loss", cos_l, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        
    #     # Log the overall validation loss
    #     self.log("val_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
    #     avg_pearson /= 3
    #     self.log("val_avgPearsonR", avg_pearson, on_step=False, on_epoch=True, prog_bar=False)
        


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
    L.seed_everything(42, workers=True)

    root_dir = Path('/home/mihirneal/Developer/')
    vision_dir = root_dir / 'algonauts/internvl3_8b_8bit/'
    # vision_dir = root_dir / 'algonauts/qwen_omni3-5'
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
    stimulus_window = 12

    # subject = [1]
    # subjects = [1] #@param ["1", "2", "3", "5"] {type:"raw", allow-input: true}
    subjects = [1, 2, 3, 5] #@param ["1", "2", "3", "5"] {type:"raw", allow-input: true}
    train_ds = AlgonautsMultiSubjectDataset(features_dir, fmri_dir, movies=movies_train, subjects=subjects, excluded_samples_start=excluded_samples_start, excluded_samples_end=excluded_samples_end, hrf_delay=hrf_delay, stimulus_window=stimulus_window)
    val_ds = AlgonautsMultiSubjectDataset(features_dir, fmri_dir, movies=movies_val, subjects=subjects, excluded_samples_start=excluded_samples_start, excluded_samples_end=excluded_samples_end, hrf_delay=hrf_delay, stimulus_window=stimulus_window)
    train_loader = DataLoader(train_ds,
                            batch_size=32, 
                            num_workers=4,
                            pin_memory=True, 
                            prefetch_factor=2,
                            persistent_workers=True,
                            shuffle=True
                        )

    val_loader = DataLoader(val_ds, 
                            batch_size=32, 
                            num_workers=4, 
                            pin_memory=True, 
                            prefetch_factor=2, 
                            persistent_workers=True,
                            drop_last=True
                        )


    # print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")

    for i, batch in enumerate(train_loader):
        vision, audio, lang, fmri, sub = batch['video'], batch['audio'], batch['language'], batch['fmri'], batch['subject_id']
        print(f"Vision embeds: {vision.shape}")
        print(f"Audio embeds: {audio.shape}")
        print(f"Language embeds: {lang.shape}")
        print('fmri: ', fmri.shape)
        print(sub)
        
        break

    # import sys; sys.exit()

    project = "alg_multisub"
    # run_name = "test"
    # run_name = "cos1NoCentFus_1024emb_15sw_5lr_drop1"
    run_name = "sub1_2_3_5_FrozAE"
    wandb_logger = WandbLogger(
        project=project,
        name=run_name,
        dir=root_dir / "algonauts" /"algonauts2025/wandb_logs"
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=root_dir / 'algonauts' / f'algonauts2025/checkpoints/{project}/{run_name}',
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
        'dropout_prob': 0.4, #def 0.1
        'encoder_dropout_prob': 0.1,
        'num_layers': 4, #def: 6
        'num_attn_heads': 8,
        'stimulus_window': stimulus_window,
        'weight_decay': 0.01,
        'alpha': 0.6,
        'subjects': subjects,
        'hrf_delay': hrf_delay,
        'decay_factor': 0.2,
        'epochs': epochs,
        'warmup_epochs': None,
    }


    model = MMDTemporal(config)

    # vid = torch.randn([32, 15, 3584])
    # aud = torch.randn([32, 15, 1, 1280])
    # text = torch.randn(32, 2048)
    # subID = torch.randn(32, 1)

    # logits = model(video=vid, audio=aud, text=text)[:, -1, :]
    # print(logits.shape)
    # import sys; sys.exit()




    torch.set_float32_matmul_precision('high')
    summary = ModelSummary(model, max_depth=2)
    print(summary)

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
        gradient_clip_val=1.0
    )
    # ckpt_path = "/home/mihirneal/Developer/algonauts/algonauts2025/checkpoints/alg_extTrain/Int20_Whis12_Lla7_baseline/step=62550-val_pearson_r=0.2967.ckpt"
    ckpt_path = None
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path if ckpt_path else None)
    wandb.finish()


