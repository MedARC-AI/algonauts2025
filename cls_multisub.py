from collections import defaultdict
import os
from types import NoneType
import h5py
import numpy as np
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, ModelSummary
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.combined_loader import CombinedLoader
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import zipfile
import random
import sys
import math
# Assuming you have a utils.py with these functions
# from utils import load_fmri, align_features_and_fmri_samples, align_features_and_fmri_samples_friends_s7, CosineLRSchedulerWithWarmup, calculate_metrics, normalize, normalize_across_episodes, check_fmri_centering

# Mock utility functions for demonstration if utils.py is not available
def load_fmri(fmri_dir, sub_id):
    print(f"Mock loading fMRI for sub {sub_id}")
    return defaultdict(lambda: np.random.rand(100, 1000))

def align_features_and_fmri_samples(stimuli_features, fmri_data, excluded_start, excluded_end, hrf_delay, window, movies):
    print("Mock aligning features and fMRI")
    num_samples = 500 # Mock number of samples
    
    aligned_fmri = np.random.rand(num_samples, 1000)
    
    aligned_features = {
        'audio': np.random.rand(num_samples, window, 1, 1280),
        'visual': np.random.rand(num_samples, window, 3584),
        'language': np.random.rand(num_samples, 2048)
    }
    return aligned_features, aligned_fmri

def calculate_metrics(pred, target):
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    
    mae = np.mean(np.abs(pred - target))
    mse = np.mean((pred - target)**2)
    
    # Calculate Pearson correlation coefficient row-wise
    pred_mean = np.mean(pred, axis=1, keepdims=True)
    target_mean = np.mean(target, axis=1, keepdims=True)
    pred_std = np.std(pred, axis=1, keepdims=True)
    target_std = np.std(target, axis=1, keepdims=True)
    
    cov = np.mean((pred - pred_mean) * (target - target_mean), axis=1)
    
    # Handle cases where standard deviation is zero to avoid division by zero
    valid_mask = (pred_std.flatten() > 0) & (target_std.flatten() > 0)
    pearson_r = np.zeros(pred.shape[0])
    if np.any(valid_mask):
       pearson_r[valid_mask] = (cov[valid_mask] / (pred_std.flatten()[valid_mask] * target_std.flatten()[valid_mask]))
       
    pearson_r = np.mean(pearson_r)

    return mae, mse, 0, pearson_r, 0


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
    def __init__(
            self, 
            features_dir, 
            fmri_dir, 
            movies, 
            subjects, 
            excluded_samples_start=5, 
            excluded_samples_end=5, 
            hrf_delay=3, 
            stimulus_window=5,
            mean=None,
            std=None
            ):
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

        if mean is None and std is None:
            print("Calculating global mean and std from training data")
            self.mean = np.mean(self.aligned_fmri, axis=0)
            self.std = np.std(self.aligned_fmri, axis=0)
            self.std[self.std == 0] = 1e-6
        else:
            self.mean = mean
            self.std = std
        
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
        Helper function to load all stimulus features.
        This is a mock implementation. Replace with your actual loading logic.
        """
        stimuli_features = defaultdict(dict)
        print("Mock loading stimulus features from H5 files.")
        # This part is complex and depends on your file structure.
        # For this example, we'll just create dummy data structures.
        for movie in self.movies:
            if 'friends' in movie:
                season = movie.split('-')[1]
                for ep_num in range(1, 5): # Mock episodes
                    ep = f"{season}e{ep_num:02d}"
                    stimuli_features['visual'][ep] = np.random.rand(180, 3584)
                    stimuli_features['audio'][ep] = np.random.rand(180, 1280)
                    stimuli_features['language'][ep] = np.random.rand(180, 2048)
            else: # movie10
                movie_name = movie.replace('movie10-', '')
                for part in range(1, 3): # Mock partitions
                    base = f"{movie_name}_p{part}"
                    stimuli_features['visual'][base] = np.random.rand(180, 3584)
                    stimuli_features['audio'][base] = np.random.rand(180, 1280)
                    stimuli_features['language'][base] = np.random.rand(180, 2048)
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


class MMDTemporal(L.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.latent_dim = config['latent_dim']
        self.learning_rate = config['learning_rate']
        self.dropout_prob = config['dropout_prob']
        self.encoder_dropout_prob = config['encoder_dropout_prob']
        self.num_layers = config['num_layers']
        self.stimulus_window = config['stimulus_window']
        self.weight_decay = config['weight_decay']
        self.alpha = config['alpha']
        self.vision_proj_dim = config['vision_proj_dim']
        self.audio_proj_dim = config['audio_proj_dim']
        self.language_proj_dim = config['language_proj_dim'] 
        self.num_attn_heads = config['num_attn_heads']
        self.subjects = config['subjects']
        self.minlr_mult = config['minlr_mult']
        
        # --- Subject and Positional Embeddings ---
        # Create a mapping from subject ID to a zero-based index
        self.subject_map = {sub_id: i for i, sub_id in enumerate(self.subjects)}
        # Create a learnable CLS token for each subject
        self.subject_cls_embedding = nn.Embedding(len(self.subjects), self.latent_dim)
        # Positional embedding for the sequence (stimulus window + 1 for the CLS token)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.stimulus_window + 1, self.latent_dim))


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
        self.vision_aud_cross_attn = nn.MultiheadAttention(
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
            nn.Linear(self.latent_dim, self.latent_dim), nn.GELU(),
            nn.Dropout(self.dropout_prob), nn.Linear(self.latent_dim, self.latent_dim)
        )
        self.text_ffn = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim), nn.GELU(),
            nn.Dropout(self.dropout_prob), nn.Linear(self.latent_dim, self.latent_dim)
        )
        
        # --- Temporal Encoder ---
        # Standard Transformer Encoder to process the sequence with the CLS token
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim, 
            nhead=self.num_attn_heads,
            dim_feedforward=self.latent_dim * 4, # A common practice
            dropout=self.encoder_dropout_prob,
            batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )

        # --- Final Projection to fMRI space ---
        # Takes the output of the CLS token as input
        self.fmri_proj = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.latent_dim, 1000)
        )
      
        self.save_hyperparameters()

    def forward(self, video, audio, subID, text):
        batch_size = video.shape[0]

        # --- Input Projections ---
        vision_p = self.vision_proj(video) # (B, T, latent_dim)
        audio_p = self.audio_proj(audio.squeeze(2)) # (B, T, latent_dim)
        lang_p = self.language_proj(text).unsqueeze(1) # (B, 1, latent_dim)

        # --- Multimodal Fusion ---
        ctx_vis, _ = self.vision_aud_cross_attn(query=vision_p, key=audio_p, value=vision_p)
        vis_enh = vision_p + ctx_vis 
        ctx_aud, _ = self.audio_vis_cross_attn(query=audio_p, key=vision_p, value=audio_p)
        aud_enh = audio_p + ctx_aud

        fused_av_stack = self.av_fusion_norm(vis_enh + aud_enh)
        ffn_av = self.av_ffn(fused_av_stack)
        
        txt_cxt, _ = self.text_condition_attn(query=fused_av_stack, key=lang_p, value=lang_p)
        ffn_txt = self.text_ffn(txt_cxt)
        
        cond_seq = self.text_condition_norm(ffn_av + ffn_txt) # (B, T, L)
        
        # --- Temporal Processing with Subject-Specific CLS Token ---
        
        # 1. Get subject indices and retrieve the corresponding CLS tokens
        sub_indices = torch.tensor([self.subject_map[s.item()] for s in subID], device=self.device)
        cls_tokens = self.subject_cls_embedding(sub_indices).unsqueeze(1) # (B, 1, L)

        # 2. Prepend CLS tokens to the feature sequence
        # Final sequence shape: (B, T+1, L)
        full_sequence = torch.cat([cls_tokens, cond_seq], dim=1)

        # 3. Add positional embeddings
        full_sequence += self.pos_embedding

        # 4. Pass through the Transformer Encoder
        encoder_output = self.temporal_encoder(full_sequence) # (B, T+1, L)

        # 5. Extract the output of the CLS token (the first token)
        # This serves as the aggregated representation for the entire sequence
        aggregated_features = encoder_output[:, 0, :] # (B, L)

        # 6. Final Projection to fMRI space
        fmri_recon = self.fmri_proj(aggregated_features) # (B, 1000)
    
        return fmri_recon
    
    def training_step(self, batch, batch_idx):
        vision, audio, text, fmri, subID = batch['video'], batch['audio'], batch['language'], batch['fmri'], batch['subject_id']
        recon_fmri = self(vision, audio, subID, text)

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
        vision, audio, text, fmri, subID = batch['video'], batch['audio'], batch['language'], batch['fmri'], batch['subject_id']
        recon_fmri = self(vision, audio, subID, text)
        
        mae, mse, _, pearson_r, _ = calculate_metrics(pred=recon_fmri, target=fmri)
        cosine_loss = (1 - F.cosine_similarity(recon_fmri, fmri, dim=1)).mean()
        loss = (self.alpha * mse) + ((1 - self.alpha) * cosine_loss)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_mse", mse, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val_mae", mae, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val_pearson_r", pearson_r, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val/cosine_loss", cosine_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        # --- Calculate and Log Metrics Per Subject ---
        subject_pearsons = []
        unique_subjects = torch.unique(subID)

        for sub in unique_subjects:
            mask = (subID == sub)
            
            if mask.sum() == 0:
                continue

            recon_fmri_sub = recon_fmri[mask]
            fmri_sub = fmri[mask]

            mae_sub, mse_sub, _, pearson_r_sub, _ = calculate_metrics(pred=recon_fmri_sub, target=fmri_sub)
            cos_sub = (1 - F.cosine_similarity(recon_fmri_sub, fmri_sub, dim=1)).mean() 
            self.log(f"sub{sub.item()}/val_mse", mse_sub, on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"sub{sub.item()}/val_mae", mae_sub, on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"sub{sub.item()}/val_pearson_r", pearson_r_sub, on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"sub{sub.item()}/val_cosine_loss", cos_sub, on_step=False, on_epoch=True, sync_dist=True)

            subject_pearsons.append(pearson_r_sub)

        # --- Calculate and Log Average Pearson R Across Subjects ---
        if subject_pearsons:
            avg_pearson_r = sum(subject_pearsons) / len(subject_pearsons)
            self.log("val_avgpearson_r", avg_pearson_r, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)


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
        
        scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.learning_rate * self.minlr_mult
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler, "interval": "epoch", "frequency": 1
            },
        }
    

if __name__ == '__main__':
    L.seed_everything(42, workers=True)

    # --- Configuration ---
    # NOTE: Using dummy paths for demonstration. Replace with your actual paths.
    root_dir = Path('./data')
    root_dir.mkdir(exist_ok=True)
    features_dir = {
        "visual": root_dir / "visual",
        "audio": root_dir / "audio",
        "language": root_dir / "language"
    }
    fmri_dir = root_dir / 'fmri'
    for p in [fmri_dir] + list(features_dir.values()):
        p.mkdir(exist_ok=True, parents=True)

    movies_train = ["friends-s01", "movie10-bourne"]
    movies_val = ["friends-s02"]
    
    excluded_samples_start = 5
    excluded_samples_end = 5
    hrf_delay = 0
    stimulus_window = 15

    subjects = [1, 2, 3, 5]
    
    print("--- Creating Datasets ---")
    train_ds = AlgonautsMultiSubjectDataset(
        features_dir, fmri_dir, movies=movies_train, subjects=subjects, 
        excluded_samples_start=excluded_samples_start, excluded_samples_end=excluded_samples_end, 
        hrf_delay=hrf_delay, stimulus_window=stimulus_window
    )
    val_ds = AlgonautsMultiSubjectDataset(
        features_dir, fmri_dir, movies=movies_val, subjects=subjects, 
        excluded_samples_start=excluded_samples_start, excluded_samples_end=excluded_samples_end, 
        hrf_delay=hrf_delay, stimulus_window=stimulus_window, 
        mean=train_ds.mean, std=train_ds.std
    )
    
    train_loader = DataLoader(train_ds, batch_size=32, num_workers=4, pin_memory=True, persistent_workers=True, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, num_workers=4, pin_memory=True, persistent_workers=True)

    print("\n--- Verifying a batch ---")
    for i, batch in enumerate(train_loader):
        vision, audio, lang, fmri, sub = batch['video'], batch['audio'], batch['language'], batch['fmri'], batch['subject_id']
        print(f"Vision embeds: {vision.shape}")
        print(f"Audio embeds: {audio.shape}")
        print(f"Language embeds: {lang.shape}")
        print(f'fMRI: {fmri.shape}')
        print(f'Subject IDs: {sub}')
        break

    # --- Training Setup ---
    project = "alg_cls_token_test"
    run_name = "sub-cls_transformer-encoder"
    
    # Set to False to enable logging
    debug = True 
    
    wandb_logger = WandbLogger(
        project=project,
        name=run_name,
        dir=root_dir / "wandb_logs",
        log_model=True,
        offline=debug 
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=root_dir / f'checkpoints/{project}/{run_name}',
        filename='{epoch:02d}-{val_avgpearson_r:.4f}',
        monitor='val_avgpearson_r',
        mode='max',
        save_top_k=2,
        save_last=True,
    )

    epochs = 8
    config = {
        'latent_dim': 1024,
        'vision_proj_dim': 1024,
        'audio_proj_dim': 1024,
        'language_proj_dim': 1024,
        'learning_rate': 1e-5, # Might need adjustment
        'minlr_mult': 0.01, 
        'dropout_prob': 0.2,
        'encoder_dropout_prob': 0.2,
        'num_layers': 2,
        'num_attn_heads': 8,
        'stimulus_window': stimulus_window,
        'weight_decay': 0.05,
        'alpha': 0.8,
        'subjects': subjects,
        'hrf_delay': hrf_delay,
        'epochs': epochs,
    }

    model = MMDTemporal(config)

    torch.set_float32_matmul_precision('high')
    
    trainer = L.Trainer(
        accelerator='auto',
        devices=1,
        max_epochs=epochs,
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
        precision='bf16-mixed',
        log_every_n_steps=10,
        gradient_clip_val=1.0
    )
    
    print("\n--- Starting Training ---")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # print("Training commented out. Uncomment the line above to run.")
    
    if not debug:
        wandb.finish()