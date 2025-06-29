import math
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import itertools
import bisect

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, ModelSummary
from lightning.pytorch.loggers import WandbLogger
import wandb
import os

from utils import (
    load_fmri, 
    align_features_and_fmri_samples,
    calculate_metrics,
    normalize_fmri,
    check_fmri_stats,
)


class AlgonautsMultiSubjectDataset(Dataset):
    """
    PyTorch Dataset for loading multimodal features and fMRI data from multiple subjects.
    
    This dataset handles:
    - Loading visual (InternVL), audio (Whisper), and language (Llama) features
    - Aligning stimulus features with fMRI responses considering HRF delay
    - Managing data from multiple subjects in a single dataset
    """
    
    def __init__(
        self,
        features_dir: Dict[str, Path],
        fmri_dir: Path,
        movies: List[str],
        subjects: List[int],
        excluded_samples_start: int = 5,
        excluded_samples_end: int = 5,
        hrf_delay: int = 0,
        stimulus_window: int = 12,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        stimulus_features: Optional[Dict[str, Dict[str, np.ndarray]]] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            features_dir: Dictionary mapping modality names to their feature directories
            fmri_dir: Path to fMRI data directory
            movies: List of movie identifiers to include
            subjects: List of subject IDs to include
            excluded_samples_start: Number of initial TRs to exclude
            excluded_samples_end: Number of final TRs to exclude
            hrf_delay: Hemodynamic response function delay in TRs
            stimulus_window: Number of feature chunks to use for modeling each TR
            mean: Pre-computed mean for normalization (optional)
            std: Pre-computed std for normalization (optional)
        """
        self.features_dir = features_dir
        self.fmri_dir = fmri_dir
        self.movies = movies
        self.subjects = subjects
        self.excluded_samples_start = excluded_samples_start
        self.excluded_samples_end = excluded_samples_end
        self.hrf_delay = hrf_delay
        self.stimulus_window = stimulus_window
        
        if stimulus_features is not None:
            # If pre-computed stimulus features are provided, use them directly
            self.stimuli_features = stimulus_features
        else:
        # Load stimulus features
            print(f"Loading stimulus features")
            self.stimuli_features = self._load_all_stimulus_features()
        
        # Load and align data for each subject
        all_aligned_fmri = []
        base_aligned_features = None
        self.subject_ids = []
        self.sample_indices = []
        
        for idx, subject_id in enumerate(self.subjects):
            print(f"Processing Subject {subject_id}...")
            
            # Load subject's fMRI data
            fmri_data = load_fmri(self.fmri_dir, subject_id)

            if idx == 0:
                self.indexes_of_clips = self._get_indexes_of_clips(self.movies, fmri_data)
            
            # Align features and fMRI
            aligned_features_sub, aligned_fmri_sub = align_features_and_fmri_samples(
                self.stimuli_features,
                fmri_data,
                self.excluded_samples_start,
                self.excluded_samples_end,
                self.hrf_delay,
                1, # to avoid moving to ram duplicated data we keep stimulus_window = 1
                self.movies
            )
            
            # Store aligned data
            all_aligned_fmri.append(aligned_fmri_sub)
            
            # Track subject ID and sample indices
            num_samples = aligned_fmri_sub.shape[0]
            self.subject_ids.extend([subject_id] * num_samples)
            self.sample_indices.extend(range(num_samples))
            
            if base_aligned_features is None:
                base_aligned_features = aligned_features_sub
            else:
                # Concatenate features across subjects
                for mod in aligned_features_sub:
                    if mod not in base_aligned_features:
                        base_aligned_features[mod] = aligned_features_sub[mod]
                    else:
                        base_aligned_features[mod] = np.concatenate(
                            (base_aligned_features[mod], aligned_features_sub[mod]), axis=0
                        )
        
        # Concatenate all subjects' data
        print("Concatenating data from all subjects...")
        self.aligned_fmri = np.concatenate(all_aligned_fmri, axis=0)
        self.aligned_features = base_aligned_features
        self.subject_ids = np.array(self.subject_ids)
        self.sample_indices = np.array(self.sample_indices)
        
        # Compute or use provided normalization statistics
        if mean is None and std is None:
            print("Computing normalization statistics from training data...")
            _, self.mean, self.std = normalize_fmri(self.aligned_fmri)
        else:
            self.mean = mean
            self.std = std
        
        print(f"Dataset created: {len(self)} samples from {len(self.subjects)} subjects")
        print(f"fMRI shape: {self.aligned_fmri.shape}")
        for mod, features in self.aligned_features.items():
            print(f"Shape of aligned {mod} features: {features.shape}")
    
    def _load_all_stimulus_features(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Load all stimulus features for the specified movies."""
        features = defaultdict(dict)
        
        for movie in self.movies:
            if 'friends' in movie:
                self._load_friends_features(movie, features)
            else:
                self._load_movie10_features(movie, features)
        
        return dict(features)
    
    def _load_friends_features(self, movie: str, features: Dict):
        """Load features for Friends episodes."""
        season = movie.split('-')[1]
        
        # Load visual and audio features
        audio_dir = self.features_dir['audio'] / 'audio'
        episode_files = sorted([f for f in os.listdir(audio_dir) 
                               if f"{season}e" in f and '_features_' in f])
        
        for episode_file in episode_files:
            episode_base = episode_file.split('_features_')[0]
            episode_key = episode_base.split('_')[1]
            
            # Visual features - InternVL3
            visual_path = self.features_dir['visual'] / 'visual' / f"{episode_base}_features_visual.h5"
            with h5py.File(visual_path, 'r') as f:
                features['visual'][episode_key] = f['language_model.model.layers.40.post_attention_layernorm'][:]
            
            # Audio features - Whisper
            audio_path = self.features_dir['audio'] / 'audio' / f"{episode_base}_features_audio.h5"
            with h5py.File(audio_path, 'r') as f:
                features['audio'][episode_key] = f['layers.12.fc2'][:]
        
        # # Language features - Llama
        # lang_path = (self.features_dir['language'] / 'friends' / 
        #             'meta-llama__Llama-3.2-1B' / 'context-long_summary-0.h5')
        # with h5py.File(lang_path, 'r') as f:
        #     for ep in f.keys():
        #         if ep.startswith(season):
        #             features['language'][ep] = f[ep]['model.layers.7'][:]
    
    def _load_movie10_features(self, movie: str, features: Dict):
        """Load features for movie10 clips."""
        movie_name = movie.replace('movie10-', '')
        
        # Visual and audio features
        audio_dir = self.features_dir['audio'] / 'audio'
        partitions = sorted([f for f in os.listdir(audio_dir) 
                           if movie_name in f and '_features_' in f])
        
        for partition in partitions:
            partition_base = partition.split('_features_')[0]
            
            # Visual features - InternVL3
            visual_path = self.features_dir['visual'] / 'visual' / f"{partition_base}_features_visual.h5"
            with h5py.File(visual_path, 'r') as f:
                features['visual'][partition_base] = f['language_model.model.layers.40.post_attention_layernorm'][:]
            
            # Audio features - Whisper
            audio_path = self.features_dir['audio'] / 'audio' / f"{partition_base}_features_audio.h5"
            with h5py.File(audio_path, 'r') as f:
                features['audio'][partition_base] = f['layers.12.fc2'][:]
        
        # # Language features - Llama
        # lang_path = (self.features_dir['language'] / 'movie10' / 
        #             'meta-llama__Llama-3.2-1B' / 'context-long_summary-0.h5')
        # with h5py.File(lang_path, 'r') as f:
        #     for base in f.keys():
        #         if base.startswith(movie_name):
        #             features['language'][base] = f[base]['model.layers.7'][:]
    
    def _get_closest_clip_index(self, x: int) -> int:
        """
        Given a sorted list `start_indices` and a number `x`,
        return (start, distance) where `start` is the greatest element ≤ x.
        """
        # bisect_right gives insertion point to the right of any equals
        pos = bisect.bisect_right(self.indexes_of_clips, x) - 1
        if pos < 0:
            # x is before the very first start; we’ll default to the first block
            start = self.indexes_of_clips[0]
        else:
            start = self.indexes_of_clips[pos]
        return start, x - start

    def _get_indexes_of_clips(self, movies: str, fmri_data: Dict[str, np.ndarray]) -> List[int]:
        all_m_splits = []
        for movie in movies:
            ### Get the IDs of all movies splits for the selected movie ###
            if movie[:7] == 'friends':
                id = movie[8:]
            elif movie[:7] == 'movie10':
                id = movie[8:]
            movie_splits = [key for key in fmri_data if id in key[:len(id)]]
            all_m_splits.extend(movie_splits)
        
        size_of_clips = [len(fmri_data[key]) for key in all_m_splits]
        indexes_of_start_clips = [0] + list(itertools.accumulate(size_of_clips[:-1]))
        return indexes_of_start_clips

    def __len__(self) -> int:
        return self.aligned_fmri.shape[0]
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # original_idx = self.sample_indices[idx]
        
        # start_clip, distance = self._get_closest_clip_index(original_idx)
        # if distance >= self.stimulus_window:
        #     c_audio = torch.from_numpy(self.aligned_features['audio'][original_idx-self.stimulus_window+1:original_idx+1]).float().squeeze()
        #     c_video = torch.from_numpy(self.aligned_features['visual'][original_idx-self.stimulus_window+1:original_idx+1]).float().squeeze()
    
        # else:
        #     c_audio = torch.from_numpy(self.aligned_features['audio'][start_clip:start_clip+self.stimulus_window]).float().squeeze()
        #     c_video = torch.from_numpy(self.aligned_features['visual'][start_clip:start_clip+self.stimulus_window]).float().squeeze()

        start_clip, distance = self._get_closest_clip_index(idx)
        if distance >= self.stimulus_window:
            c_audio = torch.from_numpy(self.aligned_features['audio'][idx-self.stimulus_window+1:idx+1]).float().squeeze()
            c_video = torch.from_numpy(self.aligned_features['visual'][idx-self.stimulus_window+1:idx+1]).float().squeeze()
    
        else:
            c_audio = torch.from_numpy(self.aligned_features['audio'][start_clip:start_clip+self.stimulus_window]).float().squeeze()
            c_video = torch.from_numpy(self.aligned_features['visual'][start_clip:start_clip+self.stimulus_window]).float().squeeze()


        return {
            'audio': c_audio,
            'video': c_video,
            # 'language': torch.from_numpy(self.aligned_features['language'][original_idx]).float(),
            'fmri': torch.from_numpy(self.aligned_fmri[idx]).float(),
            'subject': self.subject_ids[idx]
        }



class AudioVisualFusion(nn.Module):
    """Handles cross-modal attention between vision and audio."""
    
    def __init__(self, latent_dim: int, num_heads: int, dropout: float):
        super().__init__()
        
        # Cross-attention layers
        self.vision_audio_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.audio_vision_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.av_fusion_norm = nn.LayerNorm(latent_dim)
        
        # Feed-forward network
        self.av_ffn = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim)
        )
    
    def forward(
        self, 
        vision_features: torch.Tensor,
        audio_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform audio-visual fusion.
        
        Args:
            vision_features: (B, T, D) vision features
            audio_features: (B, T, D) audio features
            
        Returns:
            (B, T, D) fused features
        """
        # Vision-audio cross-attention
        vision_context, _ = self.vision_audio_attn(
            query=vision_features,
            key=audio_features,
            value=vision_features
        )
        vision_enhanced = vision_features + vision_context
        
        # Audio-vision cross-attention
        audio_context, _ = self.audio_vision_attn(
            query=audio_features,
            key=vision_features,
            value=audio_features
        )
        audio_enhanced = audio_features + audio_context
        
        # Combine audio-visual features
        av_fused = self.av_fusion_norm(vision_enhanced + audio_enhanced)
        av_fused = av_fused + self.av_ffn(av_fused)
        
        return av_fused
    


class TemporalProcessor(nn.Module):
    """Temporal processing module with time-specific transformations."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        stimulus_window: int,
        dropout: float = 0.5
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.stimulus_window = stimulus_window
        
        # Time-specific linear transformations
        self.time_specific_weights = nn.Parameter(
            torch.Tensor(stimulus_window, input_dim, hidden_dim)
        )
        self.time_specific_biases = nn.Parameter(
            torch.Tensor(stimulus_window, hidden_dim)
        )
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.time_specific_weights, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.time_specific_weights[0])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.time_specific_biases, -bound, bound)
        
        # Post-processing layers
        self.post_process = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Temporal aggregation
        self.temporal_aggregation = nn.Linear(stimulus_window, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process temporal features.
        
        Args:
            x: (B, T, D) input features
            
        Returns:
            (B, H) aggregated temporal features
        """
        # Apply time-specific transformations
        x_transformed = torch.einsum('btd,tdh->bth', x, self.time_specific_weights)
        x_transformed = x_transformed + self.time_specific_biases
        
        # Post-process
        x_processed = self.post_process(x_transformed)
        
        # Aggregate across time
        x_aggregated = x_processed.permute(0, 2, 1)
        x_aggregated = self.temporal_aggregation(x_aggregated).squeeze(-1)
        
        return x_aggregated


class AudioVisualfMRIModel(L.LightningModule):
    """
    Audio-Visual model for fMRI reconstruction.
    
    Architecture:
    1. Modality-specific projections (vision, audio)
    2. Audio-visual fusion with cross-attention
    3. Temporal processing with time-specific transformations
    4. Final projection to fMRI space
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Extract configuration
        self.latent_dim = config['latent_dim']
        self.dropout_prob = config['dropout_prob']
        self.encoder_dropout_prob = config['encoder_dropout_prob']
        self.stimulus_window = config['stimulus_window']
        self.num_attn_heads = config['num_attn_heads']
        self.learning_rate = config['learning_rate']
        self.weight_decay = config['weight_decay']
        self.subjects = config['subjects']
        
        # Input dimensions
        self.vision_input_dim = 5120  # InternVL features
        self.audio_input_dim = 1280   # Whisper features
        self.fmri_output_dim = 1000
        
        # Modality-specific projections
        self.vision_proj = self._build_projection(
            self.vision_input_dim,
            config['vision_proj_dim'],
            self.latent_dim
        )
        
        self.audio_proj = self._build_projection(
            self.audio_input_dim,
            config['audio_proj_dim'],
            self.latent_dim
        )
        
        # Audio-visual fusion
        self.av_fusion = AudioVisualFusion(
            latent_dim=self.latent_dim,
            num_heads=self.num_attn_heads,
            dropout=self.encoder_dropout_prob
        )
        
        # Temporal processing
        self.temporal_processor = TemporalProcessor(
            input_dim=self.latent_dim,
            hidden_dim=self.latent_dim,
            stimulus_window=self.stimulus_window,
            dropout=0.5
        )
        
        # Final fMRI projection
        self.fmri_proj_subjects = nn.ModuleDict()
        for sub_id in self.subjects:
            # Keys in ModuleDict must be strings
            self.fmri_proj_subjects[f"sub_{sub_id}"] = self._build_projection(
                input_dim=self.latent_dim,
                hidden_dim=self.latent_dim,
                output_dim=1000
            
            )
                
        self.save_hyperparameters()
    
    def _build_projection(self, input_dim: int, hidden_dim: int, output_dim: int) -> nn.Module:
        """Build a two-layer projection network."""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
        subject: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            video: (B, T, 3584) visual features
            audio: (B, T, 1, 1280) audio features
            
        Returns:
            (B, 1000) reconstructed fMRI signals
        """
        # Project inputs to common latent space
        vision_latent = self.vision_proj(video)  # (B, T, D)
        audio_latent = self.audio_proj(audio.squeeze(2))  # (B, T, D)
        
        # Audio-visual fusion
        fused_features = self.av_fusion(vision_latent, audio_latent)  # (B, T, D)
        
        # Temporal processing
        temporal_features = self.temporal_processor(fused_features)  # (B, D)

        batch_size = temporal_features.size(0)
        fmri_reconstruction = torch.zeros(
            batch_size, self.fmri_output_dim, 
            device=self.device, 
            dtype=temporal_features.dtype
        )

        # Process each subject in the batch with its specific MLP
        for sub_id in self.subjects:
            mask = (subject == sub_id)
            
            if not torch.any(mask):
                continue

            sub_features = temporal_features[mask]
            sub_mlp = self.fmri_proj_subjects[f"sub_{sub_id}"]
            sub_pred_fmri = sub_mlp(sub_features)
            fmri_reconstruction[mask] = sub_pred_fmri
        
        return fmri_reconstruction
    
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.8) -> torch.Tensor:
        """Compute combined MSE and cosine similarity loss."""
        mse = F.mse_loss(pred, target)
        cosine_loss = (1 - F.cosine_similarity(pred, target, dim=1)).mean()
        return alpha * mse + (1 - alpha) * cosine_loss
        
    def _step_loop(self, batch, run_type):
        video, audio, fmri, subject = batch['video'], batch['audio'], batch['fmri'], batch['subject']
        
        # Forward pass
        pred_fmri = model(video, audio, subject)
        
        # Compute loss
        loss = self.compute_loss(pred_fmri, fmri, alpha)
        mae, mse, _, pearson_r, _ = calculate_metrics(pred=pred_fmri, target=fmri)

        self.log(f"{run_type}/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{run_type}/mae", mae, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f"{run_type}/mse", mse, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f"{run_type}_pearson_r", pearson_r, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        return loss
    
    def training_step(self, batch):
        return self._step_loop(batch, run_type="train")
    
    def validation_step(self, batch):
        return self._step_loop(batch, run_type="val")
    
    def on_train_epoch_end(self):
        optimizer = self.optimizers()
        if optimizer:
            current_lr = optimizer.param_groups[0]['lr']
            self.log('learning_rate', current_lr, on_step=False, on_epoch=True, prog_bar=False)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8
        )

        scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.learning_rate * 0.001
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            },
        }
if __name__ == '__main__':
    L.seed_everything(42, workers=True)

    root_dir =  Path('/home/mihir/projects/')

    # Feature directories (no language features)
    features_dir = {
        "visual": root_dir / 'datasets' / "InternVL3_38B",
        "audio": root_dir / 'datasets' / 'whisper_feat' / 'whisper'
    }

    fmri_dir = root_dir / 'datasets' / 'algonauts_2025.competitors' / 'fmri'

    # Verify paths exist
    for name, path in features_dir.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} features directory not found: {path}")
    if not fmri_dir.exists():
        raise FileNotFoundError(f"fMRI directory not found: {fmri_dir}")

    # Experiment configuration
    movies_train = [
        "friends-s01", "friends-s02", "friends-s03",
        "friends-s04", "friends-s05", "friends-s06", "movie10-bourne", 
        "movie10-wolf"
    ]
    movies_val = ["movie10-figures", "movie10-life"]

    # Dataset parameters
    excluded_samples_start = 5
    excluded_samples_end = 5
    hrf_delay = 0
    stimulus_window = 15
    subjects = [1, 2, 3]

    # Model configuration
    model_config = {
        'latent_dim': 1024,
        'vision_proj_dim': 1024,
        'audio_proj_dim': 1024,
        'dropout_prob': 0.4,
        'encoder_dropout_prob': 0.2,
        'num_attn_heads': 8,
        'stimulus_window': stimulus_window,
        'learning_rate': 1e-5,
        'weight_decay': 0.04,
        'alpha': 0.8,
        'subjects': subjects,
    }

    # Training configuration
    batch_size = 32
    num_workers = 4
    learning_rate = 1e-5
    epochs = 15
    alpha = 0.8  # Weight for MSE vs cosine loss


    # Save configuration
    config = {
        **model_config,
        
        'hrf_delay': hrf_delay,
        'excluded_samples_start': excluded_samples_start,
        'excluded_samples_end': excluded_samples_end,
        'batch_size': batch_size,
        'epochs': epochs,
        'movies_train': movies_train,
        'movies_val': movies_val,
        'feature_dirs': {k: str(v) for k, v in features_dir.items()}, 
        'fmri_dir': str(fmri_dir)  
    }

    # Save configuration (including full experiment config)
    full_config = {
        **config,  
        'model_config': model_config  
    }

    # Create datasets
    print("Creating training dataset...")
    train_dataset = AlgonautsMultiSubjectDataset(
        features_dir, fmri_dir,
        movies=movies_train,
        subjects=subjects,
        excluded_samples_start=excluded_samples_start,
        excluded_samples_end=excluded_samples_end,
        hrf_delay=hrf_delay,
        stimulus_window=stimulus_window
    )

    print("Creating validation dataset...")
    val_dataset = AlgonautsMultiSubjectDataset(
        features_dir, fmri_dir,
        movies=movies_val,
        subjects=subjects,
        excluded_samples_start=excluded_samples_start,
        excluded_samples_end=excluded_samples_end,
        hrf_delay=hrf_delay,
        stimulus_window=stimulus_window,
        mean=train_dataset.mean,
        std=train_dataset.std
    )

    # print("Creating OOD test dataset...")
    # ood_dataset = AlgonautsDataset(
    #     features_dir, fmri_dir,
    #     movies=movies_ood,
    #     subject=subject,
    #     excluded_samples_start=excluded_samples_start,
    #     excluded_samples_end=excluded_samples_end,
    #     hrf_delay=hrf_delay,
    #     stimulus_window=stimulus_window,
    #     mean=train_dataset.mean,
    #     std=train_dataset.std
    # )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    # ood_loader = DataLoader(
    #     ood_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=num_workers,
    #     pin_memory=True,
    #     prefetch_factor=2,
    #     persistent_workers=True
    # )

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")

    for i, batch in enumerate(val_loader):
       video, audio, fmri, subject = batch['video'], batch['audio'], batch['fmri'], batch['subject']
       print("vision: ", video.shape) 
       print("audio: ", audio.shape) 
       print("fmri: ", fmri.shape) 
       print("subject: ", subject.shape) 
       print(subject)
       break


    # Initialize model
    torch.set_float32_matmul_precision('high')
    model = AudioVisualfMRIModel(model_config)
    summary = ModelSummary(max_depth=3)
    # print(summary)

    project="alg_avMulti"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"subALL_hrf{hrf_delay}_sw{stimulus_window}_{timestamp}"
    save_dir = root_dir / 'algonauts2025' / 'checkpoints' / exp_name
    wandb_logger = WandbLogger(
        project=project,
        name=exp_name,
        dir=root_dir / 'algonauts2025' / 'wandb' ,
        config=full_config
    )

    ckpt_call = ModelCheckpoint(
        dirpath=save_dir,
        filename='{step:04d}-{val_pearson_r:.4f}',
        monitor='val_pearson_r',
        mode='max',
        verbose=True,
        save_top_k=1,
        save_last=True,
    )

    debug = False
    trainer = L.Trainer(
        accelerator='auto',
        devices=[2],
        max_epochs=epochs,
        # callbacks=[early_stopping],
        callbacks=[ckpt_call],
        logger=wandb_logger if not debug else None,
        precision='bf16-mixed',
        log_every_n_steps=1,
        gradient_clip_val=1.0
    )
    # ckpt_path = "/home/mihirneal/Developer/algonauts/algonauts2025/checkpoints/alg_extTrain/Int20_Whis12_Lla7_baseline/step=62550-val_pearson_r=0.2967.ckpt"
    ckpt_path = None
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=[val_loader], ckpt_path=ckpt_path if ckpt_path else None)
    # model = MMDTemporal.load_from_checkpoint("/home/mihir/projects/algonauts2025/checkpoints/alg_ablations_ood/test_LinSub1_ood_base/step=23628-val_pearson_r=0.2914.ckpt")
    # if ood:
    #     trainer.test(dataloaders=test_loader)
    wandb.finish()