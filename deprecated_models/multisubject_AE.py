import math
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union

from datetime import datetime

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
from fmri_vae import fMRILinearAE

from utils import (
    load_fmri, 
    align_features_and_fmri_samples,
    calculate_metrics,
    normalize_fmri,
    check_fmri_stats,
)

class AlgonautsMultiSubjectDataset(Dataset):
    """
    PyTorch Dataset for loading multimodal features and fMRI data for multiple subjects.

    This dataset handles:
    - Loading visual (InternVL) and audio (Whisper) features
    - Loading fMRI data for one or more subjects
    - Aligning stimulus features with fMRI responses considering HRF delay
    """

    def __init__(
        self,
        features_dir: Dict[str, str],
        fmri_dir: str,
        movies: List[str],
        subjects: Union[int, List[int]],
        excluded_samples_start: int = 5,
        excluded_samples_end: int = 5,
        hrf_delay: int = 0,
        stimulus_window: int = 12,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None
    ):
        """
        Initialize the dataset.

        Args:
            features_dir: Dictionary mapping modality names to their feature directories.
            fmri_dir: Path to the fMRI data directory.
            movies: List of movie identifiers to include.
            subjects: A single subject ID or a list of subject IDs.
            excluded_samples_start: Number of initial TRs to exclude.
            excluded_samples_end: Number of final TRs to exclude.
            hrf_delay: Hemodynamic response function delay in TRs.
            stimulus_window: Number of feature chunks to use for modeling each TR.
            mean: Pre-computed mean for normalization (optional).
            std: Pre-computed std for normalization (optional).
        """
        self.features_dir = {k: Path(v) for k, v in features_dir.items()}
        self.fmri_dir = Path(fmri_dir)
        self.movies = movies
        self.subjects = [subjects] if isinstance(subjects, int) else subjects
        self.excluded_samples_start = excluded_samples_start
        self.excluded_samples_end = excluded_samples_end
        self.hrf_delay = hrf_delay
        self.stimulus_window = stimulus_window

        # Load stimulus features (once, as they are shared across subjects)
        print("Loading stimulus features...")
        self.stimuli_features = self._load_all_stimulus_features()

        all_aligned_fmri = []
        all_subject_ids = []

        for subject in self.subjects:
            print(f"Processing data for subject {subject}...")

            # Load fMRI data for the current subject
            fmri_data = load_fmri(self.fmri_dir, subject)

            # Align features and fMRI data
            aligned_features, aligned_fmri = align_features_and_fmri_samples(
                self.stimuli_features,
                fmri_data,
                self.excluded_samples_start,
                self.excluded_samples_end,
                self.hrf_delay,
                self.stimulus_window,
                self.movies
            )
            
            all_aligned_fmri.append(aligned_fmri)
            
            # Create a subject ID array for the samples of the current subject
            all_subject_ids.append(np.full(aligned_fmri.shape[0], subject))

        # Concatenate data from all subjects
        self.aligned_fmri = np.concatenate(all_aligned_fmri, axis=0)
        self.subject_ids = np.concatenate(all_subject_ids, axis=0)
        self.aligned_features = aligned_features # Features are aligned but not duplicated per subject

        # Compute or use provided normalization statistics
        if mean is None and std is None:
            print("Computing normalization statistics from the data...")
            _, self.mean, self.std = normalize_fmri(self.aligned_fmri)
        else:
            self.mean = mean
            self.std = std

        # Print data statistics for debugging
        check_fmri_stats(self.aligned_fmri, "Aggregated fMRI data")

        print(f"Dataset created for subjects {self.subjects}: {len(self)} samples")
        print(f"fMRI shape: {self.aligned_fmri.shape}")
        print(f"Subject IDs shape: {self.subject_ids.shape}")

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
        audio_dir = self.features_dir['audio'] / 'audio'
        episode_files = sorted([f for f in os.listdir(audio_dir) if f"{season}e" in f and '_features_' in f])

        for episode_file in episode_files:
            episode_base = episode_file.split('_features_')[0]
            episode_key = episode_base.split('_')[1]
            
            visual_path = self.features_dir['visual'] / 'visual' / f"{episode_base}_features_visual.h5"
            with h5py.File(visual_path, 'r') as f:
                features['visual'][episode_key] = f['language_model.model.layers.20.post_attention_layernorm'][:]
            
            audio_path = audio_dir / f"{episode_base}_features_audio.h5"
            with h5py.File(audio_path, 'r') as f:
                features['audio'][episode_key] = f['layers.12.fc2'][:]

    def _load_movie10_features(self, movie: str, features: Dict):
        """Load features for movie10 clips."""
        movie_name = movie.replace('movie10-', '')
        audio_dir = self.features_dir['audio'] / 'audio'
        partitions = sorted([f for f in os.listdir(audio_dir) if movie_name in f and '_features_' in f])

        for partition in partitions:
            partition_base = partition.split('_features_')[0]
            
            visual_path = self.features_dir['visual'] / 'visual' / f"{partition_base}_features_visual.h5"
            with h5py.File(visual_path, 'r') as f:
                features['visual'][partition_base] = f['language_model.model.layers.20.post_attention_layernorm'][:]
            
            audio_path = audio_dir / f"{partition_base}_features_audio.h5"
            with h5py.File(audio_path, 'r') as f:
                features['audio'][partition_base] = f['layers.12.fc2'][:]

    def __len__(self) -> int:
        return self.aligned_fmri.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'audio': torch.from_numpy(self.aligned_features['audio'][idx % len(self.aligned_features['audio'])]).float(),
            'video': torch.from_numpy(self.aligned_features['visual'][idx % len(self.aligned_features['visual'])]).float(),
            'fmri': torch.from_numpy(self.aligned_fmri[idx]).float(),
            'subject': torch.tensor(self.subject_ids[idx], dtype=torch.long)
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


class AudioVisualContrastiveModel(L.LightningModule):
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
        self.temperature = config['temperature']
        
        # Input dimensions
        self.vision_input_dim = 3584  # InternVL features
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

        self.proj = nn.Linear(self.latent_dim, 768)
        self.fmri_AE = fMRILinearAE.load_from_checkpoint("/home/mihir/projects/epoch=39_val_pearson_r=0.955.ckpt")
        self.fmri_AE.eval()
        for param in self.fmri_AE.autoencoder.parameters():
            param.requires_grad = False

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
        audio: torch.Tensor
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
        embeds = self.proj(temporal_features)
        
        
        return embeds
    
    
    def contrastive_loss(self, stimulus_embeds, fmri_embeds):
        # Calculate cosine similarity between all pairs
        logits = torch.matmul(stimulus_embeds, fmri_embeds.T) / self.temperature
        
        # Labels are the diagonal elements, as they represent the positive pairs
        labels = torch.arange(logits.shape[0], device=self.device)
        
        # Calculate cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        
        return loss
        
    def _step_loop(self, batch, run_type):
        video, audio, fmri, subj = batch['video'], batch['audio'], batch['fmri'], batch['subject']
        
        # Forward pass
        stimuli_embed = self(video, audio)
        fmri_enc = self.fmri_AE.autoencoder.encoder(fmri)

        # Compute loss
        loss = self.contrastive_loss(stimulus_embeds=stimuli_embed, fmri_embeds=fmri_enc)
        # mae, mse, _, pearson_r, _ = calculate_metrics(pred=pred_fmri, target=fmri)

        self.log(f"{run_type}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
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
        "visual": root_dir / 'datasets' / "InternVL3_feat",
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
    subjects = [1, 2, 3, 5]

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
        'temperature': 0.07,
    }

    # Training configuration
    batch_size = 32
    num_workers = 4
    learning_rate = 1e-5
    weight_decay = 0.02
    epochs = 20

    alpha = 0.8  # Weight for MSE vs cosine loss

    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


    # Save configuration
    config = {
        **model_config,
        'subjects': subjects,
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


    # Initialize model
    torch.set_float32_matmul_precision('high')
    model = AudioVisualContrastiveModel(model_config)
    summary = ModelSummary(max_depth=3)
    # print(summary)

    project="alg_constrastive"
    exp_name = f"contrastive_multisub_hrf{hrf_delay}_sw{stimulus_window}_{timestamp}"
    save_dir = root_dir / 'algonauts2025' / 'checkpoints' / exp_name
    wandb_logger = WandbLogger(
        project=project,
        name=exp_name,
        dir=root_dir / 'algonauts2025' / 'wandb' ,
        config=full_config
    )

    ckpt_call = ModelCheckpoint(
        dirpath=save_dir,
        filename='{step:04d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        verbose=True,
        save_top_k=1,
        save_last=True,
    )

    debug = False
    trainer = L.Trainer(
        accelerator='auto',
        devices=[1],
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