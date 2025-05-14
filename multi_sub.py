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
from utils import load_fmri, align_features_and_fmri_samples, align_features_and_fmri_samples_friends_s7, CosineLRSchedulerWithWarmup, calculate_metrics, normalize, normalize_across_episodes, check_fmri_centering


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
                dir_list = sorted(os.listdir(self.features_dir['audio'] + 'audio')) #Just to get a list of all directories
                for episode in dir_list:
                    if f"{season}e" in episode and '_features_' in episode:
                        episode_base = episode.split('_features_')[0] # friends_s01e01 and so on....
                        
                        for modality in ['audio', 'visual']:
                            with h5py.File(os.path.join(self.features_dir[modality], modality, f"{episode_base}_features_{modality}.h5"), 'r') as f:
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
                lang_dir_list = sorted(os.listdir(self.features_dir['language'] + 'language'))
                for episode in lang_dir_list:
                    if f"{season}e" in episode and '_features_' in episode:
                        episode_base = episode.split('_features_')[0]
                        
                        with h5py.File(os.path.join(self.features_dir['language'], 'language', f"{episode_base}_features_language.h5"), 'r') as f:
                            try:
                                st_season_episode = episode_base.split('_')[1]
                                stimuli_features['language'][st_season_episode] = f[st_season_episode]['language_pooler_output'][:]
                            except:
                                print("error friends lang")
                                f.visit(lambda x: print(x))
            else:
                movie_name = movie.replace('movie10-', '')
                partitions = sorted([f for f in os.listdir(self.features_dir['audio'] + 'audio') if movie_name in f and '_features_' in f])
                
                for partition in partitions:
                    partition_base = partition.split('_features_')[0]
                    
                    for modality in ['audio', 'visual']:
                        with h5py.File(os.path.join(self.features_dir[modality], modality, f"{partition_base}_features_{modality}.h5"), 'r') as f:
                            try:
                                stimuli_features[modality][partition_base] = f['language_model.model.layers.20.post_attention_layernorm'][:]
                                # stimuli_features[modality][partition_base] = f[partition_base][modality][:]
                            except:
                                try:
                                    stimuli_features[modality][partition_base] = f['layers.31.fc2'][:]
                                except:
                                    print("error movie vid/aud")
                                    f.visit(lambda x: print(x))

                lang_partitions = sorted([f for f in os.listdir(self.features_dir['language'] + 'language') if movie_name in f and '_features_' in f])
                
                for partition in lang_partitions:
                    partition_base = partition.split('_features_')[0]
                    
                    with h5py.File(os.path.join(self.features_dir['language'], 'language', f"{partition_base}_features_language.h5"), 'r') as f:
                        try:
                            stimuli_features['language'][partition_base] = f[partition_base]['language_pooler_output'][:]
                        except:
                            print("error movie lang")
                            f.visit(lambda x: print(x))
                            sys.exit()
        fmri_data = {}
        for i in self.subject:
            fmri_data[f"sub{i}"] = load_fmri(self.fmri_dir, i)

        # import sys; sys.exit()
        # fmri_data = load_fmri(self.fmri_dir, self.subject)x
        # fmri = []
        # for key in fmri_data.keys():
        #     data = fmri_data[key]
        #     fmri.append(data)
        # fmri = np.concatenate(fmri, axis=0)
        # print('fMRI data shape: ', fmri.shape)
        # check_fmri_centering(fmri_data=fmri)
        # fmri_data = normalize_across_episodes(fmri_data)

        x, self.aligned_fmri_sub01 = align_features_and_fmri_samples(
            stimuli_features, 
            fmri_data["sub1"], 
            self.excluded_samples_start, 
            self.excluded_samples_end, 
            self.hrf_delay, 
            self.stimulus_window, 
            self.movies
        )
        del x
        x, self.aligned_fmri_sub02 = align_features_and_fmri_samples(
            stimuli_features, 
            fmri_data["sub2"], 
            self.excluded_samples_start, 
            self.excluded_samples_end, 
            self.hrf_delay, 
            self.stimulus_window, 
            self.movies
        )
        del x
        x, self.aligned_fmri_sub03 = align_features_and_fmri_samples(
            stimuli_features, 
            fmri_data["sub3"], 
            self.excluded_samples_start, 
            self.excluded_samples_end, 
            self.hrf_delay, 
            self.stimulus_window, 
            self.movies
        )
        del x
        self.aligned_features, self.aligned_fmri_sub05 = align_features_and_fmri_samples(
            stimuli_features, 
            fmri_data["sub5"], 
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
            'fmri': {
                'sub01': self.aligned_fmri_sub01[idx],
                'sub02': self.aligned_fmri_sub02[idx],
                'sub03': self.aligned_fmri_sub03[idx],
                'sub05': self.aligned_fmri_sub05[idx]
            },
        }
    
    def get_raw_stimuli(self):
        return self.raw_stimuli
   

vision_dir = '/home/pranav/mihir/algonauts_challenge/internvl3_8b_8bit/'
# vision_dir = '/home/pranav/mihir/algonauts_challenge/AlgonautsDS-features/developer_kit/stimulus_features/raw/'
# audio_dir = '/home/pranav/mihir/algonauts_challenge/AlgonautsDS-features/developer_kit/stimulus_features/raw/'
audio_dir = '/home/pranav/mihir/algonauts_challenge/whisper/'
lang_dir = '/home/pranav/mihir/algonauts_challenge/AlgonautsDS-features/developer_kit/stimulus_features/raw/'
features_dir = {
    "visual": vision_dir,
    "audio": audio_dir,
    "language": lang_dir
}
fmri_dir = '/home/pranav/mihir/algonauts_challenge/algonauts_2025.competitors/fmri/'
# movies_train = ["friends-s01"]
# movies_train = ["movie10-bourne", "movie10-figures", "movie10-life", "movie10-wolf"]
movies_train = ["friends-s01", "friends-s02", "friends-s03", "friends-s04", "friends-s05", "movie10-bourne", "movie10-figures", "movie10-life", "movie10-wolf"]
movies_val = ["friends-s06"]
modality = "all"  #@param ["visual", "audio", "language", "all"]

excluded_samples_start = 5  #@param {type:"slider", min:0, max:20, step:1}
excluded_samples_end = 5  #@param {type:"slider", min:0, max:20, step:1}
hrf_delay = 0 #default: 3  #@param {type:"slider", min:0, max:10, step:1}
stimulus_window = 15

# subject = [1]
subject = [1, 2, 3, 5] #@param ["1", "2", "3", "5"] {type:"raw", allow-input: true}



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
    k = fmri.keys()
    print('fmri: ', k)
    print(f"fMRI_sub01: {fmri['sub01'].shape}")
    print(f"fMRI_sub02: {fmri['sub02'].shape}")
    print(f"fMRI_sub03: {fmri['sub03'].shape}")
    print(f"fMRI_sub05: {fmri['sub05'].shape}")
    break




class MultiModalFusion(L.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.latent_dim = config['latent_dim']
        self.codebook_size = config['codebook_size']
        self.learning_rate = config['learning_rate']
        self.dropout_prob = config['dropout_prob']
        self.encoder_dropout_prob = config['encoder_dropout_prob']
        self.num_layers = config['num_layers']
        self.stimulus_window = config['stimulus_window']
        self.weight_decay = config['weight_decay']
        self.alpha = config['alpha']
        self.vision_proj_dim = config['vision_proj_dim']
        self.audio_proj_dim = config['audio_proj_dim']
        self.num_attn_heads = config['num_attn_heads']
        self.subjects = config['subjects']
        self.decay_factor = config['decay_factor']

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

        self.audio_cross_attn = nn.MultiheadAttention(
            embed_dim=self.latent_dim,
            num_heads=self.num_attn_heads,
            dropout=self.encoder_dropout_prob, # Use configured dropout
            batch_first=True
        )
        self.video_cross_attn = nn.MultiheadAttention(
            embed_dim=self.latent_dim,
            num_heads=self.num_attn_heads,
            dropout=self.encoder_dropout_prob,
            batch_first=True
        )
        # Layer to combine the two context-aware outputs
        self.combine_proj = nn.Linear(self.latent_dim * 2, self.latent_dim)
        self.layer_norm1 = nn.LayerNorm(self.latent_dim)
        self.layer_norm2 = nn.LayerNorm(self.latent_dim)
        # self.fusion_layer = nn.Sequential(
        #     nn.Linear((self.latent_dim + (self.latent_dim // 2)), self.latent_dim),
        #     nn.ReLU(),
        #     nn.Dropout(self.dropout_prob),
        #     nn.Linear(self.latent_dim, self.latent_dim)
        # )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim, 
            nhead=self.num_attn_heads,
            dim_feedforward=self.latent_dim,
            dropout=self.encoder_dropout_prob,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )

        self.pos_embedding = nn.Parameter(torch.zeros(1, self.stimulus_window, self.latent_dim))

        # self.fmri_proj = nn.Linear(self.latent_dim, 1000)
        self.fmri_proj = nn.ModuleDict({
            'sub0'+str(i): nn.Linear(self.latent_dim, 1000) for i in self.subjects
        })
        
        self.save_hyperparameters()

    def forward(self, video, audio, text):
        vision_features = self.vision_proj(video)
        audio_features = self.audio_proj(audio.squeeze())

        audio_context, _ = self.audio_cross_attn(query=audio_features, key=vision_features, value=vision_features)
        video_context, _ = self.video_cross_attn(query=vision_features, key=audio_features, value=audio_features)
        audio_enhanced = audio_features + audio_context 
        video_enhanced = vision_features + video_context

        audio_enhanced = self.layer_norm1(audio_enhanced)
        video_enhanced = self.layer_norm2(video_enhanced)
        combined_features = torch.cat((audio_enhanced, video_enhanced), dim=-1)

        # fused_features = torch.cat((vision_features, audio_features), dim=-1)
        fused_emb = self.combine_proj(combined_features)
        # vision_features += self.pos_embedding
        fused_emb += self.pos_embedding

        logits = self.transformer_encoder(fused_emb) #Shared backbone
        # fmri_recon = self.fmri_proj(logits.mean(dim=1))
        return logits
    
    # def pearson_loss(self, pred, target, epsilon=1e-6):
    #     pred_mean = torch.mean(pred, dim=1, keepdim=True)
    #     target_mean = torch.mean(target, dim=1, keepdim=True)
    #     pred_centered = pred - pred_mean
    #     target_centered = target - target_mean

    #     cov = torch.sum(pred_centered * target_centered, dim=1)
    #     pred_std = torch.sqrt(torch.sum(pred_centered**2, dim=1) + epsilon)
    #     target_std = torch.sqrt(torch.sum(target_centered**2, dim=1) + epsilon)

    #     corr = cov / (pred_std * target_std)
    #     return -torch.mean(corr) # Minimize negative correlation -> Maximize correlation

    def training_step(self, batch, batch_idx):
        vision, audio, text, fmri = batch['video'], batch['audio'], batch['language'], batch['fmri']


        logits = self(vision, audio, text)
        multi_mse, multi_mae, multi_pearson_r, multi_cosine = [], [], [], []
        for i in self.subjects:
            rec_fmri = self.fmri_proj[f'sub0{i}'](logits[:, -1, :])
            mae, mse, _, pearson_r, _ = calculate_metrics(
                pred=rec_fmri,
                target=fmri[f'sub0{i}'],
            )
            cosine_loss = (1 - F.cosine_similarity(rec_fmri, fmri[f'sub0{i}'], dim=1)).mean()
            self.log(f"sub0{i}/train_mse", mse, on_step=False, on_epoch=True, prog_bar=False)
            self.log(f"sub0{i}/train_mae", mse, on_step=False, on_epoch=True, prog_bar=False)
            self.log(f"sub0{i}/train_pearson_r", mse, on_step=False, on_epoch=True, prog_bar=False)
            self.log(f"sub0{i}/train_cosine", mse, on_step=False, on_epoch=True, prog_bar=False)
            multi_mae.append(mae)
            multi_mse.append(mse)
            multi_pearson_r.append(pearson_r)
            multi_cosine.append(cosine_loss)

        # loss = self.alpha * mse + ((1-self.alpha) * cosine_loss)
        loss = sum(multi_mse)
        self.log("multisub_train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        # pearson_loss = self.pearson_loss(recon_fmri, fmri)
        # self.log("train_pearson_loss", pearson_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("sub01/train_mse", mse_01, on_step=False, on_epoch=True, prog_bar=False)
        # self.log("sub02/train_mse", mse_02, on_step=False, on_epoch=True, prog_bar=False)
        # self.log("sub03/train_mse", mse_03, on_step=False, on_epoch=True, prog_bar=False)
        # self.log("sub05/train_mse", mse_05, on_step=False, on_epoch=True, prog_bar=False)
        # self.log("train_mae", mae, on_step=False, on_epoch=True, prog_bar=False)
        # # self.log("train_r2", r2, on_step=False, on_epoch=True, prog_bar=False)
        # self.log("train_pearson_r", pearson_r, on_step=False, on_epoch=True, prog_bar=False)
        # self.log("train/cosine_loss", cosine_loss, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        vision, audio, text, fmri = batch['video'], batch['audio'], batch['language'], batch['fmri']
        # print(vision.shape, audio.shape, text.shape, fmri.shape)
        logits = self(vision, audio, text)
        multi_mse, multi_mae, multi_pearson_r, multi_cosine = [], [], [], []
        for i in self.subjects:
            rec_fmri = self.fmri_proj[f'sub0{i}'](logits[:, -1, :])
            mae, mse, _, pearson_r, _ = calculate_metrics(
                pred=rec_fmri,
                target=fmri[f'sub0{i}'],
            )
            cosine_loss = (1 - F.cosine_similarity(rec_fmri, fmri[f'sub0{i}'], dim=1)).mean()
            self.log(f"sub0{i}/val_mse", mse, on_step=False, on_epoch=True, prog_bar=False)
            self.log(f"sub0{i}/val_mae", mse, on_step=False, on_epoch=True, prog_bar=False)
            self.log(f"sub0{i}/val_pearson_r", mse, on_step=False, on_epoch=True, prog_bar=False)
            self.log(f"sub0{i}/val_cosine", mse, on_step=False, on_epoch=True, prog_bar=False)
            multi_mae.append(mae)
            multi_mse.append(mse)
            multi_pearson_r.append(pearson_r)
            multi_cosine.append(cosine_loss)

        # loss = self.alpha * mse + ((1-self.alpha) * cosine_loss)
        val_loss = sum(multi_mse)
        self.log("multisub_val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True)
        # val_pear_avg = 0
        # for i in subject:

        #     recon_fmri = self.fmri_proj[str(i)](logits[:, -1, :]) #Extracting last hidden state

        #     mae, mse, _, pearson_r, _ = calculate_metrics(
        #         pred=recon_fmri,
        #         target=fmri[f'sub0{i}'],
        #     )
        #     # pearson_loss = self.pearson_loss(recon_fmri, fmri)
        #     # self.log("val_pearson_loss", pearson_loss, on_step=False, on_epoch=True, prog_bar=True)
        #     prefix = f"sub0{i}"
        #     cosine_loss = (1 - F.cosine_similarity(recon_fmri, fmri[f'sub0{i}'], dim=1)).mean()
        #     loss = (self.alpha * mse) + ((1 - self.alpha) * cosine_loss)
        #     val_pear_avg += pearson_r
        #     self.log(f"{prefix}/val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        #     self.log(f"{prefix}/val_mse", mse, on_step=False, on_epoch=True, prog_bar=False)
        #     self.log(f"{prefix}/val_mae", mae, on_step=False, on_epoch=True, prog_bar=False)
        #     # self.log("val_r2", r2, on_step=False, on_epoch=True, prog_bar=False)
        #     self.log(f"{prefix}/val_pearson_r", pearson_r, on_step=False, on_epoch=True, prog_bar=False)
        #     self.log(f"{prefix}/val_cosine_loss", cosine_loss, on_step=False, on_epoch=True, prog_bar=False)
        
        # val_pear_avg /= len(self.subjects)
        # self.log("val_pearson_ravg", val_pear_avg, on_epoch=True, on_step=False, prog_bar=False)

    def on_train_epoch_end(self):
        """
        Log the current learning rate at the end of each training epoch.
        """
        # Get the optimizer
        optimizer = self.optimizers()
        if optimizer:
            # Access the learning rate from the optimizer's parameter groups
            # Assuming a single optimizer and parameter group
            current_lr = optimizer.param_groups[0]['lr']
            self.log('learning_rate', current_lr, on_step=False, on_epoch=True, prog_bar=False)

    
    def configure_optimizers(self):
        """
        Configure the optimizer.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        
        # Reduce learning rate when validation loss plateaus
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     mode='max',
        #     factor=0.1,
        #     patience=5,
        #     verbose=True,
        #     min_lr=self.learning_rate * 0.01
        # )

        scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.learning_rate * 0.01
        )

        # scheduler = CosineAnnealingWarmRestarts(
        #     optimizer=optimizer,
        #     T_0=self.trainer.max_epochs // 2,
        #     T_mult=1,
        #     eta_min=self.learning_rate * 0.01
        # )

        # scheduler = CosineAnnealingWarmDecayedRestarts(
        #     optimizer=optimizer,
        #     T_0= self.trainer.max_epochs // 2,
        #     T_mult = 2,
        #     eta_min=self.learning_rate * 0.01,
        #     decay=self.decay_factor
        # )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                # "monitor": "val_pearson_r",
                "interval": "epoch",
                "frequency": 1
            },
        }


project = "multialgonauts-transformer"
# run_name = "test"
# run_name = "cos1NoCentFus_1024emb_15sw_5lr_drop1"
run_name = "baseline"
wandb_logger = WandbLogger(
    project=project,
    name=run_name,
    dir="/home/pranav/mihir/algonauts_challenge/algonauts2025/wandb_logs"
)
checkpoint_callback = ModelCheckpoint(
    dirpath=f'/home/pranav/mihir/algonauts_challenge/algonauts2025/checkpoints/{run_name}',
    filename='{step:04d}-{val_pearson_r:.4f}',
    monitor='val_pearson_ravg',
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
    'learning_rate': 5e-6,
    'dropout_prob': 0.1,
    'encoder_dropout_prob': 0.1,
    'num_layers': 6,
    'num_attn_heads': 8,
    'stimulus_window': stimulus_window,
    'weight_decay': 0.01,
    'alpha': 1.0,
    'subjects': subject,
    'hrf_delay': hrf_delay,
    'decay_factor': 0.2,
    'epochs': epochs
}


model = MultiModalFusion(config)

# vid = torch.randn([32, 15, 3584])
# aud = torch.randn([32, 15, 1, 1280])
# text = torch.randn(32, 768)

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
)
# ckpt_path = "/home/pranav/mihir/algonauts_challenge/algonauts2025/checkpoints/fus_5l_12sw_5lr_drop1/step=97152-val_pearson_r=0.2384.ckpt"
ckpt_path = None
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path if ckpt_path else None)
wandb.finish()


