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
    print(f"fMRI: {fmri.shape}")
    break

# InternVL3 bs, sw, 3584
# Whisper bs, sw, 1, 1280

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
        self.language_proj_dim = config['vision_proj_dim']
        self.num_attn_heads = config['num_attn_heads']
        self.subjects = config['subjects']
        self.decay_factor = config['decay_factor']
        self.warmup_epochs = config['warmup_epochs']
        self.minlr_mult = config['minlr_mult']

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
        # self.no_text_embedding = nn.Parameter(torch.randn(768))

        self.audio_vis_cross_attn = nn.MultiheadAttention(
            embed_dim=self.latent_dim,
            num_heads=self.num_attn_heads,
            dropout=self.encoder_dropout_prob,
            batch_first=True
        )

        self.vision_aud_cross_attn = nn.MultiheadAttention(
            embed_dim=self.latent_dim,
            num_heads=self.num_attn_heads,
            dropout=self.encoder_dropout_prob,
            batch_first=True
        )

        self.av_fusion_norm = nn.LayerNorm(self.latent_dim)

        self.text_condition_attn = nn.MultiheadAttention(
            embed_dim=self.latent_dim,
            num_heads=self.num_attn_heads,
            dropout=self.encoder_dropout_prob,
            batch_first=True
        )
        self.text_condition_norm = nn.LayerNorm(self.latent_dim)

        # fusion_enc = nn.TransformerEncoderLayer(
        #     d_model=self.latent_dim,
        #     nhead=4, 
        #     dim_feedforward=self.latent_dim * 2,
        #     dropout=self.encoder_dropout_prob,
        #     batch_first=True
        # )

        # self.fusion_transformer = nn.TransformerEncoder(
        #     fusion_enc,
        #     num_layers=2
        # )



        # self.audio_cross_attn = nn.MultiheadAttention(
        #     embed_dim=self.latent_dim,
        #     num_heads=self.num_attn_heads,
        #     dropout=self.encoder_dropout_prob, # Use configured dropout
        #     batch_first=True
        # )
        # self.video_cross_attn = nn.MultiheadAttention(
        #     embed_dim=self.latent_dim,
        #     num_heads=self.num_attn_heads,
        #     dropout=self.encoder_dropout_prob,
        #     batch_first=True
        # )
        # # Layer to combine the two context-aware outputs
        # self.combine_proj = nn.Linear(self.latent_dim * 2, self.latent_dim)
        # self.layer_norm1 = nn.LayerNorm(self.latent_dim)
        # self.layer_norm2 = nn.LayerNorm(self.latent_dim)
        # self.fusion_layer = nn.Sequential(
        #     nn.Linear((self.latent_dim + (self.latent_dim // 2)), self.latent_dim),
        #     nn.ReLU(),
        #     nn.Dropout(self.dropout_prob),
        #     nn.Linear(self.latent_dim, self.latent_dim)
        # )

        temp_enc = nn.TransformerEncoderLayer(
            d_model=self.latent_dim, 
            nhead=self.num_attn_heads,
            dim_feedforward=self.latent_dim,
            dropout=self.encoder_dropout_prob,
            batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temp_enc,
            num_layers=self.num_layers
        )

        self.pos_embedding = nn.Parameter(torch.zeros(1, self.stimulus_window, self.latent_dim))

        # self.fmri_proj = nn.Linear(self.latent_dim, 1000)
        self.fmri_proj = nn.Linear(self.latent_dim, 1000)
        
        self.save_hyperparameters()

    def forward(self, video, audio, text):
        batch_size = video.shape[0]
        # text_nan_mask = torch.isnan(text.sum(dim=-1))

        # text[text_nan_mask] = self.no_text_embedding

        vision_p = self.vision_proj(video)
        audio_p = self.audio_proj(audio.squeeze())
        lang_p = self.language_proj(text).unsqueeze(1)

        ctx_vis, _ = self.vision_aud_cross_attn(query=vision_p, key=audio_p, value=vision_p)
        vis_enh = vision_p + ctx_vis
        ctx_aud, _ = self.vision_aud_cross_attn(query=audio_p, key=vision_p, value=audio_p)
        aud_enh = audio_p + ctx_aud

        fused_av_stack = self.av_fusion_norm(vis_enh + aud_enh)

        txt_cxt, _ = self.text_condition_attn(query=fused_av_stack, key=lang_p, value=lang_p)
        cond_seq = self.text_condition_norm(fused_av_stack + txt_cxt)


        # modal_stack = torch.stack([vision_p, audio_p, lang_p], dim=2) # bs, sw, 3, latent_dim
        # fusion_input = modal_stack.view(-1, 3, self.latent_dim) # bs * sw, 3, latent_dim
        # fusion_out = self.fusion_transformer(fusion_input)[:, 0, :].view(batch_size, self.stimulus_window, self.latent_dim)



        # audio_context, _ = self.audio_cross_attn(query=audio_features, key=vision_features, value=vision_features)
        # video_context, _ = self.video_cross_attn(query=vision_features, key=audio_features, value=audio_features)
        # audio_enhanced = audio_features + audio_context 
        # video_enhanced = vision_features + video_context

        # audio_enhanced = self.layer_norm1(audio_enhanced)
        # video_enhanced = self.layer_norm2(video_enhanced)
        # combined_features = torch.cat((audio_enhanced, video_enhanced), dim=-1)

        # # fused_features = torch.cat((vision_features, audio_features), dim=-1)
        # fused_emb = self.combine_proj(combined_features)
        # vision_features += self.pos_embedding
        cond_seq += self.pos_embedding

        logits = self.temporal_encoder(cond_seq) #Shared backbone
        # fmri_recon = self.fmri_proj(logits.mean(dim=1))
        fmri_recon = self.fmri_proj(logits[:, -1, :]) 
        return fmri_recon

    
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
        recon_fmri = self(vision, audio, text)

        mae, mse, _, pearson_r, _ = calculate_metrics(
            pred=recon_fmri,
            target=fmri,
        )

        cosine_loss = (1 - F.cosine_similarity(recon_fmri, fmri, dim=1)).mean()
        loss = self.alpha * mse + ((1-self.alpha) * cosine_loss)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        # pearson_loss = self.pearson_loss(recon_fmri, fmri)
        # self.log("train_pearson_loss", pearson_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_mse", mse, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_mae", mae, on_step=False, on_epoch=True, prog_bar=False)
        # self.log("train_r2", r2, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_pearson_r", pearson_r, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/cosine_loss", cosine_loss, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        vision, audio, text, fmri= batch['video'], batch['audio'], batch['language'], batch['fmri']
        audio = audio.squeeze()
        recon_fmri = self(vision, audio, text)

        mae, mse, _, pearson_r, _ = calculate_metrics(
            pred=recon_fmri,
            target=fmri,
        )
        # pearson_loss = self.pearson_loss(recon_fmri, fmri)
        # self.log("val_pearson_loss", pearson_loss, on_step=False, on_epoch=True, prog_bar=True)
        cosine_loss = (1 - F.cosine_similarity(recon_fmri, fmri, dim=1)).mean()
        loss = (self.alpha * mse) + ((1 - self.alpha) * cosine_loss)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_mse", mse, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_mae", mae, on_step=False, on_epoch=True, prog_bar=False)
        # self.log("val_r2", r2, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_pearson_r", pearson_r, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/cosine_loss", cosine_loss, on_step=False, on_epoch=True, prog_bar=False)

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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.05,
            patience=10,
            # verbose=True,
            min_lr=self.learning_rate * 0.01
        )

        # warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        #     optimizer,
        #     start_factor=0.1,
        #     end_factor=1.0,
        #     total_iters=self.warmup_epochs
        # )

        # scheduler = CosineAnnealingLR(
        #     optimizer=optimizer,
        #     # T_max=self.trainer.max_epochs,
        #     T_max=self.trainer.max_epochs - self.warmup_epochs,
        #     eta_min=self.learning_rate * self.minlr_mult
        # )

        # combined_scheduler = torch.optim.lr_scheduler.SequentialLR(
        #     optimizer,
        #     schedulers=[warmup_scheduler, scheduler],
        #     milestones=[self.warmup_epochs]
        # )

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
                "monitor": "val_pearson_r",
                "interval": "epoch",
                "frequency": 1
            },
        }


project = "algonauts-posttrain"
# run_name = "test"
# run_name = "cos1NoCentFus_1024emb_15sw_5lr_drop1"
run_name = "baseline+lr6e8"
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
epochs = 20
config = {
    'latent_dim': 1024,
    'codebook_size': 1000,
    'vision_proj_dim': 1024,
    'audio_proj_dim': 1024,
    'learning_rate': 6e-8,
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

ckpt_path = "/home/mihirneal/Developer/algonauts/algonauts2025/checkpoints/algonauts-friendsonly/basellama_warm3_drop3_lr3e5/step=83400-val_pearson_r=0.2842.ckpt"
model = MultiModalFusion.load_from_checkpoint(ckpt_path)

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

ckpt_path = None
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path if ckpt_path else None)
wandb.finish()


