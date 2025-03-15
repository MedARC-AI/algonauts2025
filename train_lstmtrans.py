from collections import defaultdict
import os
import h5py
import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.model_summary import ModelSummary
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import zipfile
import datetime
import socket
from utils import load_fmri, align_features_and_fmri_samples, align_features_and_fmri_samples_friends_s7, CosineLRSchedulerWithWarmup, calculate_metrics, AlgonautsDataset


seed = 42
np.random.seed(seed)
torch.manual_seed(seed)  # Choose any number
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

run_name = "spa_featsel5wind_linproj"

print("=== Job Info ===")
print(datetime.datetime.now().strftime("%a %b %d %I:%M:%S %p UTC %Y"))
print(f"Hostname: {socket.gethostname()}")
print(f"IP: {socket.gethostbyname(socket.gethostname())}")
print(f"User: mihirneal")
print(f"Working Directory: {os.getcwd()}")
print(f"Run Name: {run_name}")

print("\n=== System Info ===")
cpu_info = os.popen('lscpu | grep "Model name" | cut -f 2 -d ":"').read().strip()
print(f"CPU:{cpu_info}")
mem_info = os.popen("free -h | grep Mem | awk '{print $2}'").read().strip()
print("Memory: " + mem_info)
print(f"GPU: {os.popen('nvidia-smi --query-gpu=gpu_name --format=csv,noheader').read().strip()}")
cuda_ver = os.popen('nvidia-smi | grep "CUDA Version"').read().strip()
cuda_ver = cuda_ver.replace("|", "").strip()
print(f"CUDA Version: {cuda_ver}")

print("\n=== GPU Status ===")
print(datetime.datetime.now().strftime("%a %b %d %H:%M:%S %Y"))
os.system('nvidia-smi')

print("\n=== Starting Training ===\n")


features_dir = '/home/pranav/mihir/algonauts_challenge/AlgonautsDS-features/developer_kit/stimulus_features/raw/'
fmri_dir = '/home/pranav/mihir/algonauts_challenge/algonauts_2025.competitors/fmri/'
# movies_train = ["friends-s01"]
movies_train = ["friends-s01", "friends-s02", "friends-s03", "friends-s04", "friends-s05", "movie10-bourne", "movie10-figures", "movie10-life", "movie10-wolf"]
movies_val = ["friends-s06"]
modality = "all"  #@param ["visual", "audio", "language", "all"]

excluded_samples_start = 5  #@param {type:"slider", min:0, max:20, step:1}
excluded_samples_end = 5  #@param {type:"slider", min:0, max:20, step:1}
hrf_delay = 3  #@param {type:"slider", min:0, max:10, step:1}
stimulus_window = 10

subject = 1 #@param ["1", "2", "3", "5"] {type:"raw", allow-input: true}

train_ds = AlgonautsDataset(features_dir, fmri_dir, movies=movies_train, subject=subject, excluded_samples_start=excluded_samples_start, excluded_samples_end=excluded_samples_end, hrf_delay=hrf_delay, stimulus_window=stimulus_window)
val_ds = AlgonautsDataset(features_dir, fmri_dir, movies=movies_val, subject=subject, excluded_samples_start=excluded_samples_start, excluded_samples_end=excluded_samples_end, hrf_delay=hrf_delay, stimulus_window=stimulus_window)

print("Train dataset length: ", len(train_ds))
print("Validation dataset length: ", len(val_ds))


class SpatialAttention(nn.Module):
    def __init__(self, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attn = nn.Linear()
        



class FMRIPredictor(L.LightningModule):
    def __init__(self, config: dict):
        """
        Initialize the fMRI prediction model.
        
        Args:
            V (int): Vocabulary size of fMRI tokens (default: 1000).
            learning_rate (float): Learning rate for the optimizer (default: 1e-3).
        """
        super().__init__()
        self.out_features = config['out_features']  # Vocabulary size
        self.learning_rate = config['learning_rate']
        
        # Audio processing: LSTM to summarize sequence
        self.audio_lstm = nn.LSTM(
            input_size=128, 
            hidden_size=256, 
            num_layers=1, 
            batch_first=True
        )
        self.audio_norm = nn.LayerNorm(128)
        self.audio_proj = nn.Linear(128, 256)
        self.audio_attn = nn.MultiheadAttention(embed_dim=128, num_heads=4)
        
        # Video processing: Reduce dimensionality, then LSTM
        # self.video_linear = nn.Linear(8192, 512)
        self.vision_heads = 64
        assert 8192 % self.vision_heads == 0
        self.spatial_vision_proj = nn.Linear(128, 64)
        self.spatial_vision_attn = nn.MultiheadAttention(embed_dim=64, num_heads=4)
        self.temporal_vision_dim = self.vision_heads * 64
        self.temporal_vision_norm = nn.LayerNorm(self.temporal_vision_dim)
        self.temporal_vision_attn = nn.MultiheadAttention(embed_dim=self.temporal_vision_dim, num_heads=8)
        self.vision_proj = nn.Linear(self.temporal_vision_dim, 1024)
        # self.video_lstm = nn.LSTM(
        #     input_size=512, 
        #     hidden_size=512, 
        #     num_layers=1, 
        #     batch_first=True
        # )
        self.text_linear = nn.Linear(768, 256)
        self.text_norm = nn.LayerNorm(768)
        # Learnable CLS token for fusion
        # self.cls_token = nn.Parameter(torch.randn(1, 512))
        
        # # Transformer encoder for modality fusion
        # self.transformer_encoder = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(
        #         d_model=512, 
        #         nhead=8, 
        #         dim_feedforward=2048, 
        #         dropout=0.1
        #     ), 
        #     num_layers=1
        # )
        
        # Prediction head: Output logits for 32 tokens
        self.pred_head = nn.Linear(1536, self.out_features)
        self.max_lr = config['learning_rate']
        self.min_lr = config['learning_rate'] * 0.01
        self.warmup_steps = config['warmup_steps']
        self.weight_decay = config['weight_decay']

        self.save_hyperparameters()
    
    def forward(self, audio, video, text):
        batch_size, seq_len, _ = audio.size()
        text_missing = torch.isnan(text).any(dim=1)  # Shape: (batch_size,)
        text = torch.nan_to_num(text, nan=0.0)  # Shape: (batch_size, 768)
        
        audio = audio.transpose(0, 1)  # (seq_len, batch_size, 128)
        temporal_audio, _ = self.audio_attn(audio, audio, audio)
        temporal_audio = temporal_audio.transpose(0, 1)  # (batch_size, seq_len, 128)
        audio_summary = temporal_audio.mean(dim=1)  # (batch_size, 128)
        audio_norm = self.audio_norm(audio_summary)  # (batch_size, 128)
        audio_proj = self.audio_proj(audio_norm)  # (batch_size, 256)
        
        # print("audio_proj shape: ", audio_proj.shape)
        
        # _, (audio_hidden, _) = self.audio_lstm(audio)  # (1, batch_size, 256)
        # audio_summary = audio_hidden[-1]  # (batch_size, 256)
        # audio_summary = self.audio_proj(audio_summary)  # (batch_size, 512)

        vision = video.reshape(batch_size, seq_len, self.vision_heads, -1).permute(0, 2, 1, 3).reshape(batch_size * self.vision_heads, seq_len, -1)
        vision_groups = self.spatial_vision_proj(vision).transpose(0, 1)  # (seq_len, batch_size * self.vision_heads, 64)
        vision_groups, _ = self.spatial_vision_attn(vision_groups, vision_groups, vision_groups)  # (seq_len, batch_size * self.vision_heads, 64)
        vision_groups = vision_groups.transpose(0, 1)  # (batch_size * self.vision_heads, seq_len, 64)
        vision_groups = vision_groups.reshape(batch_size, self.vision_heads, seq_len, -1).permute(0, 2, 1, 3)  # (batch_size, self.vision_heads, seq_len, 64)
        spatial_vision = vision_groups.reshape(batch_size, seq_len, -1)  # (batch_size, seq_len, 64 * self.vision_heads)

        spatial_vision = self.temporal_vision_norm(spatial_vision).transpose(0, 1)  # (seq_len, batch_size, 64 * self.vision_heads)
        temporal_vision, _ = self.temporal_vision_attn(spatial_vision, spatial_vision, spatial_vision)  # (seq_len, batch_size, 64 * self.vision_heads)
        temporal_vision = temporal_vision.transpose(0, 1)  # (batch_size, seq_len, 64 * self.vision_heads)
        vision_pool = temporal_vision.mean(dim=1)  # (batch_size, 64 * self.vision_heads)
        vision_proj = self.vision_proj(vision_pool)  # (batch_size, 1024)
        # video_reduced = self.video_linear(video)  # (batch_size, 5, 512)
        # _, (video_hidden, _) = self.video_lstm(video_reduced)  # (1, batch_size, 512)
        # video_summary = video_hidden[-1]  # (batch_size, 512)
        # print("vision_proj shape: ", vision_proj.shape)
        # Step 5: Project text embedding to common dimension
        text_norm = self.text_norm(text)  # (batch_size, 768)
        text_proj = self.text_linear(text_norm)  # (batch_size, 256)
        # print("text_proj shape: ", text_proj.shape)
        # Step 6: Prepare input for fusion with a CLS token
        concat_vector = torch.cat([audio_proj, vision_proj, text_proj], dim=1)  # (batch_size, 1536)
        # cls_token = self.cls_token.expand(batch_size, -1)  # (batch_size, 512)
        # fusion_input = torch.stack(
        #     [cls_token, audio_summary, video_summary, text_proj], 
        #     dim=1
        # )  # (batch_size, 4, 512)
        
        # # Step 7: Create attention mask to ignore text when missing
        # attn_mask = torch.zeros(batch_size, 4, dtype=torch.bool, device=audio.device)
        # attn_mask[:, 3] = text_missing  # Mask text position (index 3) if NaN was present
        
        # # Step 8: Fuse modalities with Transformer encoder
        # fusion_output = self.transformer_encoder(
        #     fusion_input.transpose(0, 1),  # (4, batch_size, 512)
        #     src_key_padding_mask=attn_mask
        # )  # (4, batch_size, 512)
        # cls_output = fusion_output[0]  # (batch_size, 512)
        assert not torch.isnan(concat_vector).any(), "cls_output contains NaN"
        # Step 9: Predict fMRI tokens
        pred = self.pred_head(concat_vector)  # (batch_size, out_features)
        return pred
    
    def training_step(self, batch, batch_idx):
        """
        Training step: Compute loss for a batch.
        """
        audio, video, text, fmri_tokens = batch['audio'], batch['video'], batch['language'], batch['fmri']
        logits = self(audio, video, text)  # (batch_size, 32, V)
        # CrossEntropyLoss expects (N, C) and (N,) targets
        # loss = F.cross_entropy(logits.view(-1, self.V), fmri_tokens.view(-1))
        mae, mse, r2, pearson_r, _ = calculate_metrics(
            pred=logits,
            target=fmri_tokens,
        )
        self.log('train/loss', mse, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/mae', mae, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train/r2', r2, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train/pearson_r', pearson_r, on_step=True, on_epoch=True, prog_bar=False)
        
        return mse
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step: Compute loss for validation.
        """
        audio, video, text, fmri_tokens = batch['audio'], batch['video'], batch['language'], batch['fmri']
        logits = self(audio, video, text)
        # loss = F.cross_entropy(logits.view(-1, self.V), fmri_tokens.view(-1))
        loss = F.mse_loss(logits, fmri_tokens)
        mae, mse, r2, pearson_r, fig = calculate_metrics(
            pred=logits,
            target=fmri_tokens,
        )

        if batch_idx == 0 and not debug:
            self.logger.experiment.log({
                "val/pearson_r": pearson_r,
                "val/mae": mae,
                "val/r2": r2,
                "val/loss": loss,
                "val/reconstruction": fig,
                "global_step": self.global_step,
            })
        else:
            self.log('val/loss', loss, on_epoch=True, prog_bar=True)
            self.log('val/mae', mae, on_epoch=True, prog_bar=False)
            self.log('val/r2', r2, on_epoch=True, prog_bar=False)
            self.log('val/pearson_r', pearson_r, on_epoch=True, prog_bar=False)

    def on_after_backward(self):
    # Compute the gradient norm before clipping
        grad_norm = 0
        for p in self.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        self.log('train/grad_norm', grad_norm, on_step=True)
    
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
        
        # Linear learning rate decay
        scheduler = CosineLRSchedulerWithWarmup(
            optimizer,
            max_lr=self.max_lr,
            min_lr=self.min_lr,
            warmup_steps=self.warmup_steps,
            max_steps=self.trainer.estimated_stepping_batches,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            },
        }


config = {
    'out_features': 1000,
    'learning_rate': 1e-4,
    'warmup_steps': 5000,
    'weight_decay': 0.01
}
model = FMRIPredictor(config)
audio = torch.randn(4, 5, 128)
video = torch.randn(4, 5, 8192)
text = torch.randn(4, 768)
# text_mask = torch.randint(0, 2, (4,))
# fmri_tokens = torch.randint(0, 1000, (1, 32))
out = model(audio, video, text)
print("fmri tokens shape: ", out.shape)

project = "algonauts_transformer"
wandb_logger = WandbLogger(
    project=project,
    name=run_name,
    dir="/home/pranav/mihir/algonauts_challenge/algonauts2025/wandb_logs"
)
checkpoint_callback = ModelCheckpoint(
    dirpath=f'/home/pranav/mihir/algonauts_challenge/algonauts2025/checkpoints/{run_name}',
    filename='{step:04d}-{val_pearson_r:.4f}',
    monitor='val/pearson_r',
    mode='max',
    save_top_k=1,
    save_last=True,
)

early_stopping = EarlyStopping(
    monitor='val/pearson_r',
    patience=5,
    mode='max',
    verbose=True,
    min_delta=1e-4
)

train_loader = DataLoader(train_ds,
                          batch_size=32, 
                          num_workers=8,
                          pin_memory=True, 
                          prefetch_factor=2,
                          persistent_workers=True
                    )

val_loader = DataLoader(val_ds, 
                        batch_size=32, 
                        num_workers=8, 
                        pin_memory=True, 
                        prefetch_factor=2, 
                        persistent_workers=True
                    )

torch.set_float32_matmul_precision('high')
summary = ModelSummary(model, max_depth=2)

debug = False
trainer = L.Trainer(
    accelerator='auto',
    devices=1,
    max_epochs=10,
    callbacks=[checkpoint_callback, early_stopping],
    logger=wandb_logger if not debug else None,
    precision='bf16-mixed',
    log_every_n_steps=1,
)

trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
wandb.finish()