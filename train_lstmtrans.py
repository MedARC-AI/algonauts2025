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

run_name = "concat_redfeat_transNOCLS_redLR"

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
stimulus_window = 5

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
        self.vision_dropout = config['vision_dropout']
        self.fmri_dropout = config['fmri_dropout']
        self.transformer_dropout = config['transformer_dropout']
        self.debug = config['debug']
        self.video_proj = nn.Sequential(
            nn.Linear(8192, 2048),
            nn.ReLU(),
            nn.Dropout(self.vision_dropout),
            nn.Linear(2048, 1024),
        )

        self.audio_proj = nn.Sequential(
            nn.Linear(128, 256),
            # nn.ReLU(),
            # nn.Dropout(self.audio_dropout),
        )

        # Audio processing: LSTM to summarize sequence
        # self.audio_lstm = nn.LSTM(
        #     input_size=128, 
        #     hidden_size=256, 
        #     num_layers=1, 
        #     batch_first=True
        # )

        self.vector_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=1536,
                nhead=8,
                dim_feedforward=2048,
                dropout=self.transformer_dropout
            ),
            num_layers=2
        )

        self.text_proj = nn.Linear(768, 256)
        self.concat_norm = nn.LayerNorm(1536)
        # self.audio_norm = nn.LayerNorm(128)
        # self.audio_proj = nn.Linear(128, 256)
        # self.audio_attn = nn.MultiheadAttention(embed_dim=128, num_heads=4)
        
        # # Video processing: Reduce dimensionality, then LSTM
        # # self.video_linear = nn.Linear(8192, 512)
        # self.vision_heads = 64
        # assert 8192 % self.vision_heads == 0
        # self.spatial_vision_proj = nn.Linear(128, 64)
        # self.spatial_vision_attn = nn.MultiheadAttention(embed_dim=64, num_heads=4)
        # self.temporal_vision_dim = self.vision_heads * 64
        # self.temporal_vision_norm = nn.LayerNorm(self.temporal_vision_dim)
        # self.temporal_vision_attn = nn.MultiheadAttention(embed_dim=self.temporal_vision_dim, num_heads=8)
        # self.vision_proj = nn.Linear(self.temporal_vision_dim, 1024)
        # # self.video_lstm = nn.LSTM(
        # #     input_size=512, 
        # #     hidden_size=512, 
        # #     num_layers=1, 
        # #     batch_first=True
        # # )
        # self.text_linear = nn.Linear(768, 256)
        # self.text_norm = nn.LayerNorm(768)
        # # Learnable CLS token for fusion
        # # self.cls_token = nn.Parameter(torch.randn(1, 512))
        
        # # # Transformer encoder for modality fusion
        # # self.transformer_encoder = nn.TransformerEncoder(
        # #     nn.TransformerEncoderLayer(
        # #         d_model=512, 
        # #         nhead=8, 
        # #         dim_feedforward=2048, 
        # #         dropout=0.1
        # #     ), 
        # #     num_layers=1
        # # )
        
        # # Prediction head: Output logits for 32 tokens
        self.pred_head = nn.Sequential(
            nn.Linear(1536, 1024),
            nn.ReLU(),
            nn.Dropout(self.fmri_dropout),
            nn.Linear(1024, 1000),
        )
        self.max_lr = config['learning_rate']
        self.min_lr = config['learning_rate'] * 0.01
        self.warmup_steps = config['warmup_steps']
        self.weight_decay = config['weight_decay']

        self.save_hyperparameters()
    
    def forward(self, audio, video, text):
        batch_size, seq_len, _ = audio.size()
        text_missing = torch.isnan(text).any(dim=1)  # Shape: (batch_size,)
        text = torch.nan_to_num(text, nan=0.0)  # Shape: (batch_size, 768)
        

        vision = self.video_proj(video)
        audio = self.audio_proj(audio)
        text = self.text_proj(text).unsqueeze(1).repeat(1, seq_len, 1) #Shape: (batch_size, seq_len, 256)

        concat_vector = torch.cat([audio, vision, text], dim=2) #Shape: (batch_size, seq_len, 1536)
        concat_vector = self.concat_norm(concat_vector)
        concat_vector = self.vector_transformer(concat_vector)

        hidden_last_layer = concat_vector[:, -1, :].squeeze(1) #Shape: (batch_size, 1536)
        assert not torch.isnan(hidden_last_layer).any(), "hidden_last_layer contains NaN"
        pred = self.pred_head(hidden_last_layer) #Shape: (batch_size, 1000)
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

        if batch_idx == 0 and not self.debug:
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



# audio = torch.randn(4, 5, 128)
# video = torch.randn(4, 5, 8192)
# text = torch.randn(4, 768)
# # text_mask = torch.randint(0, 2, (4,))
# # fmri_tokens = torch.randint(0, 1000, (1, 32))
# out = model(audio, video, text)
# print("fmri tokens shape: ", out.shape)

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
debug = False
config = {
    'out_features': 1000,
    'transformer_dropout': 0.1,
    'vision_dropout': 0.2,
    'fmri_dropout': 0.3,
    'learning_rate': 1e-4,
    'warmup_steps': 5000,
    'weight_decay': 0.01,
    'debug': debug
}
model = FMRIPredictor(config)
torch.set_float32_matmul_precision('high')
summary = ModelSummary(model, max_depth=2)


trainer = L.Trainer(
    accelerator='auto',
    devices=1,
    max_epochs=10,
    callbacks=[checkpoint_callback, early_stopping],
    logger=wandb_logger if not debug else None,
    precision='bf16-mixed',
    # gradient_clip_val=0.75,
    log_every_n_steps=1,
)

trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
wandb.finish()