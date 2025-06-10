import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
import numpy as np
import lightning as L
from torchmetrics.functional import pearson_corrcoef
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.model_summary import ModelSummary
import h5py
import wandb
from pathlib import Path
# from vector_quantize_pytorch import VectorQuantize
from utils import calculate_metrics

# class ResidualBlock(nn.Module):
#    def __init__(self, in_dim, out_dim):
#        super().__init__()
#        self.downsample = in_dim != out_dim
#        self.net = nn.Sequential(
#            nn.Linear(in_dim, out_dim),
#            nn.LayerNorm(out_dim),
#            nn.GELU(),
#            nn.Linear(out_dim, out_dim),
#            nn.LayerNorm(out_dim),
#            nn.GELU()
#        )
#        if self.downsample:
#            self.proj = nn.Linear(in_dim, out_dim)
   
#    def forward(self, x):
#        if self.downsample:
#            return self.proj(x) + self.net(x)
#        return x + self.net(x)

# class Encoder(nn.Module):
#    def __init__(self, input_dim=1000, hidden_dims=[512, 384, 256], num_tokens=32, codebook_dim=64):
#        super().__init__()
       
#        # Initial projection with one residual block
#        self.input_proj = ResidualBlock(input_dim, hidden_dims[0])
       
#        # Main network with one residual block per layer
#        layers = []
#        for i in range(len(hidden_dims)-1):
#            layers.append(ResidualBlock(hidden_dims[i], hidden_dims[i+1]))
#        self.layers = nn.Sequential(*layers)
       
#        # Project to token space with one residual block
#        self.token_proj = ResidualBlock(hidden_dims[-1], num_tokens * codebook_dim)
       
#        self.num_tokens = num_tokens
#        self.codebook_dim = codebook_dim
       
#    def forward(self, x):
#        x = self.input_proj(x)
#        x = self.layers(x)
#        x = self.token_proj(x)
#        return x.view(x.shape[0], self.num_tokens, self.codebook_dim)

# class Decoder(nn.Module):
#    def __init__(self, output_dim=1000, hidden_dims=[256, 384, 512], num_tokens=32, codebook_dim=64):
#        super().__init__()
       
#        # Process tokens with one residual block
#        self.token_proj = ResidualBlock(num_tokens * codebook_dim, hidden_dims[0])
       
#        # Main network with one residual block per layer
#        layers = []
#        for i in range(len(hidden_dims)-1):
#            layers.append(ResidualBlock(hidden_dims[i], hidden_dims[i+1]))
#        self.layers = nn.Sequential(*layers)
       
#        # Final projection with one residual block
#        self.output_proj = ResidualBlock(hidden_dims[-1], output_dim)
       
#    def forward(self, x):
#        # x shape: [batch_size, num_tokens, codebook_dim] 
#        x = x.reshape(x.shape[0], -1)  # Flatten tokens
#        x = self.token_proj(x)
#        x = self.layers(x)
#        return self.output_proj(x)
   

# class MultiHeadAttention(nn.Module):
#     def __init__(self, dim, num_heads=2, dropout=0.1):
#         super().__init__()
#         assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
        
#         # Separate projections for Q, K, V
#         self.q_proj = nn.Linear(dim, dim)
#         self.k_proj = nn.Linear(dim, dim)
#         self.v_proj = nn.Linear(dim, dim)
        
#         self.out_proj = nn.Linear(dim, dim)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         B, N, C = x.shape  # batch, num_tokens, channels
#         H = self.num_heads

#         # Generate Q, K, V with separate projections
#         q = self.q_proj(x).reshape(B, N, H, self.head_dim).transpose(1, 2)  # B, H, N, head_dim
#         k = self.k_proj(x).reshape(B, N, H, self.head_dim).transpose(1, 2)
#         v = self.v_proj(x).reshape(B, N, H, self.head_dim).transpose(1, 2)

#         # Attention
#         attn = (q @ k.transpose(-2, -1)) * self.scale  # B, H, N, N
#         attn = attn.softmax(dim=-1)
#         attn = self.dropout(attn)
        
#         # Apply attention to V
#         out = (attn @ v).transpose(1, 2).reshape(B, N, C)  # B, N, C
#         out = self.out_proj(out)
#         out = self.dropout(out)
        
#         return out

# class TransformerBlock(nn.Module):
#     def __init__(self, dim, num_heads=2, mlp_ratio=2, dropout=0.1):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(dim)
#         self.norm2 = nn.LayerNorm(dim)
#         self.attn = MultiHeadAttention(dim, num_heads, dropout)
        
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, mlp_hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(mlp_hidden_dim, dim)
#         )
        
#     def forward(self, x):
#         # Pre-norm architecture
#         x = x + self.attn(self.norm1(x))
#         x = x + self.mlp(self.norm2(x))
#         return x

# class CustomViTEncoder(nn.Module):
#     def __init__(
#         self,
#         input_dim=1000,
#         num_tokens=32,
#         token_dim=8,
#         num_layers=4,
#         num_heads=2,
#         dropout=0.1
#     ):
#         super().__init__()
#         self.num_tokens = num_tokens
#         self.token_dim = token_dim
        
#         # Initial projection and reshape
#         self.input_proj = nn.Sequential(
#             nn.Linear(input_dim, num_tokens * token_dim),
#             # nn.LayerNorm(num_tokens * token_dim),
#             # nn.GELU(),
#             # # nn.Dropout(dropout),
#             # nn.Linear(num_tokens * token_dim, num_tokens * token_dim),
#             # # nn.Dropout(dropout)
#         )
        
#         # Positional embedding
#         self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, token_dim) * 0.02)  # smaller init
        
#         # Transformer layers
#         # trans_enc = nn.TransformerEncoderLayer(
#         #     d_model=token_dim,
#         #     nhead=num_heads,
#         #     dim_feedforward=token_dim,
#         #     dropout=dropout,
#         # )
#         # self.transformer_blocks = nn.TransformerEncoder(
#         #     trans_enc,
#         #     num_layers=num_layers
#         # )


#         self.transformer_blocks = nn.ModuleList([
#             TransformerBlock(
#                 dim=token_dim,
#                 num_heads=num_heads,
#                 mlp_ratio=2,
#                 dropout=dropout
#             ) for _ in range(num_layers)
#         ])
        
#         self.norm = nn.LayerNorm(token_dim)
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self, x):
#         # Project and reshape: [B, 1000] -> [B, 32, 8]
#         x = self.input_proj(x)
#         x = x.reshape(-1, self.num_tokens, self.token_dim)
        
#         # Add positional embeddings and dropout
#         x = x + self.pos_embed
#         x = self.dropout(x)
        
#         # Apply transformer blocks
#         # print("input shape: ", x.shape)
#         # x = self.transformer_blocks(x)
#         for block in self.transformer_blocks:
#             x = block(x)
            
#         x = self.norm(x)
#         return x

# # Mirror image of encoder for decoder
# class CustomViTDecoder(nn.Module):
#     def __init__(
#         self,
#         output_dim=1000,
#         num_tokens=32,
#         token_dim=8,
#         num_layers=4,
#         num_heads=2,
#         dropout=0.1
#     ):
#         super().__init__()
#         self.num_tokens = num_tokens
#         self.token_dim = token_dim
        
#         # Positional embedding
#         self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, token_dim) * 0.02)
        
#         # Transformer layers
#         self.transformer_blocks = nn.ModuleList([
#             TransformerBlock(
#                 dim=token_dim,
#                 num_heads=num_heads,
#                 mlp_ratio=2,
#                 dropout=dropout
#             ) for _ in range(num_layers)
#         ])

#         # trans_dec = nn.TransformerEncoderLayer(
#         #     d_model=token_dim,
#         #     nhead=num_heads,
#         #     dim_feedforward=token_dim,
#         #     dropout=dropout,
#         # )
#         # self.transformer_blocks = nn.TransformerEncoder(
#         #     trans_dec,
#         #     num_layers=num_layers
#         # )




#         self.norm = nn.LayerNorm(token_dim)
#         self.dropout = nn.Dropout(dropout)
        
#         # Output projection
#         self.output_proj = nn.Sequential(
#             nn.Linear(num_tokens * token_dim, output_dim),
#             # nn.LayerNorm(num_tokens * token_dim),
#             # nn.GELU(),
#             # # nn.Dropout(dropout),
#             # nn.Linear(num_tokens * token_dim, output_dim)
#         )
        
#     def forward(self, x):
#         # Add positional embeddings and dropout
#         x = x + self.pos_embed
#         x = self.dropout(x)
        
#         # Apply transformer blocks

#         # x = self.transformer_blocks(x)
#         for block in self.transformer_blocks:
#             x = block(x)
            
#         x = self.norm(x)
        
#         # Project back to original dimension
#         x = x.reshape(x.shape[0], -1)
#         x = self.output_proj(x)
        
#         return x

# class VQVAE(L.LightningModule):
#     def __init__(
#             self, 
#             input_dim=1000, 
#             hidden_dims=[512, 384, 256], 
#             num_tokens=32,
#             num_layers=2,
#             codebook_size=1024, 
#             codebook_dim=8,
#             commitment_weight=0.25,
#             quantizer_decay=0.99,
#             learning_rate=3e-4,
#             weight_decay=0.01
#             ):
#         super().__init__()
        
#         self.encoder = CustomViTEncoder(num_layers=num_layers, num_tokens=num_tokens, token_dim=codebook_dim)
#         self.decoder = CustomViTDecoder(num_layers=num_layers, num_tokens=num_tokens, token_dim=codebook_dim)
#         self.quantizer = VectorQuantize(
#                 dim=codebook_dim,
#                 codebook_size=codebook_size,
#                 decay=quantizer_decay,
#                 commitment_weight=commitment_weight
#                 )
#         self.learning_rate = learning_rate
#         self.weight_decay = weight_decay
#         self.save_hyperparameters()
        
#     def forward(self, x):
#         z = self.encoder(x)
#         # print("encoder: ", z.shape)
#         # z_q, indices, commitment_loss = self.quantizer(z)
#         x_recon = self.decoder(z)
#         # return x_recon, commitment_loss, indices
#         return x_recon
    
#     def calculate_metrics(self, x, x_recon):
#         # Flatten the tensors for correlation calculation
#         x = x.to(torch.float64)
#         x_recon = x_recon.to(torch.float64)
#         x_flat = x.reshape(x.shape[0], -1)
#         x_recon_flat = x_recon.reshape(x_recon.shape[0], -1)
        
#         # Calculate Pearson R for each sample in batch
#         correlations = torch.stack([
#             pearson_corrcoef(x_flat[i], x_recon_flat[i])
#             for i in range(x_flat.shape[0])
#         ])
#         avg_pearson_r = correlations.mean()
        
#         # Calculate variance explained
#         total_variance = torch.var(x_flat, dim=1).sum()
#         residual_variance = torch.var(x_flat - x_recon_flat, dim=1).sum()
#         variance_explained = 1 - (residual_variance / total_variance)


#         mse = F.mse_loss(x_recon, x)
#         mae = F.l1_loss(x_recon, x)

        
#         return mae, mse, variance_explained, avg_pearson_r


#     def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
#         x = batch[0]
#         if len(x.shape) == 1:
#             x = x.unsqueeze(0)
#         x_recon = self(x)
#         # x_recon, commitment_loss, _ = self(x)
        
#         # total_loss = recon_loss + commitment_loss

#         mae, mse, train_var, train_pearson_r = self.calculate_metrics(x, x_recon)
        
#         # Log metrics
#         self.log('train_mse', mse)
#         self.log('train_mae', mae)
#         self.log('train_variance', train_var)
#         self.log('train_pearson_r', train_pearson_r)
        
#         return mse
    
#     def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
#         x = batch[0]
#         if len(x.shape) == 1:
#             x = x.unsqueeze(0)
#         x_recon = self(x)
#         # x_recon, commitment_loss, _ = self(x)
        
#         # recon_loss = F.mse_loss(x_recon, x)
#         # total_loss = recon_loss + commitment_loss

#         mae, mse, val_var, val_pearson_r = self.calculate_metrics(x, x_recon)
        
#         # Log metrics
#         self.log('val_mse', mse)
#         self.log('val_mae', mae)
#         self.log('val_variance', val_var)
#         self.log('val_pearson_r', val_pearson_r)
    
#     def configure_optimizers(self):
#         optimizer = torch.optim.AdamW(
#             self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay,
#             betas=(0.9, 0.95), eps=1e-8,
#         )
        
#         # warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
#         #     optimizer,
#         #     start_factor=0.1,
#         #     end_factor=1.0,
#         #     total_iters=self.warmup_epochs
#         # )

#         scheduler = CosineAnnealingLR(
#             optimizer=optimizer,
#             T_max=self.trainer.max_epochs,
#             # T_max=self.trainer.max_epochs - self.warmup_epochs,
#             eta_min=self.learning_rate * 0.01
#         )

#         # combined_scheduler = torch.optim.lr_scheduler.SequentialLR(
#         #     optimizer,
#         #     schedulers=[warmup_scheduler, scheduler],
#         #     milestones=[self.warmup_epochs]
#         # )
        
#         return {
#             "optimizer": optimizer,
#             "lr_scheduler": {
#                 "scheduler": scheduler, "interval": "epoch", "frequency": 1
#             },
#         }


class fMRIAutoencoder(nn.Module):
    """
    A simple feed-forward autoencoder for fMRI data.
    """
    def __init__(self, input_dim=1000, latent_dim=128, hidden_dims=None):
        """
        Args:
            input_dim (int): Dimensionality of the input fMRI data (number of parcels).
            latent_dim (int): Dimensionality of the bottleneck layer.
            hidden_dims (list of int, optional): List of hidden layer dimensions for encoder/decoder.
                                                  If None, a default architecture is used.
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        if hidden_dims is None:
            # Default architecture: 1000 -> 512 -> 256 -> latent_dim
            hidden_dims_encoder = [512, 256]
        else:
            hidden_dims_encoder = hidden_dims

        # --- Encoder ---
        encoder_layers = []
        current_dim = input_dim
        for h_dim in hidden_dims_encoder:
            encoder_layers.append(nn.Linear(current_dim, h_dim))
            encoder_layers.append(nn.ReLU()) # Or nn.LeakyReLU(), nn.GELU()
            # encoder_layers.append(nn.Dropout(0.1)) # Optional: Add dropout for regularization
            current_dim = h_dim
        encoder_layers.append(nn.Linear(current_dim, latent_dim))
        # No activation for the bottleneck layer typically, or a linear one.
        # Some might use nn.Tanh() if data is normalized to [-1, 1]
        self.encoder = nn.Sequential(*encoder_layers)

        # --- Decoder ---
        # Mirrored architecture of the encoder
        decoder_layers = []
        hidden_dims_decoder = hidden_dims_encoder[::-1] # Reverse the hidden_dims list
        current_dim = latent_dim
        for h_dim in hidden_dims_decoder:
            decoder_layers.append(nn.Linear(current_dim, h_dim))
            decoder_layers.append(nn.ReLU()) # Or nn.LeakyReLU(), nn.GELU()
            # decoder_layers.append(nn.Dropout(0.1)) # Optional
            current_dim = h_dim
        decoder_layers.append(nn.Linear(current_dim, input_dim))
        # No activation or Sigmoid/Tanh if input is normalized to [0,1] or [-1,1]
        # For general reconstruction, linear output is common.
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        """
        Forward pass through the autoencoder.
        Args:
            x (torch.Tensor): Input fMRI data, shape (batch_size, input_dim).
        Returns:
            torch.Tensor: Reconstructed fMRI data, shape (batch_size, input_dim).
            torch.Tensor: Latent space representation, shape (batch_size, latent_dim).
        """
        # Ensure input is flattened if it's not already
        x = x.view(x.size(0), -1)
        latent_representation = self.encoder(x)
        reconstruction = self.decoder(latent_representation)
        return reconstruction, latent_representation

# --- 2. PyTorch Lightning Module ---
class fMRILinearAE(L.LightningModule):
    """
    PyTorch Lightning wrapper for the fMRIAutoencoder.
    Handles training, validation, testing, optimizers, and loss.
    """
    def __init__(self, input_dim=1000, latent_dim=128, hidden_dims=None, learning_rate=1e-3, weight_decay=1e-5):
        """
        Args:
            input_dim (int): Dimensionality of the input fMRI data.
            latent_dim (int): Dimensionality of the bottleneck layer.
            hidden_dims (list of int, optional): Hidden layer dimensions for the autoencoder.
            learning_rate (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay (L2 penalty) for the optimizer.
        """
        super().__init__()
        self.save_hyperparameters() # Saves init arguments to self.hparams

        self.autoencoder = fMRIAutoencoder(
            input_dim=self.hparams.input_dim,
            latent_dim=self.hparams.latent_dim,
            hidden_dims=self.hparams.hidden_dims
        )
        # self.criterion = nn.MSELoss() # Mean Squared Error is common for reconstruction
        # self.mae = nn.L1Loss() # Mean Absolute Error (robust to outliers)

    def forward(self, x):
        # Delegate to the autoencoder's forward method
        return self.autoencoder(x)

    def calculate_metrics(self, x, x_recon):
        from scipy.stats import pearsonr
        # Flatten the tensors for correlation calculation
        x = x.to(torch.float64)
        x_recon = x_recon.to(torch.float64)
        x_flat = x.reshape(x.shape[0], -1)
        x_recon_flat = x_recon.reshape(x_recon.shape[0], -1)
        
        # Calculate variance explained
        total_variance = torch.var(x_flat, dim=1).sum()
        residual_variance = torch.var(x_flat - x_recon_flat, dim=1).sum()
        variance_explained = 1 - (residual_variance / total_variance)


        mse = F.mse_loss(x_recon, x)
        mae = F.l1_loss(x_recon, x)
        avg_pearson_r = pearsonr(x_recon.detach().cpu().flatten(), x.detach().cpu().flatten())[0]         
        return mae, mse, variance_explained, avg_pearson_r
    
    def training_step(self, batch, batch_idx):
        x = batch # Assuming batch is (data, labels) or just (data,)
                     # For autoencoders, labels are the input itself if not provided.
                     # If your DataLoader only yields x, then: x = batch[0] or x = batch
        reconstructed_x, latent_z = self(x)
        mae, mse, train_var, train_pearson_r = self.calculate_metrics(x, reconstructed_x)
        # loss = self.criterion(reconstructed_x, x.view(x.size(0), -1))
        # mae = self.mae(reconstructed_x, x.view(x.size(0), -1))

        self.log('train_loss', mse, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_mae', mae, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_variance', train_var, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_pearson_r', train_pearson_r, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        # You can also log other metrics, e.g., Mean Absolute Error if MSE is the primary loss
        # mae_loss = nn.L1Loss()(reconstructed_x, x.view(x.size(0), -1))
        # self.log('train_mae', mae_loss, on_step=False, on_epoch=True)
        return mse 

    def validation_step(self, batch, batch_idx):
        x = batch # Or x = batch[0] or x = batch
        reconstructed_x, latent_z = self(x)
        # print("recon: ", reconstructed_x.shape)
        # print("gt: ", x.shape)
        mae, mse, val_var, val_pearson_r = self.calculate_metrics(x, reconstructed_x)
        # print("pear_r", val_pearson_r)
        # loss = self.criterion(reconstructed_x, x.view(x.size(0), -1))
        # mae = self.mae(reconstructed_x, x.view(x.size(0), -1))
        self.log('val_loss', mse, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_mae', mae, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_variance', val_var, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_pearson_r', val_pearson_r, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        # mae_loss = nn.L1Loss()(reconstructed_x, x.view(x.size(0), -1))
        # self.log('val_mae', mae_loss, on_step=False, on_epoch=True)

    # def test_step(self, batch, batch_idx):
    #     x, _ = batch # Or x = batch[0] or x = batch
    #     if isinstance(x, list):
    #         x = x[0]

    #     reconstructed_x, latent_z = self(x)
    #     loss = self.criterion(reconstructed_x, x.view(x.size(0), -1))
    #     self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    #     # mae_loss = nn.L1Loss()(reconstructed_x, x.view(x.size(0), -1))
    #     # self.log('test_mae', mae_loss, on_step=False, on_epoch=True)
    #     return loss

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        # Optional: Add a learning rate scheduler
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
        # return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.1,
            patience=5,
            min_lr=self.hparams.learning_rate * 0.01
        )
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "monitor": "val_pearson_r",
            "interval": "epoch",
            "frequency": 1
        },
    }



class FMRIDataset(Dataset):
    """
    A PyTorch Dataset class to load fMRI data from one or more H5 files.

    Args:
        h5_files (list or Path): A list of paths to the H5 files to be loaded.
    """
    def __init__(self, h5_files):
        data_list = []
        # Ensure h5_files is a list to handle single or multiple file paths
        if not isinstance(h5_files, list):
            h5_files = [h5_files]
            
        print(f"Loading data from {len(h5_files)} file(s)...")
        for h5_file in h5_files:
            try:
                with h5py.File(h5_file, 'r') as f:
                    # Iterate over keys in the H5 file (e.g., 'data')
                    for key in f.keys():
                        # Load data, convert to a float tensor, and append to the list
                        data = torch.from_numpy(f[key][:]).float()
                        data_list.append(data)
                        # print(f"  - Loaded {data.shape[0]} samples from {h5_file.name} (key: {key})")
            except Exception as e:
                print(f"Error loading {h5_file}: {str(e)}")
        
        # Concatenate all loaded data tensors into a single tensor
        if data_list:
            self.data = torch.cat(data_list, dim=0)
            print(f"Total samples loaded: {self.data.shape[0]}")
        else:
            self.data = torch.empty(0)
            print("Warning: No data was loaded.")

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Returns a single sample from the dataset at the given index."""
        return self.data[idx]

if __name__ == '__main__':
    # Set a seed for reproducibility
    L.seed_everything(42, workers=True)

    # Define the root directory and subjects
    root_dir = Path('/home/mihirneal/Developer/algonauts/')
    subjects = ['01', '02', '03', '05']
    fmri_dir = root_dir / 'algonauts_2025.competitors/fmri'

    # --- Generate the list of all file paths ---
    all_files = []
    for sub in subjects:
        subject_dir = fmri_dir / f'sub-{sub}/func'
        # File path for the 'friends' task data
        friends_file = subject_dir / f'sub-{sub}_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5'
        # File path for the 'movie10' task data
        movie_file = subject_dir / f'sub-{sub}_task-movie10_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_bold.h5'
        
        if friends_file.exists():
            all_files.append(friends_file)
        else:
            print(f"Warning: File not found {friends_file}")
            
        if movie_file.exists():
            all_files.append(movie_file)
        else:
            print(f"Warning: File not found {movie_file}")

    # --- Create the full dataset and split it ---
    if all_files:
        # Instantiate the dataset with all merged data
        full_dataset = FMRIDataset(all_files)

        # Define the split ratio for validation (e.g., 20%)
        val_split_ratio = 0.2
        val_size = int(len(full_dataset) * val_split_ratio)
        train_size = len(full_dataset) - val_size

        # Perform the random split to get training and validation datasets
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        print("\n--- Dataset Summary ---")
        print(f"Total samples in full dataset: {len(full_dataset)}")
        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
    else:
        print("\nCould not create datasets because no files were found.")

    # Create dataloaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")




    run_name = "baseline_multi768dim"
    project = "fmri_linearAE"
    wandb_logger = WandbLogger(
        project=project,
        name=run_name,
        save_dir=root_dir / "algonauts2025/wandb_logs"
    )

    checkpoint_callback = ModelCheckpoint(
                dirpath=root_dir / f'algonauts2025/checkpoints/{project}/{run_name}',
                filename='{epoch:02d}_{val_pearson_r:.3f}',
                monitor='val_pearson_r',
                mode='max',
                save_top_k=1,
                save_last=True
    )

    early_stopping = EarlyStopping(
                monitor='val_pearson_r',
                mode='max',
                patience=5,
                verbose=True,
                min_delta=1e-4
            )
    
    torch.set_float32_matmul_precision('high')

    INPUT_DIM = 1000  # Number of fMRI parcels
    LATENT_DIM = 768   # Desired dimensionality of the latent space (bottleneck)
    # Example: 1000 -> 512 -> 256 -> 128 -> LATENT_DIM
    HIDDEN_DIMS_ENCODER = [768, 768] # Encoder hidden layers
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0.01 # For L2 regularization
    BATCH_SIZE = 64
    MAX_EPOCHS = -1
 
    model = fMRILinearAE(
        input_dim=INPUT_DIM,
        latent_dim=LATENT_DIM,
        hidden_dims=HIDDEN_DIMS_ENCODER,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    summary = ModelSummary(model, max_depth=3)
    print(summary)

    debug = False
    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        precision=32,
        logger=wandb_logger if not debug else None,
        callbacks=[checkpoint_callback, early_stopping]
    )

    ckpt_path = None
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path if ckpt_path else None)
    wandb.finish()
