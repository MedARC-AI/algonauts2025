import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from vqvae import VectorQuantize

class ResidualBlock(nn.Module):
   def __init__(self, in_dim, out_dim):
       super().__init__()
       self.downsample = in_dim != out_dim
       self.net = nn.Sequential(
           nn.Linear(in_dim, out_dim),
           nn.LayerNorm(out_dim),
           nn.GELU(),
           nn.Linear(out_dim, out_dim),
           nn.LayerNorm(out_dim),
           nn.GELU()
       )
       if self.downsample:
           self.proj = nn.Linear(in_dim, out_dim)
   
   def forward(self, x):
       if self.downsample:
           return self.proj(x) + self.net(x)
       return x + self.net(x)

class Encoder(nn.Module):
   def __init__(self, input_dim=1000, hidden_dims=[512, 384, 256], num_tokens=32, codebook_dim=64):
       super().__init__()
       
       # Initial projection with one residual block
       self.input_proj = ResidualBlock(input_dim, hidden_dims[0])
       
       # Main network with one residual block per layer
       layers = []
       for i in range(len(hidden_dims)-1):
           layers.append(ResidualBlock(hidden_dims[i], hidden_dims[i+1]))
       self.layers = nn.Sequential(*layers)
       
       # Project to token space with one residual block
       self.token_proj = ResidualBlock(hidden_dims[-1], num_tokens * codebook_dim)
       
       self.num_tokens = num_tokens
       self.codebook_dim = codebook_dim
       
   def forward(self, x):
       x = self.input_proj(x)
       x = self.layers(x)
       x = self.token_proj(x)
       return x.view(x.shape[0], self.num_tokens, self.codebook_dim)

class Decoder(nn.Module):
   def __init__(self, output_dim=1000, hidden_dims=[256, 384, 512], num_tokens=32, codebook_dim=64):
       super().__init__()
       
       # Process tokens with one residual block
       self.token_proj = ResidualBlock(num_tokens * codebook_dim, hidden_dims[0])
       
       # Main network with one residual block per layer
       layers = []
       for i in range(len(hidden_dims)-1):
           layers.append(ResidualBlock(hidden_dims[i], hidden_dims[i+1]))
       self.layers = nn.Sequential(*layers)
       
       # Final projection with one residual block
       self.output_proj = ResidualBlock(hidden_dims[-1], output_dim)
       
   def forward(self, x):
       # x shape: [batch_size, num_tokens, codebook_dim] 
       x = x.reshape(x.shape[0], -1)  # Flatten tokens
       x = self.token_proj(x)
       x = self.layers(x)
       return self.output_proj(x)

class VQVAE(L.LightningModule):
    def __init__(
            self, 
            input_dim=1000, 
            hidden_dims=[512, 384, 256], 
            num_tokens=32, 
            codebook_size=1024, 
            codebook_dim=8,
            commitment_weight=0.25,
            quantizer_decay=0.99,
            learning_rate=3e-4,
            weight_decay=0.01
            ):
        super().__init__()
        
        self.encoder = Encoder(input_dim, hidden_dims, num_tokens, codebook_dim)
        self.decoder = Decoder(input_dim, hidden_dims[::-1], num_tokens, codebook_dim)
        self.quantizer = VectorQuantize(
                dim=codebook_dim,
                codebook_size=codebook_size,
                decay=quantizer_decay,
                commitment_weight=commitment_weight
                )
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_hyperparameters()
        
    def forward(self, x):
        z = self.encoder(x)
        z_q, indices, commitment_loss = self.quantizer(z)
        x_recon = self.decoder(z_q)
        return x_recon, commitment_loss, indices
    
    def encode(self, x):
        z = self.encoder(x)
        _, indices, _ = self.quantizer(z)
        return indices
        
    def decode(self, indices):
        z_q = self.quantizer.get_codes_from_indices(indices)
        return self.decoder(z_q)
    