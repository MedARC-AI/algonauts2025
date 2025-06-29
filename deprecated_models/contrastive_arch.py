import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.utils.data import DataLoader, Dataset
import h5py
import numpy as np
import os
from collections import defaultdict
from pathlib import Path
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


# ========================================================================================
# 1. MODEL DEFINITIONS
# ========================================================================================

class fMRIEncoder(nn.Module):
    """
    A simple MLP to encode fMRI data into the shared latent space.
    Its input dimension is flexible to handle different parcellation schemes.
    """
    def __init__(self, input_dim=1000, latent_dim=1024, hidden_dim=2048):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, fmri_data):
        # fmri_data shape: [batch_size, input_dim]
        embedding = self.model(fmri_data)
        return embedding


class MultiModalFusion(nn.Module):
    """
    This is the STIMULUS ENCODER.
    It performs the complex fusion of video, audio, and text that we designed.
    """
    def __init__(self, config):
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

        # --- Projection Layers ---
        self.vision_proj = nn.Linear(3584, self.latent_dim)
        self.audio_proj = nn.Linear(1280, self.latent_dim)
        self.language_proj = nn.Linear(2048, self.latent_dim)

        # --- Bidirectional Audio-Visual Fusion ---
        self.vision_queries_audio_attn = nn.MultiheadAttention(self.latent_dim, num_heads=8, dropout=self.attn_prob, batch_first=True)
        self.audio_queries_vision_attn = nn.MultiheadAttention(self.latent_dim, num_heads=8, dropout=self.attn_prob, batch_first=True)
        self.av_fusion_norm = nn.LayerNorm(self.latent_dim)

        # --- Text Conditioning ---
        self.text_condition_attn = nn.MultiheadAttention(self.latent_dim, num_heads=8, dropout=self.attn_prob, batch_first=True)
        self.text_condition_norm = nn.LayerNorm(self.latent_dim)

        # --- Temporal Processing ---
        self.pos_embedding = nn.Parameter(torch.zeros(1, config['stimulus_window'], self.latent_dim))
        temporal_encoder_layer = nn.TransformerEncoderLayer(d_model=self.latent_dim, nhead=8, batch_first=True)
        self.temporal_transformer = nn.TransformerEncoder(temporal_encoder_layer, num_layers=4)

    def forward(self, video, audio, text):
        batch_size = video.shape[0]
        stimulus_window = video.shape[1]

        # --- Project Modalities ---
        vision_p = self.vision_proj(video)
        audio_p = self.audio_proj(audio)
        lang_p = self.language_proj(text)

        # --- Bidirectional AV Fusion ---
        ctx_for_vision, _ = self.vision_queries_audio_attn(query=vision_p, key=audio_p, value=audio_p)
        vision_enhanced = vision_p + ctx_for_vision
        ctx_for_audio, _ = self.audio_queries_vision_attn(query=audio_p, key=vision_p, value=vision_p)
        audio_enhanced = audio_p + ctx_for_audio
        fused_av_sequence = self.av_fusion_norm(vision_enhanced + audio_enhanced)

        # --- Condition with Text ---
        lang_p_seq = lang_p.unsqueeze(1)
        text_context, _ = self.text_condition_attn(query=fused_av_sequence, key=lang_p_seq, value=lang_p_seq)
        conditioned_sequence = self.text_condition_norm(fused_av_sequence + text_context)

        # --- Temporal Analysis ---
        conditioned_sequence_with_pos = conditioned_sequence + self.pos_embedding
        hidden_states = self.temporal_transformer(conditioned_sequence_with_pos)
        last_hidden_state = hidden_states[:, -1, :] # Output embedding for the window
        return last_hidden_state


class ContrastiveAlignmentModel(L.LightningModule):
    """
    The main LightningModule for Stage 1.
    It holds both encoders and computes the contrastive loss.
    """
    def __init__(self, stimulus_encoder, fmri_encoder, learning_rate=1e-4, temperature=0.07):
        super().__init__()
        self.save_hyperparameters(ignore=['stimulus_encoder', 'fmri_encoder'])
        self.stimulus_encoder = stimulus_encoder
        self.fmri_encoder = fmri_encoder
        self.temperature = temperature
        self.learning_rate = learning_rate

    def training_step(self, batch, batch_idx):
        video, audio, text, fmri = batch['video'], batch['audio'], batch['language'], batch['fmri']

        # 1. Get embeddings from both encoders
        stimulus_embeds = self.stimulus_encoder(video, audio, text)
        fmri_embeds = self.fmri_encoder(fmri)

        # 2. Normalize embeddings
        stimulus_embeds = F.normalize(stimulus_embeds, p=2, dim=1)
        fmri_embeds = F.normalize(fmri_embeds, p=2, dim=1)

        # 3. Calculate similarity matrix (logits)
        logits = torch.matmul(stimulus_embeds, fmri_embeds.T) / self.temperature
        labels = torch.arange(len(logits), device=self.device)

        # 4. Calculate symmetric loss
        loss_stim_to_fmri = F.cross_entropy(logits, labels)
        loss_fmri_to_stim = F.cross_entropy(logits.T, labels)
        loss = (loss_stim_to_fmri + loss_fmri_to_stim) / 2

        self.log('train_contrastive_loss', loss, on_step=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-2)
        # You can add a scheduler here if you desire, e.g., CosineAnnealingLR
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            verbose=True,
            min_lr= self.learning_rate * 0.01
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }

