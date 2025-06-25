"""
Utility functions for Algonauts fMRI reconstruction model.
"""

import os
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from typing import Dict, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm


def is_interactive():
    import __main__ as main

    return not hasattr(main, "__file__")

def to_builtin(o):
    if isinstance(o, (np.generic,)):       # np.float32, np.int64, …
        return o.item()
    if isinstance(o, torch.Tensor) and o.ndim == 0:
        return o.item()
    raise TypeError  

def load_fmri(root_data_dir: str, subject: int) -> Dict[str, np.ndarray]:
    """
    Load the fMRI responses for the selected subject.
    
    Args:
        root_data_dir: Root data directory containing fMRI data
        subject: Subject ID (1, 2, 3, or 5)
    
    Returns:
        Dictionary containing fMRI responses for each movie/episode
    """
    fmri = {}
    
    # Load Friends fMRI data
    fmri_file = f'sub-0{subject}_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5'
    fmri_path = os.path.join(root_data_dir, f'sub-0{subject}', 'func', fmri_file)
    
    with h5py.File(fmri_path, 'r') as f:
        for key, val in f.items():
            # Remove the prefix to get episode identifier
            episode_key = str(key[13:])  # Remove 'task-friends-' prefix
            fmri[episode_key] = val[:].astype(np.float32)
    
    # Load Movie10 fMRI data
    fmri_file = f'sub-0{subject}_task-movie10_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_bold.h5'
    fmri_path = os.path.join(root_data_dir, f'sub-0{subject}', 'func', fmri_file)
    
    with h5py.File(fmri_path, 'r') as f:
        for key, val in f.items():
            # Remove the prefix to get movie identifier
            movie_key = key[13:]  # Remove 'task-movie10-' prefix
            fmri[movie_key] = val[:].astype(np.float32)
    
    # Average the fMRI responses across repeats for 'figures'
    all_keys = list(fmri.keys())
    figures_splits = 12
    for s in range(figures_splits):
        movie = f'figures{s+1:02d}'
        keys_movie = [k for k in all_keys if movie in k]
        if len(keys_movie) == 2:
            fmri[movie] = ((fmri[keys_movie[0]] + fmri[keys_movie[1]]) / 2).astype(np.float32)
            del fmri[keys_movie[0]]
            del fmri[keys_movie[1]]
    
    # Average the fMRI responses across repeats for 'life'
    all_keys = list(fmri.keys())
    life_splits = 5
    for s in range(life_splits):
        movie = f'life{s+1:02d}'
        keys_movie = [k for k in all_keys if movie in k]
        if len(keys_movie) == 2:
            fmri[movie] = ((fmri[keys_movie[0]] + fmri[keys_movie[1]]) / 2).astype(np.float32)
            del fmri[keys_movie[0]]
            del fmri[keys_movie[1]]
    
    return fmri


def align_features_and_fmri_samples(
    features: Dict[str, Dict[str, np.ndarray]], 
    fmri: Dict[str, np.ndarray],
    excluded_samples_start: int,
    excluded_samples_end: int,
    hrf_delay: int,
    stimulus_window: int,
    movies: list
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Align stimulus features with fMRI response samples.
    
    Args:
        features: Dictionary containing stimulus features for each modality
        fmri: Dictionary containing fMRI responses
        excluded_samples_start: Number of initial TRs to exclude
        excluded_samples_end: Number of final TRs to exclude
        hrf_delay: Hemodynamic response function delay in TRs
        stimulus_window: Number of feature chunks to use for modeling each TR (visual/audio only)
        movies: List of movies to process
    
    Returns:
        Tuple of (aligned_features, aligned_fmri)
    """
    ### HRF delay ###
    # The hrf_delay parameter accounts for temporal offset between stimulus
    # presentation and the peak BOLD response. It shifts the stimulus timeline
    # backward relative to the fMRI timeline.
    #
    # Example: With hrf_delay=3, fMRI TR 20 will be modeled using stimulus 
    # features from around TR 17 (20 - 3), because the BOLD signal at TR 20
    # primarily reflects neural activity that occurred ~3 TRs earlier.

    ### Stimulus window ###
    # The stimulus_window parameter determines how many consecutive stimulus
    # feature chunks are used to model each fMRI TR. Instead of using only
    # the single stimulus chunk corresponding to the HRF-delayed timepoint,
    # this creates a temporal window that captures stimulus information
    # leading up to that timepoint.
    #
    # This accounts for the fact that neural responses can integrate information
    # over time, and the BOLD response reflects cumulative neural activity.
    #
    # Example: With hrf_delay=3 and stimulus_window=5, fMRI TR 20 will be
    # modeled using stimulus chunks [13, 14, 15, 16, 17], which represents
    # the 5 stimulus chunks leading up to and including the timepoint that
    # corresponds to the peak BOLD response (TR 17 = 20 - 3).
    #
    # Note: stimulus_window only applies to visual and audio features, since
    # language features are already extracted using transcript words spanning
    # multiple movie chunks, so each fMRI TR uses only the corresponding
    # language feature chunk (with HRF delay applied).
    
    # Initialize empty containers for available modalities
    aligned_features = {}
    for mod in features.keys():
        aligned_features[mod] = []
    
    aligned_fmri = np.empty((0, 1000), dtype=np.float32)
    
    # Process each movie
    for movie in movies:
        # Get movie identifier
        if movie.startswith('friends'):
            movie_id = movie[8:]  # e.g., 's01' from 'friends-s01'
        else:
            movie_id = movie[8:]  # e.g., 'bourne' from 'movie10-bourne'
        
        # Find all splits for this movie
        movie_splits = [key for key in fmri if movie_id in key[:len(movie_id)]]
        
        # Process each split
        for split in movie_splits:
            # Extract fMRI data for this split
            fmri_split = fmri[split]
            if excluded_samples_end > 0:
                fmri_split = fmri_split[excluded_samples_start:-excluded_samples_end]
            else:
                fmri_split = fmri_split[excluded_samples_start:]
            aligned_fmri = np.append(aligned_fmri, fmri_split, axis=0)
            
            # Align features for each fMRI sample
            for s in range(len(fmri_split)):
                # Process each modality
                for mod in features.keys():
                    # Visual and audio features use stimulus window
                    if mod == 'visual' or mod == 'audio':
                        # Calculate indices for stimulus window
                        if s < (stimulus_window + hrf_delay):
                            idx_start = excluded_samples_start
                            idx_end = idx_start + stimulus_window
                        else:
                            idx_start = s + excluded_samples_start - hrf_delay - stimulus_window + 1
                            idx_end = idx_start + stimulus_window
                        
                        # Handle case where features are shorter than expected
                        if idx_end > len(features[mod][split]):
                            idx_end = len(features[mod][split])
                            idx_start = max(0, idx_end - stimulus_window)
                        
                        # Extract features
                        f = features[mod][split][idx_start:idx_end]
                        aligned_features[mod].append(f)
                    
                    # Language features only use single sample
                    elif mod == 'language':
                        # Calculate index with HRF delay
                        if s < hrf_delay:
                            idx = excluded_samples_start
                        else:
                            idx = s + excluded_samples_start - hrf_delay
                        
                        # Handle case where there are fewer language samples
                        if idx >= len(features[mod][split]) - hrf_delay:
                            f = features[mod][split][-1, :]
                        else:
                            f = features[mod][split][idx]
                        
                        aligned_features[mod].append(f)
    
    # Convert to numpy arrays
    for mod in aligned_features:
        if aligned_features[mod]:  # Only convert if not empty
            aligned_features[mod] = np.asarray(aligned_features[mod], dtype=np.float32)
    
    return aligned_features, aligned_fmri

def align_features_and_fmri_samples_friends_s7(
    features_friends_s7,
    root_data_dir, 
    hrf_delay=3, 
    stimulus_window=5,
):
    """
    Align stimulus features with fMRI samples for Friends S7, with proper normalization.
    
    Parameters
    ----------
    features_friends_s7 : dict
        Dictionary containing stimulus features for Friends season 7.
    root_data_dir : str
        Root data directory.
    hrf_delay : int
        Hemodynamic response function delay.
    stimulus_window : int  
        Number of feature chunks for visual/audio.
        
    Returns
    -------
    aligned_features_friends_s7 : dict
        Aligned and normalized stimulus features.
    """

    ### Empty results dictionary ###
    aligned_features_friends_s7 = {}

    ### Loop over subjects ###
    subjects = [1, 2, 3, 5]
    desc = "Aligning stimulus and fMRI features of the four subjects"
    for sub in tqdm(subjects, desc=desc):
        aligned_features_friends_s7[f'sub-0{sub}'] = {}

        ### Load the Friends season 7 fMRI samples ###
        samples_dir = os.path.join(root_data_dir, f'sub-0{sub}', 'target_sample_number',
            f'sub-0{sub}_friends-s7_fmri_samples.npy')
        fmri_samples = np.load(samples_dir, allow_pickle=True).item()

        ### Loop over Friends season 7 episodes ###
        for epi, samples in fmri_samples.items():
            features_epi = {"visual": [], "audio": []}
            
            # Only include language if it exists in features
            if "language" in features_friends_s7:
                features_epi["language"] = []

            ### Loop over fMRI samples ###
            for s in range(samples):
                ### Loop across modalities ###
                for mod in features_friends_s7.keys():
                    if mod not in features_epi:
                        continue  # Skip if modality not expected

                    ### Visual and audio features ###
                    # If visual or audio modality, model each fMRI sample using
                    # the N stimulus feature samples up to the fMRI sample of
                    # interest minus the hrf_delay (where N is defined by the
                    # 'stimulus_window' variable)
                    if mod == 'visual' or mod == 'audio':
                        # In case there are not N stimulus feature samples up to
                        # the fMRI sample of interest minus the hrf_delay (where
                        # N is defined by the 'stimulus_window' variable), model
                        # the fMRI sample using the first N stimulus feature
                        # samples
                        if s < (stimulus_window + hrf_delay):
                            idx_start = 0
                            idx_end = idx_start + stimulus_window
                        else:
                            idx_start = s - hrf_delay - stimulus_window + 1
                            idx_end = idx_start + stimulus_window
                        
                        # In case there are less visual/audio feature samples
                        # than fMRI samples minus the hrf_delay, use the last N
                        # visual/audio feature samples available (where N is
                        # defined by the 'stimulus_window' variable)
                        if idx_end > len(features_friends_s7[mod][epi]):
                            idx_end = len(features_friends_s7[mod][epi])
                            idx_start = max(0, idx_end - stimulus_window)
                        
                        # Ensure we don't go below 0
                        idx_start = max(0, idx_start)
                        
                        f = features_friends_s7[mod][epi][idx_start:idx_end]
                        features_epi[mod].append(f)

                    ### Language features ###
                    # Since language features already consist of embeddings
                    # spanning several samples, only model each fMRI sample
                    # using the corresponding stimulus feature sample minus the
                    # hrf_delay
                    elif mod == 'language':
                        # In case there are no language features for the fMRI
                        # sample of interest minus the hrf_delay, model the fMRI
                        # sample using the first language feature sample
                        if s < hrf_delay:
                            idx = 0
                        else:
                            idx = s - hrf_delay
                        
                        # In case there are fewer language feature samples than
                        # fMRI samples minus the hrf_delay, use the last
                        # language feature sample available
                        if idx >= (len(features_friends_s7[mod][epi]) - hrf_delay):
                            f = features_friends_s7[mod][epi][-1, :]
                        else:
                            f = features_friends_s7[mod][epi][idx]
                        
                        features_epi[mod].append(f)
                
            # Convert lists to numpy arrays
            for mod in features_epi:
                if features_epi[mod]:  # Only convert if not empty
                    features_epi[mod] = np.asarray(features_epi[mod], dtype=np.float32)

            ### Add the episode stimulus features to the features dictionary ###
            aligned_features_friends_s7[f'sub-0{sub}'][epi] = features_epi

    return aligned_features_friends_s7

def calculate_metrics(
    pred: np.ndarray, 
    target: np.ndarray
) -> Tuple[float, float, float, float, None]:
    """
    Calculate regression metrics between predictions and targets.
    
    Args:
        pred: Predicted values
        target: Target values
    
    Returns:
        Tuple of (MAE, MSE, R², Pearson correlation, None)
    """
    # Ensure inputs are numpy arrays
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().float().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().float().cpu().numpy()
    
    # Convert to torch tensors for MAE/MSE calculation
    pred_tensor = torch.from_numpy(pred).float()
    target_tensor = torch.from_numpy(target).float()
    
    # Calculate metrics
    mae = F.l1_loss(pred_tensor, target_tensor).item()
    mse = F.mse_loss(pred_tensor, target_tensor).item()
    r2 = r2_score(target.flatten(), pred.flatten())
    pearson_r = pearsonr(pred.flatten(), target.flatten())[0]
    
    return mae, mse, r2, pearson_r, None


def normalize_fmri(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize fMRI data by computing z-scores for each parcel.
    
    Args:
        data: fMRI data of shape (n_samples, n_parcels)
    
    Returns:
        Tuple of (normalized_data, means, stds)
    """
    assert data.shape[1] == 1000, f"Expected 1000 parcels, got {data.shape[1]}"
    
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    
    # Avoid division by zero
    stds = np.where(stds == 0, 1e-6, stds)
    
    # Standardize
    normalized_data = (data - means) / stds
    
    return normalized_data, means, stds


def check_fmri_stats(fmri_data: np.ndarray, name: str = "fMRI data"):
    """
    Print statistics about fMRI data for debugging.
    
    Args:
        fmri_data: fMRI data array
        name: Name to display in output
    """
    print(f"\n{name} statistics:")
    print(f"  Shape: {fmri_data.shape}")
    print(f"  Mean: {np.mean(fmri_data):.4f}")
    print(f"  Std: {np.std(fmri_data):.4f}")
    print(f"  Min: {np.min(fmri_data):.4f}")
    print(f"  Max: {np.max(fmri_data):.4f}")
    
    # Check per-parcel statistics
    parcel_means = np.mean(fmri_data, axis=0)
    parcel_stds = np.std(fmri_data, axis=0)
    print(f"  Parcel means - min: {np.min(parcel_means):.4f}, max: {np.max(parcel_means):.4f}")
    print(f"  Parcel stds - min: {np.min(parcel_stds):.4f}, max: {np.max(parcel_stds):.4f}")


def load_friends_s7_features(features_dir: Dict[str, Path], use_language=True):
    features = defaultdict(dict)
    season = 's07'
    
    # Load visual and audio features
    audio_dir = features_dir['audio'] / 'audio'
    episode_files = sorted([f for f in os.listdir(audio_dir) 
                           if f"{season}e" in f and '_features_' in f])
    
    for episode_file in episode_files:
        episode_base = episode_file.split('_features_')[0]
        episode_key = episode_base.split('_')[1]
        
        # Visual features
        visual_path = features_dir['visual'] / 'visual' / f"{episode_base}_features_visual.h5"
        with h5py.File(visual_path, 'r') as f:
            features['visual'][episode_key] = f['language_model.model.layers.20.post_attention_layernorm'][:]
        
        # Audio features
        audio_path = features_dir['audio'] / 'audio' / f"{episode_base}_features_audio.h5"
        with h5py.File(audio_path, 'r') as f:
            features['audio'][episode_key] = f['layers.12.fc2'][:]
    
    # Language features
    if use_language:
        lang_path = (features_dir['language'] / 'friends' / 
                    'meta-llama__Llama-3.2-1B' / 'context-long_summary-0.h5')
        with h5py.File(lang_path, 'r') as f:
            for ep in f.keys():
                if ep.startswith(season):
                    features['language'][ep] = f[ep]['model.layers.7'][:]
    
    return dict(features)