
import jax
from flax import nnx
import jax.numpy as jnp
import time
import numpy as np
from torch.utils.data import Dataset, DataLoader
import h5py
import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, List, Sequence
from pathlib import Path
from tqdm import tqdm
import torch
import orbax.checkpoint as ocp
import pickle
import einops
import wandb
import shutil
import json


retrain = True # This indicates if we want to retrain the model using the eval detaset, or just use the already saved best checkpoint
path_last_training = "/kaggle/input/t-att-2"
root_data_dir = "/kaggle/input/algonauts2025nsl"

config_model_path = os.path.join(path_last_training, "model_config.json")
best_config_path = os.path.join(path_last_training, "best_config.json")
training_config_path = os.path.join(path_last_training, "training_config.json")




# creating the dataset

def load_fmri(root_data_dir, subject):
    """
    Load the fMRI responses for the selected subject.

    Parameters
    ----------
    root_data_dir : str
        Root data directory.
    subject : int
        Subject used to train and validate the encoding model.

    Returns
    -------
    fmri : dict
        Dictionary containing the  fMRI responses.

    """

    fmri = {}

    ### Load the fMRI responses for Friends ###
    # Data directory
    fmri_file = f'sub-0{subject}_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5'
    fmri_dir = os.path.join(root_data_dir, 'algonauts_2025.competitors',
        'fmri', f'sub-0{subject}', 'func', fmri_file)
    # Load the the fMRI responses
    fmri_friends = h5py.File(fmri_dir, 'r')
    for key, val in fmri_friends.items():
        fmri[str(key[13:])] = val[:].astype(np.float32)
    del fmri_friends

    ### Load the fMRI responses for Movie10 ###
    # Data directory
    fmri_file = f'sub-0{subject}_task-movie10_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_bold.h5'
    fmri_dir = os.path.join(root_data_dir, 'algonauts_2025.competitors',
        'fmri', f'sub-0{subject}', 'func', fmri_file)
    # Load the the fMRI responses
    fmri_movie10 = h5py.File(fmri_dir, 'r')
    for key, val in fmri_movie10.items():
        fmri[key[13:]] = val[:].astype(np.float32)
    del fmri_movie10
    # Average the fMRI responses across the two repeats for 'figures'
    keys_all = fmri.keys()
    figures_splits = 12
    for s in range(figures_splits):
        movie = 'figures' + format(s+1, '02')
        keys_movie = [rep for rep in keys_all if movie in rep]
        fmri[movie] = ((fmri[keys_movie[0]] + fmri[keys_movie[1]]) / 2).astype(np.float32)
        del fmri[keys_movie[0]]
        del fmri[keys_movie[1]]
    # Average the fMRI responses across the two repeats for 'life'
    keys_all = fmri.keys()
    life_splits = 5
    for s in range(life_splits):
        movie = 'life' + format(s+1, '02')
        keys_movie = [rep for rep in keys_all if movie in rep]
        fmri[movie] = ((fmri[keys_movie[0]] + fmri[keys_movie[1]]) / 2).astype(np.float32)
        del fmri[keys_movie[0]]
        del fmri[keys_movie[1]]

    ### Output ###
    return fmri

def align_features_and_fmri_samples(features, fmri, excluded_samples_start,
                                    excluded_samples_end, hrf_delay, stimulus_window, movies):
    """
    Align the stimulus features with the fMRI response samples for the selected movies,
    preallocating the output arrays to avoid duplicating memory.

    Parameters
    ----------
    features : dict
        Dictionary containing the stimulus features.
    fmri : dict
        Dictionary containing the fMRI responses.
    excluded_samples_start : int
        Number of initial fMRI samples to exclude.
    excluded_samples_end : int
        Number of trailing fMRI samples to exclude.
    hrf_delay : int
        The delay (in TRs) to account for the hemodynamic response.
    stimulus_window : int
        Number of consecutive stimulus chunks to use per fMRI sample.
    movies : list
        List of movie identifiers.

    Returns
    -------
    aligned_features : np.ndarray, shape (total_samples, feature_dim)
        The aligned stimulus feature vectors.
    aligned_fmri : np.ndarray, shape (total_samples, fmri_dim)
        The aligned fMRI response samples.
    """
    import numpy as np
    from tqdm import tqdm

    # First pass: compute the total number of fMRI samples and feature vector dimensionality.
    total_samples = 0
    feature_dim = None
    fmri_dim = None  # assume all fmri splits have same number of features per sample

    for movie in movies:
        # Determine movie identifier
        if movie.startswith('friends'):
            movie_id = movie[8:]
        elif movie.startswith('movie10'):
            movie_id = movie[8:]
        else:
            continue

        # Get keys for all splits corresponding to this movie
        movie_splits = [key for key in fmri if movie_id in key[:len(movie_id)]]
        for split in movie_splits:
            # Slice the fMRI data to exclude initial and final samples
            fmri_split = fmri[split][excluded_samples_start:-excluded_samples_end]
            n_samples = fmri_split.shape[0] if hasattr(fmri_split, "shape") else len(fmri_split)
            total_samples += n_samples

            # For the first split we encounter, determine the feature dimension.
            if feature_dim is None:
                sample_feature_dim = 0
                for mod in features:
                    if mod in ['visual', 'audio']:
                        # Expecting features[mod][split] to be 2D: (n, d)
                        d = features[mod][split].shape[1]
                        sample_feature_dim += stimulus_window * d
                    elif mod == 'language':
                        # For language, each sample contributes its own feature vector (assumed to be 1D or 2D)
                        if len(features[mod][split].shape) > 1:
                            d = features[mod][split].shape[1]
                        else:
                            d = 1
                        sample_feature_dim += d
                feature_dim = sample_feature_dim

            # Also determine the fMRI dimension (assume all fMRI splits have same dimensionality)
            if fmri_dim is None:
                fmri_dim = fmri_split.shape[1]

    # Preallocate the final arrays.
    aligned_features = np.empty((total_samples, feature_dim), dtype=np.float32)
    aligned_fmri = np.empty((total_samples, fmri_dim), dtype=np.float32)

    # Second pass: fill in the preallocated arrays.
    current_index = 0
    for movie in movies:
        if movie.startswith('friends'):
            movie_id = movie[8:]
        elif movie.startswith('movie10'):
            movie_id = movie[8:]
        else:
            continue

        movie_splits = [key for key in fmri if movie_id in key[:len(movie_id)]]
        for split in tqdm(movie_splits, desc=f"Processing {movie} splits"):
            # print(split)
            fmri_split = fmri[split][excluded_samples_start:-excluded_samples_end]
            n_samples = fmri_split.shape[0]
            # Directly write the fMRI data into the preallocated array.
            aligned_fmri[current_index:current_index+n_samples, :] = fmri_split

            # For each fMRI sample, build its corresponding feature vector.
            for s in range(n_samples):
                parts = []  # list to collect modality-specific features
                for mod in features:
                    if mod in ['visual', 'audio']:
                        if s < (stimulus_window + hrf_delay):
                            idx_start = excluded_samples_start
                            idx_end = idx_start + stimulus_window
                        else:
                            idx_start = s + excluded_samples_start - hrf_delay - stimulus_window + 1
                            idx_end = idx_start + stimulus_window
                        # Adjust indices if they exceed available samples.
                        if idx_end > len(features[mod][split]):
                            idx_end = len(features[mod][split])
                            idx_start = idx_end - stimulus_window
                        f = features[mod][split][idx_start:idx_end]
                        parts.append(f.flatten())
                    elif mod == 'language':
                        if s < hrf_delay:
                            idx = excluded_samples_start
                        else:
                            idx = s + excluded_samples_start - hrf_delay
                        if idx >= (len(features[mod][split]) - hrf_delay):
                            f = features[mod][split][-1, :]
                        else:
                            f = features[mod][split][idx]
                        parts.append(f.flatten())
                # Concatenate the modality parts into one feature vector.
                aligned_features[current_index + s, :] = np.concatenate(parts)
            current_index += n_samples

    return aligned_features, aligned_fmri

def load_stimulus_features(root_data_dirs: Union[str, List[str]], modality: str, layers: List[str] = None) -> dict:
    features = {modality: {}}

    # Ensure root_data_dirs is a list
    if isinstance(root_data_dirs, str):
        root_data_dirs = [root_data_dirs]

    # Iterate over each root directory provided.
    for root_data_dir in root_data_dirs:
        root_path = Path(root_data_dir)
        # Traverse all .h5 files under the current root directory.
        for h5_file in tqdm(list(root_path.rglob("*.h5")), desc=f"Processing {root_data_dir}"):
            movie_name = h5_file.stem
            # Remove the "friends_" prefix if it exists.
            if movie_name.startswith("friends_"):
                movie_name = movie_name[len("friends_"):]
            
            datasets = []
            with h5py.File(h5_file, 'r') as f:
                # Iterate over all datasets (layers) in the file.
                for layer in f.keys():
                    data = f[layer][:]
                    if layers:
                        if layer in layers:
                            datasets.append(data)
                    else:
                        datasets.append(data)
            
            # If multiple layers exist, concatenate along axis=1.
            if len(datasets) > 1:
                datasets = [np.reshape(item, (item.shape[0], -1)) for item in datasets]
                # print(datasets[0].shape, datasets[1].shape, datasets[2].shape)
                concatenated_features = np.concatenate(datasets, axis=1)
            elif datasets:
                concatenated_features = datasets[0]
            else:
                continue
            
            # If the same movie_name is encountered from multiple directories,
            # you might choose to either overwrite or merge; here we overwrite.
            features[modality][movie_name] = concatenated_features.reshape(concatenated_features.shape[0], -1)

    return features

def print_keys_for_features(root_data_dirs: Union[str, List[str]]) -> List['str']:
    layers = []
    for root_data_dir in root_data_dirs:
        root_path = Path(root_data_dir)
        # Traverse all .h5 files under the current root directory.
        for h5_file in list(root_path.rglob("*.h5")):
            movie_name = h5_file.stem
            # Remove the "friends_" prefix if it exists.
            if movie_name.startswith("friends_"):
                movie_name = movie_name[len("friends_"):]

            with h5py.File(h5_file, 'r') as f:
                # Iterate over all datasets (layers) in the file.
                for layer in f.keys():
                    layers.append(layer)
                print(layers)
                return layers
    
# Load the model configuration
with open(config_model_path, 'r') as f:
    model_config = json.load(f)

with open(best_config_path, 'r') as f:
    best_config = json.load(f)

# Load the training configuration
with open(training_config_path, 'r') as f:
    training_config = json.load(f)

# Export variables as global
for key, value in training_config.items():
    globals()[key] = value


for key, value in best_config.items():
    globals()[key] = value




base_features_dir = "/kaggle/input/"

if retrain:    
    # # These paths are exported from the training config
    # video_features_name = 'internvl3-8b-8bit-features'
    # audio_features_name = 'whisper-features'
    # transcript_features_name = 'modernbert-features'


    # --- Example usage ---

    # Suppose the root_data_dir points to the directory containing all the .h5 files.
    root_data_dirs_audio = [os.path.join(base_features_dir, audio_features_name)]  # adjust this path as needed
    modality_audio = "audio"  # For example, we're loading audio features
    print_keys_for_features(root_data_dirs_audio)
    # layers_audio defined from the training config
    features_audio = load_stimulus_features(root_data_dirs_audio, modality_audio, layers = layers_audio)

    root_data_dirs_visual = [os.path.join(base_features_dir, video_features_name)]  # adjust this path as needed
    modality_visual = "visual" 
    print_keys_for_features(root_data_dirs_visual)
    # layers_visual defined from the training config
    features_visual = load_stimulus_features(root_data_dirs_visual, modality_visual, layers = layers_visual)

    root_data_dirs_transcript = [os.path.join(base_features_dir, transcript_features_name)] 
    modality_transcript = "language"  
    print_keys_for_features(root_data_dirs_transcript)
    # layers_transcript defined from the training config
    features_transcript = load_stimulus_features(root_data_dirs_transcript, modality_transcript, layers = layers_transcript)


    # Combine all the feature on a single variable
    features = {modality_visual: features_visual[modality_visual], modality_audio: features_audio[modality_audio], modality_transcript: features_transcript[modality_transcript]}


class DatasetFeaturesFmri(Dataset):
    def __init__(self, features: dict, modality: str, excluded_samples_start: int, excluded_samples_end: int, hrf_delay: int, stimulus_window: int, movies: list, root_data_dir: str, subject: int, return_concatenated_features: bool = False):
        self.features = features
        self.modality = modality
        self.excluded_samples_start = excluded_samples_start
        self.excluded_samples_end = excluded_samples_end
        self.hrf_delay = hrf_delay
        self.stimulus_window = stimulus_window
        self.movies = movies
        self.root_data_dir = root_data_dir
        self.subject = subject
        self.return_concatenated_features = return_concatenated_features

        self.fmri = load_fmri(self.root_data_dir, self.subject)

        self.visual_features_shape_length = features['visual']['s01e01a'][0].flatten().shape[0]
        self.audio_features_shape_length = features['audio']['s01e01a'][0].flatten().shape[0]
        self.language_features_shape_length = features['language']['s01e01a'][0].flatten().shape[0]

        self.visual_features_shape_indexes = (0, self.visual_features_shape_length * self.stimulus_window)
        self.audio_features_shape_indexes = (self.visual_features_shape_indexes[1], self.visual_features_shape_indexes[1] + self.audio_features_shape_length * self.stimulus_window)
        self.language_features_shape_indexes = (self.audio_features_shape_indexes[1], self.audio_features_shape_indexes[1] + self.language_features_shape_length) # We're not using stimulus_window for language features

        self.features, self.fmri = align_features_and_fmri_samples(self.features, self.fmri, self.excluded_samples_start, self.excluded_samples_end, self.hrf_delay, self.stimulus_window, self.movies)

    def __len__(self):
        return len(self.fmri)
    
    def __getitem__(self, index):
        if self.return_concatenated_features:
            return self.features[index], self.fmri[index]
        else:
            video_features = self.features[index][self.visual_features_shape_indexes[0]:self.visual_features_shape_indexes[1]]
            audio_features = self.features[index][self.audio_features_shape_indexes[0]:self.audio_features_shape_indexes[1]]
            language_features = self.features[index][self.language_features_shape_indexes[0]:self.language_features_shape_indexes[1]]

            video_features = einops.rearrange(video_features, '(w a) -> w a', w = self.stimulus_window)
            audio_features = einops.rearrange(audio_features, '(w a) -> w a', w = self.stimulus_window)
            

            return video_features, audio_features, language_features, self.fmri[index]
    
    def get_features(self):
        return self.features
    
    def get_fmri(self):
        return self.fmri
    
    def get_modality(self):
        return self.modality
    
    def get_excluded_samples_start(self):
        return self.excluded_samples_start
    
    def get_excluded_samples_end(self):
        return self.excluded_samples_end
    
    def get_hrf_delay(self):
        return self.hrf_delay
    
    def get_stimulus_window(self):
        return self.stimulus_window
    
    def get_movies(self):
        return self.movies
    
    def get_root_data_dir(self):
        return self.root_data_dir
    
    def get_subject(self):
        return self.subject


class DatasetFeaturesFmriMultiSubject(Dataset):
    def __init__(self, features: dict, modality: str, # Modality seems unused here?
                 excluded_samples_start: int, excluded_samples_end: int,
                 hrf_delay: int, stimulus_window: int, movies: list,
                 root_data_dir: str, subjects: list[int],
                 return_concatenated_features: bool = False,
                 device: torch.device = torch.device('cpu')): # Add device option

        # Store configuration
        self.features_input_dict = features # Keep original dict if needed later
        self.modality = modality
        self.excluded_samples_start = excluded_samples_start
        self.excluded_samples_end = excluded_samples_end
        self.hrf_delay = hrf_delay
        self.stimulus_window = stimulus_window
        self.movies = movies
        self.root_data_dir = root_data_dir
        self.subjects = sorted(list(set(subjects))) # Store unique sorted subjects
        self.return_concatenated_features = return_concatenated_features
        self.device = device

        # Calculate feature shapes based on provided example feature
        # Assuming 's01e01a' exists and features are consistent across subjects
        try:
            self.visual_features_shape_length = features['visual']['s01e01a'][0].flatten().shape[0]
            self.audio_features_shape_length = features['audio']['s01e01a'][0].flatten().shape[0]
            self.language_features_shape_length = features['language']['s01e01a'][0].flatten().shape[0]
        except KeyError as e:
             raise ValueError(f"Could not find example feature key 's01e01a' in provided features dict: {e}")
        except IndexError as e:
             raise ValueError(f"Example feature 's01e01a' seems empty or has wrong structure: {e}")


        self.visual_features_shape_indexes = (0, self.visual_features_shape_length * self.stimulus_window)
        self.audio_features_shape_indexes = (self.visual_features_shape_indexes[1], self.visual_features_shape_indexes[1] + self.audio_features_shape_length * self.stimulus_window)
        self.language_features_shape_indexes = (self.audio_features_shape_indexes[1], self.audio_features_shape_indexes[1] + self.language_features_shape_length) # No window for language

        # --- Data Loading and Alignment per Subject ---
        all_aligned_features = []
        all_aligned_fmri = []
        all_subject_ids = []
        self._subject_indices_map = {} # Store indices range for each subject

        print(f"Initializing Dataset for subjects: {self.subjects}")
        current_index = 0
        for subj_id in self.subjects:
            print(f"\nProcessing Subject {subj_id}...")
            fmri_subj = load_fmri(self.root_data_dir, subj_id)

            # Align features and fMRI for this specific subject
            features_aligned_subj, fmri_aligned_subj = align_features_and_fmri_samples(
                self.features_input_dict, # Pass the original feature dict
                fmri_subj,
                self.excluded_samples_start,
                self.excluded_samples_end,
                self.hrf_delay,
                self.stimulus_window,
                self.movies
            )

            num_samples_subj = len(fmri_aligned_subj)
            if num_samples_subj > 0:
                all_aligned_features.append(features_aligned_subj)
                all_aligned_fmri.append(fmri_aligned_subj)
                subject_ids_subj = np.full(num_samples_subj, subj_id, dtype=np.int32)
                all_subject_ids.append(subject_ids_subj)

                # Store the index range for this subject
                self._subject_indices_map[subj_id] = (current_index, current_index + num_samples_subj)
                current_index += num_samples_subj
            else:
                 print(f"Warning: No aligned samples found for subject {subj_id}. Skipping.")
                 self._subject_indices_map[subj_id] = (current_index, current_index) # Empty range


        if not all_aligned_features:
             raise ValueError("No data loaded for any subject. Check alignment parameters or data paths.")

        # Concatenate data from all subjects
        # Using torch.from_numpy().to(self.device) for direct tensor creation
        self.features = torch.from_numpy(np.concatenate(all_aligned_features, axis=0)).float().to(self.device)
        self.fmri = torch.from_numpy(np.concatenate(all_aligned_fmri, axis=0)).float().to(self.device)
        self.subject_ids = torch.from_numpy(np.concatenate(all_subject_ids, axis=0)).long().to(self.device) # Use long for IDs

        print(f"\nDataset Initialized. Total samples: {len(self.fmri)}")
        print(f"Feature tensor shape: {self.features.shape}")
        print(f"fMRI tensor shape: {self.fmri.shape}")
        print(f"Subject ID tensor shape: {self.subject_ids.shape}")
        print(f"Subject index map: {self._subject_indices_map}")


    def __len__(self):
        return len(self.fmri)

    def __getitem__(self, index):
        features_sample = self.features[index]
        fmri_sample = self.fmri[index]
        subject_id = self.subject_ids[index] # Get subject ID for this sample

        if self.return_concatenated_features:
            # Return concatenated features, fmri, and subject_id
            return features_sample, fmri_sample, subject_id
        else:
            # Split features into modalities
            video_features = features_sample[self.visual_features_shape_indexes[0]:self.visual_features_shape_indexes[1]]
            audio_features = features_sample[self.audio_features_shape_indexes[0]:self.audio_features_shape_indexes[1]]
            language_features = features_sample[self.language_features_shape_indexes[0]:self.language_features_shape_indexes[1]]

            # Reshape features with windows (if dimensions allow)
            # Use try-except for robustness if a feature type might be missing/empty
            try:
                video_features = einops.rearrange(video_features, '(w d) -> w d', w=self.stimulus_window)
            except Exception as e:
                 # Handle cases where reshape might fail (e.g., length 0 or not divisible)
                 # print(f"Warning: Could not reshape video features at index {index}. Error: {e}")
                 # Decide how to handle: return as is, return zeros, raise error?
                 # Returning as flat tensor for now if reshape fails
                 pass # video_features remains flat

            try:
                audio_features = einops.rearrange(audio_features, '(w d) -> w d', w=self.stimulus_window)
            except Exception as e:
                # print(f"Warning: Could not reshape audio features at index {index}. Error: {e}")
                 pass # audio_features remains flat


            # language_features are returned as is (no window assumed)

            # Return split features, fmri, and subject_id
            return video_features, audio_features, language_features, fmri_sample, subject_id

    def get_subject_indices(self, subject_id: int) -> list[int]:
        """Returns a list of dataset indices corresponding to the given subject_id."""
        if subject_id not in self._subject_indices_map:
            print(f"Warning: Subject ID {subject_id} not found in this dataset.")
            return []
        start, end = self._subject_indices_map[subject_id]
        return list(range(start, end))

    # --- Keep other getter methods as needed ---
    def get_features_shape_lengths(self):
        return {
            'visual': self.visual_features_shape_length,
            'audio': self.audio_features_shape_length,
            'language': self.language_features_shape_length
            }

    def get_subjects(self) -> list[int]:
        """Returns the list of subjects included in this dataset."""
        return self.subjects


class FmriDatasetSingleSubject(Dataset):
    def __init__(self,
                 features: dict,
                 modality: str,
                 excluded_samples_start: int,
                 excluded_samples_end: int,
                 hrf_delay: int,
                 stimulus_window: int,
                 movies: list,
                 root_data_dir: str,
                 subject: int,
                 return_concatenated_features: bool = False):
        super().__init__()
        # ------------- your code exactly as before -----------------------------
        self.features = features
        self.modality = modality
        self.excluded_samples_start = excluded_samples_start
        self.excluded_samples_end = excluded_samples_end
        self.hrf_delay = hrf_delay
        self.stimulus_window = stimulus_window
        self.movies = movies
        self.root_data_dir = root_data_dir
        self.subject = subject
        self.return_concatenated_features = return_concatenated_features

        self.fmri = load_fmri(self.root_data_dir, self.subject)

        self.visual_features_shape_length  = (
            features['visual']['s01e01a'][0].flatten().shape[0]
        )
        self.audio_features_shape_length   = (
            features['audio']['s01e01a'][0].flatten().shape[0]
        )
        self.language_features_shape_length = (
            features['language']['s01e01a'][0].flatten().shape[0]
        )

        self.visual_features_shape_indexes = (
            0,
            self.visual_features_shape_length * self.stimulus_window,
        )
        self.audio_features_shape_indexes  = (
            self.visual_features_shape_indexes[1],
            self.visual_features_shape_indexes[1]
            + self.audio_features_shape_length * self.stimulus_window,
        )
        self.language_features_shape_indexes = (
            self.audio_features_shape_indexes[1],
            self.audio_features_shape_indexes[1]
            + self.language_features_shape_length,  # no window for language
        )

        self.features, self.fmri = align_features_and_fmri_samples(
            self.features,
            self.fmri,
            self.excluded_samples_start,
            self.excluded_samples_end,
            self.hrf_delay,
            self.stimulus_window,
            self.movies,
        )

    # -------------------------------------------------------------------------
    def __len__(self):
        return len(self.fmri)

    def __getitem__(self, idx: int):
        if self.return_concatenated_features:
            sample = (self.features[idx], self.fmri[idx])
        else:
            video_features = self.features[idx][
                self.visual_features_shape_indexes[0]:
                self.visual_features_shape_indexes[1]
            ]
            audio_features = self.features[idx][
                self.audio_features_shape_indexes[0]:
                self.audio_features_shape_indexes[1]
            ]
            language_features = self.features[idx][
                self.language_features_shape_indexes[0]:
                self.language_features_shape_indexes[1]
            ]

            # reshape so that time window dimension is explicit
            video_features   = einops.rearrange(
                video_features, '(w a) -> w a', w=self.stimulus_window
            )
            audio_features   = einops.rearrange(
                audio_features, '(w a) -> w a', w=self.stimulus_window
            )
            sample = (video_features, audio_features, language_features,
                      self.fmri[idx])

        # we **add** the subject id so the outer dataset can expose it
        return (*sample, self.subject)

# ──────────────────────────────────────────────────────────────────────────────
#  Multi-subject dataset  (thin wrapper around N single-subject datasets)
# ──────────────────────────────────────────────────────────────────────────────
from itertools import accumulate
from bisect import bisect_right

class FmriDatasetMultiSubject(Dataset):
    """
    Returns:  (*your_original_sample, subject_id)
    """
    def __init__(self, *, subjects: list[int] = (1, 2, 3, 5), **single_ds_kwargs):
        """
        Parameters
        ----------
        subjects : list[int]
            Which subject IDs to include.
        single_ds_kwargs :
            All the keyword arguments that a single-subject dataset expects,
            **except** 'subject'. They are forwarded internally.
        """
        super().__init__()
        self.datasets = [
            FmriDatasetSingleSubject(subject=sid, **single_ds_kwargs)
            for sid in subjects
        ]
        # pre-compute lengths so we can map a global index to (dataset, local_idx)
        self.cum_sizes = list(accumulate(len(ds) for ds in self.datasets))

    # -------------------------------------------------------------------------
    def __len__(self) -> int:
        return self.cum_sizes[-1]

    def __getitem__(self, global_idx: int):
        # find which underlying dataset this index falls into
        ds_idx = bisect_right(self.cum_sizes, global_idx)
        prev_cum = 0 if ds_idx == 0 else self.cum_sizes[ds_idx - 1]
        local_idx = global_idx - prev_cum
        return self.datasets[ds_idx][local_idx]  # sample already carries subject

    # convenience so you can pull out a single-subject dataset later
    def get_subject_dataset(self, subject_id: int) -> Dataset:
        for ds in self.datasets:
            if ds.subject == subject_id:
                return ds
        raise KeyError(f"Subject {subject_id} not in this dataset.")

# ──────────────────────────────────────────────────────────────────────────────
#  Collate function (stacks tensors, leaves scalars/ints alone)
# ──────────────────────────────────────────────────────────────────────────────
from torch.utils.data._utils.collate import default_collate

def fmri_collate(batch):
    """
    batch = list[tuple(
        video_features, audio_features, language_features,
        fmri, subject_id
    )]
    """
    # everything except 'subject_id' is a tensor you want stacked
    *data, subjects = zip(*batch)
    data = default_collate(list(zip(*data)))  # stack each field
    subjects = torch.tensor(subjects)         # shape [batch]
    return (*data, subjects)

# ──────────────────────────────────────────────────────────────────────────────
#  Flexible DataLoader factory
# ──────────────────────────────────────────────────────────────────────────────
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

def make_dataloader(dataset: FmriDatasetMultiSubject,
                    *,
                    policy: str = "mixed_shuffle",
                    subject_id: int | None = None,
                    batch_size: int = 32,
                    num_workers: int = 0,
                    pin_memory: bool = True) -> DataLoader:
    """
    policy ∈ {"mixed_shuffle", "mixed_sorted",
              "subject_shuffle", "subject_sorted"}
    subject_id must be given for the two "subject_*" policies.
    """
    if policy.startswith("subject") and subject_id is None:
        raise ValueError("You must supply subject_id for a subject-only loader.")

    if policy == "mixed_shuffle":
        sampler = RandomSampler(dataset)
    elif policy == "mixed_sorted":
        sampler = SequentialSampler(dataset)
    elif policy == "subject_shuffle":
        sub_ds  = dataset.get_subject_dataset(subject_id)
        sampler = RandomSampler(sub_ds)
        dataset = sub_ds
    elif policy == "subject_sorted":
        sub_ds  = dataset.get_subject_dataset(subject_id)
        sampler = SequentialSampler(sub_ds)
        dataset = sub_ds
    else:
        raise ValueError(f"Unknown policy '{policy}'")

    return DataLoader(dataset,
                      batch_size=batch_size,
                      sampler=sampler,
                      collate_fn=fmri_collate,
                      num_workers=num_workers,
                      pin_memory=pin_memory)



if retrain:
    # # These variables should be exported from the training_config.pkl file
    # subject = 1
    # multi_subject = True
    # subjects = [1,2,3,5]
    # modality = 'all'
    # excluded_samples_start = 5
    # excluded_samples_end = 5
    # hrf_delay = 3
    # stimulus_window = 6

    movies_train = ["friends-s01", "friends-s02", "friends-s03", "friends-s04", "friends-s05", "movie10-bourne", "movie10-figures", "movie10-wolf", "movie10-life", "friends-s06"]
    movies_val = ["friends-s07"]

    root_data_dir = '/kaggle/input/algonauts2025nsl'



if retrain:
    multi_ds = FmriDatasetMultiSubject(
        features=features,
        modality='all',
        excluded_samples_start=excluded_samples_start,
        excluded_samples_end=excluded_samples_end,
        hrf_delay=hrf_delay,
        stimulus_window=stimulus_window,
        movies=movies_train,
        root_data_dir=root_data_dir,
        subjects=subjects,
        return_concatenated_features=False,
    )


if retrain:
    batch_size = 256
    # All subjects, fully intermixed
    train_dataloader = make_dataloader(multi_ds,
                                policy="mixed_shuffle",
                                batch_size=batch_size)

    for batch in train_dataloader:
        video_features, audio_features, language_features, fmri, subject_ids = batch
        print(video_features.shape, audio_features.shape, language_features.shape, fmri.shape, subject_ids.shape)
        break




class TransformerEncoder(nnx.Module):
    def __init__(
        self, 
        hidden_size: int,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        *,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ) -> None:
        self.norm1 = nnx.LayerNorm(hidden_size, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=hidden_size,
            dropout_rate=dropout_rate,
            broadcast_dropout=False,
            decode=False,
            deterministic=False,
            rngs=rngs,
        )
        self.norm2 = nnx.LayerNorm(hidden_size, rngs=rngs)

        self.mlp = nnx.Sequential(
            nnx.Linear(hidden_size, mlp_dim, rngs=rngs),
            jax.nn.gelu,
            nnx.Dropout(rate=dropout_rate, rngs=rngs),
            nnx.Linear(mlp_dim, hidden_size, rngs=rngs),
            nnx.Dropout(rate=dropout_rate, rngs=rngs),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MultiModalTransformer(nnx.Module):
    def __init__(
        self,
        num_classes: int,
        subjects: Sequence[int],
        stimuli_window: int = 5,
        visual_dim: int = 1000,
        audio_dim: int = 1000,
        text_dim: int = 1000,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        dropout_rate: float = 0.1,
        *,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ) -> None:
        super().__init__()

        # ---------- shared backbone ----------
        self.stimuli_window = stimuli_window
        self.num_modalities = 3
        self.total_tokens   = self.stimuli_window * self.num_modalities

        self.visual_proj = nnx.Linear(visual_dim, hidden_size, rngs=rngs)
        self.audio_proj  = nnx.Linear(audio_dim,  hidden_size, rngs=rngs)
        self.text_proj   = nnx.Linear(text_dim,   hidden_size, rngs=rngs)

        t_init = jax.nn.initializers.truncated_normal(stddev=0.02)
        self.positional_embeddings = nnx.Param(
            t_init(rngs.params(), (1, self.total_tokens + 1, hidden_size), jnp.float32)
        )
        self.dropout   = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.cls_token = nnx.Param(jnp.zeros((1, 1, hidden_size), jnp.float32))

        self.encoder = nnx.Sequential(*[
            TransformerEncoder(
                hidden_size  = hidden_size,
                mlp_dim      = mlp_dim,
                num_heads    = num_heads,
                dropout_rate = dropout_rate,
                rngs         = rngs,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nnx.LayerNorm(hidden_size, rngs=rngs)

        # ---------- subject-specific classifier bank ----------
        self.subject_ids = tuple(int(s) for s in subjects)
        S = len(subjects)

        k_init = jax.nn.initializers.lecun_normal()
        self.W = nnx.Param(k_init(rngs.params(), (S, hidden_size, num_classes),
                                  jnp.float32))                   # [S,H,C]
        self.b = nnx.Param(jnp.zeros((S, num_classes), jnp.float32))  # [S,C]

    # NEW signature: subject_ids is mandatory
    def __call__(
        self,
        x_visual: jax.Array,
        x_audio:  jax.Array,
        x_text:   jax.Array,
        subject_ids: jax.Array,         # shape [B]
        *,
        return_last_hidden_state: bool = False,
    ) -> jax.Array:

        # 1. backbone forward
        v   = self.visual_proj(x_visual)
        a   = self.audio_proj(x_audio)
        t   = self.text_proj(x_text)
        tok = jnp.concatenate([v, a, t], axis=1)                  # [B,T*3,H]

        B = tok.shape[0]
        cls = jnp.tile(self.cls_token, [B, 1, 1])                 # [B,1,H]
        tok = jnp.concatenate([cls, tok], axis=1)                 # prepend
        tok = self.dropout(tok + self.positional_embeddings)

        enc = self.encoder(tok)
        cls_out = self.final_norm(enc)[:, 0]                      # [B,H]

        # 2. vectorised subject-specific head
        subj_bank = jnp.asarray(self.subject_ids, dtype=jnp.int32)  # [S]
        idx = jnp.argmax((subject_ids[:, None] == subj_bank[None, :]),
                         axis=1)
        W_sel = self.W[idx]                                        # [B,H,C]
        b_sel = self.b[idx]                                        # [B,C]
        logits = jnp.einsum('bh,bhc->bc', cls_out, W_sel) + b_sel  # [B,C]

        if return_last_hidden_state:
            return cls_out, logits
        return logits


# ## Define model config

# In[ ]:


# Testing the model
# model = VisionTransformer(num_classes=1000, in_channels=3, img_size=224, patch_size=16, num_layers=12, num_heads=12, mlp_dim=3072, hidden_size=768, dropout_rate=0.1)
# x = jnp.ones((1, 224, 224, 3))
# logits = model(x)

# model_config = {
#     'num_classes': 1000,
#     'subjects': subjects,
#     'stimuli_window': stimulus_window,
#     'visual_dim': multi_ds.datasets[0].visual_features_shape_length,
#     'audio_dim': multi_ds.datasets[0].audio_features_shape_length,
#     'text_dim': multi_ds.datasets[0].language_features_shape_length,
#     'num_layers': 8,
#     'num_heads': 16,
#     'mlp_dim': 3072,
#     'hidden_size': 2048,
#     'dropout_rate': 0.10,
# }


save_ckpt = True
save_ckpt_path = './checkpoints'
##########


model = MultiModalTransformer(**model_config)

def save_ckpt(tag, model, path='./checkpoints', display = False):
    # if path exists/tag exists, remove it
    if os.path.exists(os.path.join(path, tag)):
        shutil.rmtree(os.path.join(path, tag))

    _, state = nnx.split(model)
    pure_dict_state = nnx.to_pure_dict(state)
    if display:
        nnx.display(pure_dict_state)
    checkpointer.save(ckpt_dir / tag, pure_dict_state)
    
if retrain:
    model_config_json = json.dumps(model_config)
    with open('model_config.json', 'w') as f:
        f.write(model_config_json)
        
    if save_ckpt:
        ckpt_dir = ocp.test_utils.erase_and_create_empty(os.path.abspath(save_ckpt_path))
        checkpointer = ocp.StandardCheckpointer()
        save_ckpt('last', model, display = True)


    x_visual = video_features[0:5]
    x_audio = audio_features[0:5]
    x_text = language_features[0:5].unsqueeze(1).repeat(1, stimulus_window, 1)
    x_ids = subject_ids[0:5]

    logits = model(jnp.array(x_visual), jnp.array(x_audio), jnp.array(x_text), jnp.array(x_ids))
    print("Output shape: ", logits.shape)

else:
    try:
        # Delete the model to free up memory
        del model

        # Load the checkpoint
        checkpointer = ocp.StandardCheckpointer()
        restored_pure_dict = checkpointer.restore(os.path.join(path_last_training, 'checkpoints', 'last')) # /kaggle/input/t-att-2/checkpoints/best.orbax-checkpoint-tmp-40/_CHECKPOINT_METADATA
        abstract_model = nnx.eval_shape(lambda: MultiModalTransformer(**model_config, rngs=nnx.Rngs(0)))
        graphdef, abstract_state = nnx.split(abstract_model)
        nnx.replace_by_pure_dict(abstract_state, restored_pure_dict)
        model = nnx.merge(graphdef, abstract_state)
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


# In[ ]:


def count_model_parameters(model: nnx.Module):
    params = nnx.state(model, nnx.Param)
    total_params = sum((np.prod(x.shape) for x in jax.tree.leaves(params)), 0)
    print("Parameters on the model: ", total_params)
    return total_params
total_model_parameters = count_model_parameters(model) # Parameters on the model:  415478760 # Parameters on the model:  415491048


# In[ ]:


params = nnx.state(model, nnx.Param)
# sum_parameters


# In[ ]:


if retrain:
    import optax

    num_epochs = 50
    learning_rate = 1e-5
    regularization_strength = 1e-4
    total_steps = len(train_dataloader.dataset) // train_dataloader.batch_size
    lr_schedule_type = 'polynomial'
    optimizer_type = 'adam'

    if lr_schedule_type == 'linear':
        lr_schedule = optax.linear_schedule(
            learning_rate,
            0.0,
            num_epochs * total_steps
        )
    else:
        lr_schedule = optax.polynomial_schedule(init_value=learning_rate, end_value=learning_rate/10, power=2, transition_steps=num_epochs*total_steps)

    if optimizer_type == 'adam':
        optimizer = nnx.Optimizer(model, optax.adam(lr_schedule, regularization_strength))
    elif optimizer_type == 'sgd':
        optimizer = nnx.Optimizer(model, optax.sgd(lr_schedule, regularization_strength))
    elif optimizer_type == 'adan':
        optimizer = nnx.Optimizer(model, optax.adan(learning_rate = learning_rate, weight_decay = regularization_strength))
    elif optimizer_type == 'lion':
        optimizer = nnx.Optimizer(model, optax.lion(learning_rate = lr_schedule, weight_decay = regularization_strength))

    def compute_loss_and_logits(model: nnx.Module, x_visual: jax.Array, x_audio: jax.Array, x_text: jax.Array, y: jax.Array, sub_ids: jax.Array):
        logits = model(x_visual, x_audio, x_text, sub_ids)
        mse_loss = jnp.mean(jnp.square(logits - y))
        
        # Retrieve model parameters using nnx.state with nnx.Param
        params = nnx.state(model, nnx.Param)
        # Compute L2 regularization over all parameter leaves
        l2_reg = sum(jnp.sum(jnp.square(p)) for p in jax.tree.leaves(params))
        
        # Combine MSE loss and L2 regularization term
        total_loss = mse_loss + regularization_strength * l2_reg
        return total_loss, logits

    @nnx.jit
    def train_step(model: nnx.Module, optimizer: nnx.Optimizer, batch: tuple[jax.Array, jax.Array, jax.Array, jax.Array], train_metrics: nnx.MultiMetric):
        x_visual, x_audio, x_text, fmri, sub_ids = batch
        y = fmri

        grad_fn = nnx.value_and_grad(compute_loss_and_logits, has_aux=True)
        (loss, logits), grads = grad_fn(model, x_visual, x_audio, x_text, y, sub_ids)

        optimizer.update(grads)

        train_metrics.update(
            loss=loss,
            person_correlation=compute_person_correlation(logits, y)
        )

        return loss, logits

    # def compute_person_correlation(logits: jax.Array, y: jax.Array):
    #     return jnp.corrcoef(logits, y)[0, 1]

    def compute_person_correlation(logits: jax.Array, y: jax.Array):
        # Compute correlation for each feature separately
        # logits and y shape: [batch_size, num_features]
        
        # Center the data
        logits_centered = logits - jnp.mean(logits, axis=0, keepdims=True)
        y_centered = y - jnp.mean(y, axis=0, keepdims=True)
        
        # Compute correlation for each feature
        numerator = jnp.sum(logits_centered * y_centered, axis=0)
        denominator = jnp.sqrt(jnp.sum(logits_centered**2, axis=0) * jnp.sum(y_centered**2, axis=0))
        
        # Avoid division by zero
        correlation = jnp.where(denominator != 0, numerator / denominator, 0.0)
        
        # Return mean correlation across all features
        return jnp.mean(correlation)

    @nnx.jit 
    def eval_step(model: nnx.Module, batch: tuple[jax.Array, jax.Array, jax.Array, jax.Array], eval_metrics: nnx.MultiMetric, subject_id: int):
        x_visual, x_audio, x_text, fmri, sub_ids = batch
        y = fmri
        loss, logits = compute_loss_and_logits(model, x_visual, x_audio, x_text, y, sub_ids)
        person_correlation = compute_person_correlation(logits, y)
        eval_metrics.update(
            loss = loss,
            person_correlation = person_correlation
        )
        
    def convert_batch_to_jax(batch):
        x_visual = jnp.array(batch[0], dtype=jnp.float32)
        x_audio = jnp.array(batch[1], dtype=jnp.float32)
        x_text = jnp.array(batch[2].unsqueeze(1).repeat(1, stimulus_window, 1), dtype=jnp.float32)
        y = jnp.array(batch[3], dtype=jnp.float32)
        sub_id = jnp.array(batch[4], dtype=jnp.float32)
        return x_visual, x_audio, x_text, y, sub_id
        
    def convert_tensor_to_jax(tensor):
        return jnp.array(tensor, dtype=jnp.float32)


# In[ ]:


if retrain:
    individual_subject_loss_metrics = {
        f"loss_subject_{subject}": nnx.metrics.Average(f"loss_subject_{subject}") for subject in subjects
    }
    individual_subject_correlation_metrics = {
        f"person_correlation_{subject}": nnx.metrics.Average(f"person_correlation_{subject}") for subject in subjects
    }

    eval_metrics = nnx.MultiMetric(
        **individual_subject_loss_metrics,
        **individual_subject_correlation_metrics
    )

    all_eval_metrics = {sub: nnx.MultiMetric(loss = nnx.metrics.Average(f"loss"), 
                                                person_correlation = nnx.metrics.Average(f"person_correlation")) for sub in subjects}

    train_metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average('loss'),
        person_correlation=nnx.metrics.Average('person_correlation')
    )

    train_metrics_history = {
        "train_loss": [],
        "train_person_correlation": [],
    }

    eval_metrics_history = {
        "eval_loss": [],
        "eval_person_correlation": [],
    }


# In[ ]:


if retrain:

    bar_format = "{desc}[{n_fmt}/{total_fmt}] {postfix} [{elapsed}<{remaining}, {rate_fmt}]"




    def train_one_epoch(epoch):
        model.train()
        with tqdm(
            desc=f"[train] epoch {epoch}/{num_epochs}",
            total=total_steps,
            bar_format=bar_format
        ) as pbar:
            for batch in train_dataloader:
                loss, logits = train_step(model, optimizer, convert_batch_to_jax(batch), train_metrics)
                pbar.set_postfix(**{
                    "loss": loss,
                })
                pbar.update(1)
                

    def evaluate_model(epoch, subject_id, eval_metrics):
        model.eval()

        eval_metrics.reset()
        for val_batch in val_dataloaders[subject_id]:
            eval_step(model, convert_batch_to_jax(val_batch), eval_metrics, subject_id)

        for metric, value in eval_metrics.compute().items():
            eval_metrics_history[f"eval_{metric}"].append(value)

        print(f"Epoch {epoch} for subject {subject_id} - {eval_metrics.compute()}")



# In[ ]:


if retrain:
    from scipy.stats import pearsonr
    from nilearn import plotting
    import os
    from nilearn.maskers import NiftiLabelsMasker
    from io import BytesIO
    import io
    from PIL import Image

    def compute_correlation_plot(model, dataloader, run_name = 'vit_jax', dataset_split = 1):
        "Running this function for training set can take more than 2 minutes, so changing the random_percentage you can just sample a small part of the dataset"
        
        fmri_val = []
        fmri_val_pred = []
        for i, batch_val in enumerate(dataloader):
            x_visual, x_audio, x_text, fmri, sub_id = convert_batch_to_jax(batch_val)
            logits = model(x_visual, x_audio, x_text, sub_id)
            fmri_val_pred.append(logits)
            fmri_val.append(fmri)
            
            if i > ((len(dataloader.dataset) / train_dataloader.batch_size) * dataset_split):
                break
                
        fmri_val = np.concatenate(fmri_val, axis=0)
        fmri_val_pred = np.concatenate(fmri_val_pred, axis=0)

        output_dir = '.'
        alpha = regularization_strength

        encoding_accuracy = np.zeros(fmri_val.shape[1], dtype=np.float32)
        for p in range(fmri_val.shape[1]):
            encoding_accuracy[p] = pearsonr(fmri_val[:, p], fmri_val_pred[:, p])[0]
        mean_accuracy = np.round(np.mean(encoding_accuracy), 3)
        print(f"Stimulus Window: {stimulus_window} | Alpha: {alpha} -> Mean Accuracy: {mean_accuracy}")

        atlas_file = f'sub-0{subject}_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-dseg_parcellation.nii'
        atlas_path = os.path.join(root_data_dir, 'algonauts_2025.competitors', 'fmri',
                                f'sub-0{subject}', 'atlas', atlas_file)
        atlas_masker = NiftiLabelsMasker(labels_img=atlas_path)
        atlas_masker.fit()
        encoding_accuracy_nii = atlas_masker.inverse_transform(encoding_accuracy)

        title = (f"Encoding accuracy, sub-0{subject}, modality-{modality}, "
                f"sw-{stimulus_window}, alpha-{alpha:.2g}, mean accuracy: {mean_accuracy}")

        display = None # Initialize display to None
        fig = None     # Initialize fig to None
        pil_image = None # Initialize result to None

        try:
            display = plotting.plot_glass_brain(
                encoding_accuracy_nii,
                display_mode="lyrz",
                cmap='hot_r',
                colorbar=True,
                plot_abs=False,
                symmetric_cbar=False,
                title=title,
                # threshold='auto' # Consider adding a threshold
            )

            # --- Convert display object to PIL Image ---
            # 1. Create an in-memory buffer
            buf = io.BytesIO()

            # 2. Save the figure to the buffer
            # Access the underlying matplotlib figure via frame_axes
            fig = display.frame_axes.figure
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight') # Adjust dpi as needed

            # 3. Rewind the buffer to the beginning
            buf.seek(0)

            # 4. Load the image from the buffer using PIL
            pil_image = Image.open(buf)

            # 5. Copy the image data and close the buffer to release memory
            # (PIL can sometimes keep the buffer open lazily)
            pil_image = pil_image.copy()
            buf.close()

        except AttributeError as e:
            print(f"Error accessing figure from nilearn display object: {e}")
            print("Nilearn's internal structure might have changed or plotting failed.")
        except Exception as e:
            print(f"An error occurred during plotting or PIL conversion: {e}")
        finally:
            # 6. Close the matplotlib figure explicitly to free memory
            if display is not None:
                try:
                    # Use the display object's close method if available
                    display.close()
                except AttributeError:
                    # Fallback to closing the figure directly if display.close() doesn't exist
                    if fig is not None:
                        plt.close(fig)
            elif fig is not None:
                # If display failed but fig was somehow created
                plt.close(fig)
        return pil_image


# In[ ]:


if retrain:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    secret_value_0 = user_secrets.get_secret("wandb_key")

    # set up wandb
    import wandb
    wandb.login(key=secret_value_0)


    wandb_project = 'algonauts2025'
    run_name = 'vit_jax_internvl31b_whisper_modernbert_sw6_full_features_multi_submission'

    print(f"Wandb project: {wandb_project}, run name: {run_name}")

    wandb_config = {
        'run_name': run_name,
        'stimulus_window': stimulus_window,
        'regularization_strength': regularization_strength,
        'model_config': model_config,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'total_steps': total_steps,
        'lr_schedule': lr_schedule_type,
        'optimizer': optimizer_type,
        "total_model_parameters": total_model_parameters,
        "subjects": subjects
    }

    print(f"Wandb config: {wandb_config}")
    print(f"Wandb id: {wandb.util.generate_id()}")

    wandb.init(
        id=wandb.util.generate_id(),
        project=wandb_project,
        name=run_name,
        config=wandb_config,
        resume='allow',
    )




# In[ ]:


def log_wandb(
    epoch: int,
    train_metrics: nnx.MultiMetric,
    train_correlation_plot: wandb.Plotly,
):
    # build a single dict of everything
    metrics = {
        "epoch": epoch,
        "train_loss": train_metrics.loss.compute(),                  
        "train_correlation": train_metrics.person_correlation.compute(),                    
        "train_correlation_plot": wandb.Image(train_correlation_plot),       
    }
    
    wandb.log(metrics, step=epoch)


# In[ ]:


if retrain:
    import time


    best_epoch = 8 # num_epoch # loaded from best_config.json
    for epoch in range(best_epoch + 1):
        print(f"\n--- Epoch {epoch} ---")


        start_time = time.time()
        train_one_epoch(epoch)
        duration = time.time() - start_time
        print(f"Training took {duration:.2f} seconds")

  

        start_time = time.time()
        plot_correlation_train = compute_correlation_plot(model, train_dataloader, dataset_split = 0.1)
        duration = time.time() - start_time
        print(f"Train correlation plot took {duration:.2f} seconds")


        save_ckpt('last', model, display = False)

        
        start_time = time.time()
        log_wandb(
            epoch,
            train_metrics,
            plot_correlation_train
        )
        duration = time.time() - start_time
        print(f"W&B logging took {duration:.2f} seconds")


# ## Prepare inference data

# In[ ]:


def load_stimulus_features_s7(root_data_dirs: Union[str, List[str]], modality: str, layers: List[str] = None) -> dict:
    features = {modality: {}}

    # Ensure root_data_dirs is a list
    if isinstance(root_data_dirs, str):
        root_data_dirs = [root_data_dirs]

    # Iterate over each root directory provided.
    for root_data_dir in root_data_dirs:
        root_path = Path(root_data_dir)
        # Traverse all .h5 files under the current root directory.
        all_files = list(root_path.rglob("*.h5"))
        s7_files = [f for f in all_files if "friends_s07" in f.stem]

        print(f"Found {len(s7_files)} S7 files in {root_data_dir}")

        for h5_file in tqdm(s7_files, desc=f"Processing {root_data_dir}"):
            movie_name = h5_file.stem
            # Remove the "friends_" prefix if it exists.
            if movie_name.startswith("friends_"):
                movie_name = movie_name[len("friends_"):]
                
            datasets = []
            with h5py.File(h5_file, 'r') as f:
                # Iterate over all datasets (layers) in the file.
                for layer in f.keys():
                    data = f[layer][:]
                    if layers:
                        if layer in layers:
                            datasets.append(data)
                    else:
                        datasets.append(data)
            
            # If multiple layers exist, concatenate along axis=1.
            if len(datasets) > 1:
                datasets = [np.reshape(item, (item.shape[0], -1)) for item in datasets]
                # print(datasets[0].shape, datasets[1].shape, datasets[2].shape)
                concatenated_features = np.concatenate(datasets, axis=1)
            elif datasets:
                concatenated_features = datasets[0]
            else:
                continue
            
            # If the same movie_name is encountered from multiple directories,
            # you might choose to either overwrite or merge; here we overwrite.
            features[modality][movie_name] = concatenated_features.reshape(concatenated_features.shape[0], -1)

    return features

def align_features_and_fmri_samples_friends_s7(features_friends_s7,
    root_data_dir):
    """
    Align the stimulus feature with the fMRI response samples for Friends season
    7 episodes, later used to predict the fMRI responses for challenge
    submission.

    Parameters
    ----------
    features_friends_s7 : dict
        Dictionary containing the stimulus features for Friends season 7.
    root_data_dir : str
        Root data directory.

    Returns
    -------
    aligned_features_friends_s7 : dict
        Aligned stimulus features for each subject and Friends season 7 episode.

    """

    ### Empty results dictionary ###
    aligned_features_friends_s7 = {}

    ### HRF delay ###
    # fMRI detects the BOLD (Blood Oxygen Level Dependent) response, a signal
    # that reflects changes in blood oxygenation levels in response to activity
    # in the brain. Blood flow increases to a given brain region in response to
    # its activity. This vascular response, which follows the hemodynamic
    # response function (HRF), takes time. Typically, the HRF peaks around 5–6
    # seconds after a neural event: this delay reflects the time needed for
    # blood oxygenation changes to propagate and for the fMRI signal to capture
    # them. Therefore, this parameter introduces a delay between stimulus chunks
    # and fMRI samples for a better correspondence between input stimuli and the
    # brain response. For example, with a hrf_delay of 3, if the stimulus chunk
    # of interest is 17, the corresponding fMRI sample will be 20.

    ### Stimulus window ###
    # stimulus_window indicates how many stimulus feature samples are used to
    # model each fMRI sample, starting from the stimulus sample corresponding to
    # the fMRI sample of interest, minus the hrf_delay, and going back in time.
    # For example, with a stimulus_window of 5, and a hrf_delay of 3, if the
    # fMRI sample of interest is 20, it will be modeled with stimulus samples
    # [13, 14, 15, 16, 17]. Note that this only applies to visual and audio
    # features, since the language features were already extracted using
    # transcript words spanning several movie samples (thus, each fMRI sample
    # will only be modeled using the corresponding language feature sample,
    # minus the hrf_delay). Also note that a larger stimulus window will
    # increase compute time, since it increases the amount of stimulus features
    # used to train and validate the fMRI encoding models. Here you will use a
    # value of 5, since this is how the challenge baseline encoding models were
    # trained.

    ### Loop over subjects ###
    subjects = [1, 2, 3, 5]
    desc = "Aligning stimulus and fMRI features of the four subjects"
    for sub in tqdm(subjects, desc=desc):
        aligned_features_friends_s7[f'sub-0{sub}'] = {}

        ### Load the Friends season 7 fMRI samples ###
        samples_dir = os.path.join(root_data_dir, 'algonauts_2025.competitors',
            'fmri', f'sub-0{sub}', 'target_sample_number',
            f'sub-0{sub}_friends-s7_fmri_samples.npy')
        fmri_samples = np.load(samples_dir, allow_pickle=True).item()

        ### Loop over Friends season 7 episodes ###
        for epi, samples in fmri_samples.items():
            features_epi = []

            ### Loop over fMRI samples ###
            for s in range(samples):
                # Empty variable containing the stimulus features of all
                # modalities for each sample
                f_all = np.empty(0)

                ### Loop across modalities ###
                for mod in features_friends_s7.keys():

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
                            idx_start = idx_end - stimulus_window
                        f = features_friends_s7[mod][epi][idx_start:idx_end]
                        f_all = np.append(f_all, f.flatten())

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
                            f = features_friends_s7[mod][epi][-1,:]
                        else:
                            f = features_friends_s7[mod][epi][idx]
                        f_all = np.append(f_all, f.flatten())

                ### Append the stimulus features of all modalities for this sample ###
                features_epi.append(f_all)

            ### Add the episode stimulus features to the features dictionary ###
            aligned_features_friends_s7[f'sub-0{sub}'][epi] = np.asarray(
                features_epi, dtype=np.float32)

    return aligned_features_friends_s7


# In[ ]:


features_friends_s7 = {}

features_friends_s7['visual'] = load_stimulus_features_s7(
    [os.path.join(base_features_dir, video_features_name)],
    modality='visual',
    layers=layers_visual)['visual']

features_friends_s7['audio'] = load_stimulus_features_s7(
    [os.path.join(base_features_dir, audio_features_name)],
    modality='audio',
    layers=layers_audio)['audio']

features_friends_s7['language'] = load_stimulus_features_s7(
    [os.path.join(base_features_dir, transcript_features_name)],
    modality='language',
    layers=layers_transcript)['language']


aligned_features_friends_s7 = align_features_and_fmri_samples_friends_s7(features_friends_s7, root_data_dir)


# In[ ]:


# Empty submission predictions dictionary
submission_predictions = {}

# Loop through each subject 
desc = "Predicting fMRI responses of each subject"
for sub, features in tqdm(aligned_features_friends_s7.items(), desc=desc):

    visual_features_shape_length = features_friends_s7['visual']['s07e01a'].shape[-1]
    audio_features_shape_length = features_friends_s7['audio']['s07e01a'].shape[-1]
    language_features_shape_length = features_friends_s7['language']['s07e01a'].shape[-1]

    visual_features_shape_indexes = (0, visual_features_shape_length * stimulus_window)
    audio_features_shape_indexes = (visual_features_shape_indexes[1], visual_features_shape_indexes[1] + audio_features_shape_length * stimulus_window)
    language_features_shape_indexes = (audio_features_shape_indexes[1], audio_features_shape_indexes[1] + language_features_shape_length) # No window for language

    # Initialize the nested dictionary for each subject's predictions
    submission_predictions[sub] = {}

    # Loop through each Friends season 7 episode
    for epi, feat_epi in tqdm(features.items()):

        
        # Do the inference all at once
        subject_ids = np.array([int(sub.split('-')[1]) for _ in range(feat_epi.shape[0])])
        subject_ids = jnp.array(subject_ids, dtype=jnp.int32)

        x_visual = jnp.array(feat_epi[:, visual_features_shape_indexes[0]:visual_features_shape_indexes[1]], dtype=jnp.float32)
        x_audio = jnp.array(feat_epi[:, audio_features_shape_indexes[0]:audio_features_shape_indexes[1]], dtype=jnp.float32)
        x_text = torch.Tensor(feat_epi[:, language_features_shape_indexes[0]:language_features_shape_indexes[1]])

        x_visual = einops.rearrange(x_visual, 'b (w a) -> b w a', w = stimulus_window)
        x_audio = einops.rearrange(x_audio, 'b (w a) -> b w a', w = stimulus_window)
        x_text = jnp.array(x_text.unsqueeze(1).repeat(1, stimulus_window, 1), dtype=jnp.float32)
        
        # print(x_visual.shape)
        # Make predictions
        fmri_pred = model(x_visual, x_audio, x_text, subject_ids)
        fmri_pred = np.array(fmri_pred, dtype=np.float32)
        

        # Store formatted predictions in the nested dictionary
        submission_predictions[sub][epi] = fmri_pred


# In[ ]:


# visual_features_shape_indexes, audio_features_shape_indexes, language_features_shape_indexes


# In[ ]:


# features_friends_s7['visual']['s07e01a'].shape[-1] / 6


# In[ ]:


# aligned_features_friends_s7['sub-01']['s07e01a'].shape[-1] / 6


# In[ ]:


# Display the structure and shapes of the predicted fMRI responses dictionary
for subject, episodes_dict in submission_predictions.items():
    # Print the subject and episode number for Friends season 7
    print(f"Subject: {subject}")
    print(f"  Number of Episodes: {len(episodes_dict)}")
    # Print the predicted fMRI response shape for each episode
    for episode, predictions in episodes_dict.items():
        print(f"    - Episode: {episode}, Predicted fMRI shape: {predictions.shape}")
    print("-" * 40)  # Separator for clarity


# In[ ]:


import zipfile

# Select the saving directory
save_dir = './' #@param {type:"string"}

# Save the predicted fMRI dictionary as a .npy file
output_file = save_dir + "fmri_predictions_friends_s7.npy"
np.save(output_file, submission_predictions)
print(f"Formatted predictions saved to: {output_file}")

# Zip the saved file for submission
zip_file = save_dir + "fmri_predictions_friends_s7.zip"
with zipfile.ZipFile(zip_file, 'w') as zipf:
    zipf.write(output_file, os.path.basename(output_file))
print(f"Submission file successfully zipped as: {zip_file}")

