import time
from collections import defaultdict
import os
import h5py
import numpy as np
from torch.utils.data import Dataset
# from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts # Not used in provided snippet
# import torch # Not directly used in data loading part for profiling
# import torch.nn as nn # Not used
# import torch.nn.functional as F # Not used
# import lightning as L # Not used
# from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping # Not used
# from lightning.pytorch.loggers import WandbLogger # Not used
# from lightning.pytorch.utilities.model_summary import ModelSummary # Not used
# from lightning.pytorch.utilities.combined_loader import CombinedLoader # Not used
# import wandb # Not used
from torch.utils.data import DataLoader
# from tqdm import tqdm # Not used in provided snippet
from pathlib import Path
# import zipfile # Not used
import sys

# Assuming 'utils.py' is in the same directory or Python path
# If not, adjust sys.path or ensure it's discoverable
try:
    from utils import load_fmri, align_features_and_fmri_samples, align_features_and_fmri_samples_friends_s7, CosineLRSchedulerWithWarmup, calculate_metrics, normalize, normalize_across_episodes, check_fmri_centering, CosineAnnealingWarmDecayedRestarts
except ImportError:
    print("Warning: 'utils.py' not found or some functions are missing. load_fmri and align_features_and_fmri_samples are crucial.")
    # Define dummy functions if utils is not available, so the script doesn't crash immediately
    # This is just for the script to run; actual functionality will be missing.
    def load_fmri(fmri_dir, subject):
        print(f"DUMMY: load_fmri called for {fmri_dir}, sub {subject}")
        # Return a dictionary with expected keys but empty/dummy data
        # This structure depends on what align_features_and_fmri_samples expects
        return {'movie_key1': np.array([]), 'movie_key2': np.array([])}

    def align_features_and_fmri_samples(stimuli_features, fmri_data, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies):
        print("DUMMY: align_features_and_fmri_samples called")
        # Return dummy aligned data structures
        # This structure depends on how it's used later
        dummy_shape_features = (10, stimulus_window, 100) # Example shape: (num_samples, window_size, feature_dim)
        dummy_shape_fmri = (10, 50) # Example shape: (num_samples, fmri_dim)
        return {
            'audio': np.zeros(dummy_shape_features),
            'visual': np.zeros(dummy_shape_features),
            'language': np.zeros(dummy_shape_features)
        }, np.zeros(dummy_shape_fmri)


# --- Profiler Code ---
timings = {} # For broad sections
accumulated_operation_times = defaultdict(lambda: {'total_duration': 0.0, 'count': 0}) # For aggregated fine-grained ops

def start_timer(section_name):
    """Starts a timer for a given broad section."""
    timings[section_name] = {'start': time.perf_counter(), 'end': None, 'duration': None}
    # Removed print f"--- Starting: {section_name} ---" for less verbosity

def end_timer(section_name):
    """Ends a timer for a given broad section and calculates duration."""
    if section_name in timings and timings[section_name]['start'] is not None:
        timings[section_name]['end'] = time.perf_counter()
        timings[section_name]['duration'] = timings[section_name]['end'] - timings[section_name]['start']
        # Removed print f"--- Finished: {section_name} (Took: {timings[section_name]['duration']:.4f} seconds) ---"
    else:
        # This error print is useful for debugging the profiler itself
        print(f"Profiler Error: Timer for section '{section_name}' was not started or already ended.")

def record_op_time(category_key, duration):
    """Accumulates duration and count for a specific operation category."""
    accumulated_operation_times[category_key]['total_duration'] += duration
    accumulated_operation_times[category_key]['count'] += 1

def print_summary():
    """Prints a summary of all timed sections and aggregated operations."""
    print("\n--- Broad Section Timings ---")
    if not timings:
        print("No broad sections were timed.")
    else:
        sorted_broad_timings = sorted(timings.items(), key=lambda item: item[1]['duration'] if item[1]['duration'] is not None else float('inf'), reverse=True)
        for section_name, data in sorted_broad_timings:
            if data['duration'] is not None:
                print(f"Section: {section_name:<65} | Duration: {data['duration']:.4f} seconds")
            else:
                print(f"Section: {section_name:<65} | Duration: Not completed")
    print("-----------------------------")

    print("\n--- Aggregated Operation Timings ---")
    if not accumulated_operation_times:
        print("No aggregated operations were timed.")
    else:
        sorted_accumulated = sorted(
            accumulated_operation_times.items(),
            key=lambda item: item[1]['total_duration'],
            reverse=True
        )
        print(f"{'Operation Category':<70} | {'Total Time (s)':<15} | {'Count':<7} | {'Avg Time/Op (s)':<18}")
        print("-" * 115)
        for name, data in sorted_accumulated:
            avg_duration = data['total_duration'] / data['count'] if data['count'] > 0 else 0
            print(f"{name:<70} | {data['total_duration']:<15.4f} | {data['count']:<7} | {avg_duration:<18.6f}")
    print("----------------------------------\n")
# --- End of Profiler Code ---


class AlgonautsDataset(Dataset):
    def __init__(self, features_dir, fmri_dir, movies, subject, excluded_samples_start=5, excluded_samples_end=5, hrf_delay=3, stimulus_window=5):
        # Note: start_timer/end_timer are global and will be used by the main script part
        # We will use record_op_time for internal aggregations.

        self.features_dir = features_dir
        self.fmri_dir = fmri_dir
        self.movies = movies
        self.subject = subject
        self.excluded_samples_start = excluded_samples_start
        self.excluded_samples_end = excluded_samples_end
        self.hrf_delay = hrf_delay
        self.stimulus_window = stimulus_window
        self.partition_indices = defaultdict(list)
        
        # This specific timer is for the feature loading part within dataset init
        _feature_load_start = time.perf_counter()
        
        stimuli_features = {"visual": {}, "audio": {}, "language": {}}
        
        for movie_idx, movie in enumerate(self.movies):
            movie_type_key = "Friends" if 'friends' in movie else "Movies"
            
            if 'friends' in movie:
                season = movie.split('-')[1]
                
                # Listing Dirs (Audio/Visual for Friends) - Timed as a broader operation
                _list_dir_start = time.perf_counter()
                dir_list = sorted(os.listdir(self.features_dir['audio'] / 'audio'))
                record_op_time(f"os.listdir - Friends '{movie}' - Audio/Visual Dirs", time.perf_counter() - _list_dir_start)
                
                for episode in dir_list:
                    if f"{season}e" in episode and '_features_' in episode:
                        episode_base = episode.split('_features_')[0]
                        for modality in ['audio', 'visual']:
                            file_path = self.features_dir[modality] / modality / f"{episode_base}_features_{modality}.h5"
                            
                            _op_start = time.perf_counter()
                            try:
                                with h5py.File(file_path, 'r') as f:
                                    _open_duration = time.perf_counter() - _op_start
                                    record_op_time(f"H5 File Opening - {movie_type_key} - {modality}", _open_duration)
                                    
                                    _read_start = time.perf_counter()
                                    try:
                                        stimuli_features[modality][episode_base.split('_')[1]] = f['language_model.model.layers.20.post_attention_layernorm'][:]
                                    except KeyError:
                                        stimuli_features[modality][episode_base.split('_')[1]] = f['layers.31.fc2'][:]
                                    _read_duration = time.perf_counter() - _read_start
                                    record_op_time(f"H5 Data Reading - {movie_type_key} - {modality}", _read_duration)
                            except Exception as e:
                                _op_duration_on_fail = time.perf_counter() - _op_start
                                record_op_time(f"H5 File Opening (Failed) - {movie_type_key} - {modality}", _op_duration_on_fail)
                                print(f"Error with H5 {file_path} ({movie_type_key} {modality}): {e}")
                
                # Listing Dirs (Language for Friends)
                _list_dir_start = time.perf_counter()
                lang_dir_list = sorted(os.listdir(self.features_dir['language'] / 'language'))
                record_op_time(f"os.listdir - Friends '{movie}' - Language Dirs", time.perf_counter() - _list_dir_start)

                for episode in lang_dir_list:
                    if f"{season}e" in episode and '_features_' in episode:
                        episode_base = episode.split('_features_')[0]
                        file_path = self.features_dir['language'] / 'language' / f"{episode_base}_features_language.h5"
                        modality = 'language'
                        _op_start = time.perf_counter()
                        try:
                            with h5py.File(file_path, 'r') as f:
                                _open_duration = time.perf_counter() - _op_start
                                record_op_time(f"H5 File Opening - {movie_type_key} - {modality}", _open_duration)
                                _read_start = time.perf_counter()
                                st_season_episode = episode_base.split('_')[1]
                                stimuli_features[modality][st_season_episode] = f[st_season_episode]['language_pooler_output'][:]
                                _read_duration = time.perf_counter() - _read_start
                                record_op_time(f"H5 Data Reading - {movie_type_key} - {modality}", _read_duration)
                        except Exception as e:
                            _op_duration_on_fail = time.perf_counter() - _op_start
                            record_op_time(f"H5 File Opening (Failed) - {movie_type_key} - {modality}", _op_duration_on_fail)
                            print(f"Error with H5 {file_path} ({movie_type_key} {modality}): {e}")
            else: # Not 'friends' (General Movies)
                movie_name = movie.replace('movie10-', '')
                
                # Listing Dirs (Audio/Visual for Movies)
                _list_dir_start = time.perf_counter()
                partitions = sorted([f_name for f_name in os.listdir(self.features_dir['audio'] / 'audio') if movie_name in f_name and '_features_' in f_name])
                record_op_time(f"os.listdir - Movies '{movie_name}' - Audio/Visual Dirs", time.perf_counter() - _list_dir_start)

                for partition in partitions:
                    partition_base = partition.split('_features_')[0]
                    for modality in ['audio', 'visual']:
                        file_path = self.features_dir[modality] / modality / f"{partition_base}_features_{modality}.h5"
                        _op_start = time.perf_counter()
                        try:
                            with h5py.File(file_path, 'r') as f:
                                _open_duration = time.perf_counter() - _op_start
                                record_op_time(f"H5 File Opening - {movie_type_key} - {modality}", _open_duration)
                                _read_start = time.perf_counter()
                                try:
                                    stimuli_features[modality][partition_base] = f['language_model.model.layers.20.post_attention_layernorm'][:]
                                except KeyError:
                                    stimuli_features[modality][partition_base] = f['layers.31.fc2'][:]
                                _read_duration = time.perf_counter() - _read_start
                                record_op_time(f"H5 Data Reading - {movie_type_key} - {modality}", _read_duration)
                        except Exception as e:
                            _op_duration_on_fail = time.perf_counter() - _op_start
                            record_op_time(f"H5 File Opening (Failed) - {movie_type_key} - {modality}", _op_duration_on_fail)
                            print(f"Error with H5 {file_path} ({movie_type_key} {modality}): {e}")
                
                # Listing Dirs (Language for Movies)
                _list_dir_start = time.perf_counter()
                lang_partitions = sorted([f_name for f_name in os.listdir(self.features_dir['language'] / 'language') if movie_name in f_name and '_features_' in f_name])
                record_op_time(f"os.listdir - Movies '{movie_name}' - Language Dirs", time.perf_counter() - _list_dir_start)
                
                for partition in lang_partitions:
                    partition_base = partition.split('_features_')[0]
                    file_path = self.features_dir['language'] / 'language' / f"{partition_base}_features_language.h5"
                    modality = 'language'
                    _op_start = time.perf_counter()
                    try:
                        with h5py.File(file_path, 'r') as f:
                            _open_duration = time.perf_counter() - _op_start
                            record_op_time(f"H5 File Opening - {movie_type_key} - {modality}", _open_duration)
                            _read_start = time.perf_counter()
                            stimuli_features[modality][partition_base] = f[partition_base]['language_pooler_output'][:]
                            _read_duration = time.perf_counter() - _read_start
                            record_op_time(f"H5 Data Reading - {movie_type_key} - {modality}", _read_duration)
                        # Removed sys.exit() for profiling
                    except Exception as e:
                        _op_duration_on_fail = time.perf_counter() - _op_start
                        record_op_time(f"H5 File Opening (Failed) - {movie_type_key} - {modality}", _op_duration_on_fail)
                        print(f"Error with H5 {file_path} ({movie_type_key} {modality}): {e}")
        
        record_op_time(f"Dataset Init - Sub {subject} - Total Feature Loading Logic", time.perf_counter() - _feature_load_start)
        
        _fmri_load_start = time.perf_counter()
        fmri_data = load_fmri(self.fmri_dir, self.subject)
        record_op_time(f"Dataset Init - Sub {subject} - load_fmri Call", time.perf_counter() - _fmri_load_start)
        
        self.raw_stimuli = stimuli_features

        _align_start = time.perf_counter()
        self.aligned_features, self.aligned_fmri = align_features_and_fmri_samples(
            stimuli_features, 
            fmri_data, 
            self.excluded_samples_start, 
            self.excluded_samples_end, 
            self.hrf_delay, 
            self.stimulus_window, 
            self.movies
        )
        record_op_time(f"Dataset Init - Sub {subject} - align_features_and_fmri_samples Call", time.perf_counter() - _align_start)

    def __len__(self):
        # Ensure aligned_features is populated before trying to access its keys
        if not hasattr(self, 'aligned_features') or not self.aligned_features or 'audio' not in self.aligned_features:
            # This might happen if __init__ fails before self.aligned_features is fully set
            print("Warning: self.aligned_features not fully initialized. Returning length 0.")
            return 0
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

# --- Main part of your script ---
if __name__ == "__main__":
    overall_script_start_time = time.perf_counter()
    print("--- Starting: Overall Script Execution ---")
    start_timer("Overall Script Execution") # This will be a broad timer

    # --- User's configuration ---
    # Ensure this path is correct for your environment
    try:
        # Try to determine a sensible root directory if possible, or default
        if 'COLAB_GPU' in os.environ: # Example for Colab
             root_dir = Path('/content/')
        elif os.path.exists('/home/mihirneal/Developer/'): # Specific user path
             root_dir = Path('/home/mihirneal/Developer/')
        # Add another elif for Comp2's path if it's different and fixed
        elif os.path.exists('/home/pranav/stanford_workspace/mihir/algonauts_challenge'): # Path for Comp2
            root_dir = Path('/home/pranav/stanford_workspace/mihir/algonauts_challenge')
        else: # Default or fallback
             root_dir = Path('.') # Current directory as a fallback
             print(f"Warning: Defaulting root_dir to current directory: {root_dir.resolve()}")
        print(f"Using root_dir: {root_dir.resolve()}")
    except Exception as e:
        print(f"Error setting root_dir: {e}")
        root_dir = Path('.') # Fallback

    vision_dir = root_dir / 'algonauts/internvl3_8b_8bit/'
    audio_dir = root_dir / 'algonauts/whisper/'
    lang_dir = root_dir / 'algonauts/AlgonautsDS-features/developer_kit/stimulus_features/raw/'
    features_dir = {
        "visual": vision_dir,
        "audio": audio_dir,
        "language": lang_dir
    }
    fmri_dir = root_dir / 'algonauts/algonauts_2025.competitors/fmri/'
    movies_train = ["friends-s01", "friends-s02", "friends-s03", "friends-s04", "friends-s05", "movie10-bourne", "movie10-figures", "movie10-life", "movie10-wolf"]
    movies_val = ["friends-s06"]
    excluded_samples_start = 5
    excluded_samples_end = 5
    hrf_delay = 0 
    stimulus_window = 15
    subject = 1
    # --- End of User's configuration ---

    print("\n--- Starting: Train Dataset Instantiation ---")
    start_timer("Train Dataset Instantiation")
    train_ds = AlgonautsDataset(features_dir, fmri_dir, movies=movies_train, subject=subject, excluded_samples_start=excluded_samples_start, excluded_samples_end=excluded_samples_end, hrf_delay=hrf_delay, stimulus_window=stimulus_window)
    end_timer("Train Dataset Instantiation")
    if 'Train Dataset Instantiation' in timings and timings['Train Dataset Instantiation']['duration'] is not None:
        print(f"--- Finished: Train Dataset Instantiation (Took: {timings['Train Dataset Instantiation']['duration']:.4f} seconds) ---")


    print("\n--- Starting: Validation Dataset Instantiation ---")
    start_timer("Validation Dataset Instantiation")
    val_ds = AlgonautsDataset(features_dir, fmri_dir, movies=movies_val, subject=subject, excluded_samples_start=excluded_samples_start, excluded_samples_end=excluded_samples_end, hrf_delay=hrf_delay, stimulus_window=stimulus_window)
    end_timer("Validation Dataset Instantiation")
    if 'Validation Dataset Instantiation' in timings and timings['Validation Dataset Instantiation']['duration'] is not None:
        print(f"--- Finished: Validation Dataset Instantiation (Took: {timings['Validation Dataset Instantiation']['duration']:.4f} seconds) ---")


    print("\n--- Starting: Train DataLoader Creation ---")
    start_timer("Train DataLoader Creation")
    train_loader = DataLoader(train_ds,
                              batch_size=32, 
                              num_workers=0, # SET TO 0 FOR HDD ISOLATION TESTING
                              pin_memory=True, 
                             )
    end_timer("Train DataLoader Creation")
    if 'Train DataLoader Creation' in timings and timings['Train DataLoader Creation']['duration'] is not None:
        print(f"--- Finished: Train DataLoader Creation (Took: {timings['Train DataLoader Creation']['duration']:.4f} seconds) ---")


    print(f"\nTrain samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")

    if len(train_ds) > 0: # Proceed only if train_ds is not empty
        print("\n--- Starting: Train DataLoader - First Batch Load ---")
        start_timer("Train DataLoader - First Batch Load")
        for i, batch in enumerate(train_loader):
            vision, audio, lang, fmri = batch['video'], batch['audio'], batch['language'], batch['fmri']
            print(f"\nBatch {i}:")
            print(f"  Vision embeds shape: {vision.shape}")
            print(f"  Audio embeds shape: {audio.shape}")
            print(f"  Language embeds shape: {lang.shape}")
            print(f"  fMRI shape: {fmri.shape}")
            break 
        end_timer("Train DataLoader - First Batch Load")
        if 'Train DataLoader - First Batch Load' in timings and timings['Train DataLoader - First Batch Load']['duration'] is not None:
            print(f"--- Finished: Train DataLoader - First Batch Load (Took: {timings['Train DataLoader - First Batch Load']['duration']:.4f} seconds) ---")
    else:
        print("\nSkipping Train DataLoader - First Batch Load because train_ds is empty.")


    end_timer("Overall Script Execution")
    overall_script_end_time = time.perf_counter()
    total_script_duration = overall_script_end_time - overall_script_start_time
    if 'Overall Script Execution' in timings and timings['Overall Script Execution']['duration'] is not None:
        print(f"--- Finished: Overall Script Execution (Took: {timings['Overall Script Execution']['duration']:.4f} seconds) ---")
    print(f"Total actual script execution time (measured independently): {total_script_duration:.4f} seconds")

    print_summary()
