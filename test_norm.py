import numpy as np
import matplotlib.pyplot as plt
from utils import load_fmri, normalize
import os

# Load your fMRI data

fmri_dir = "/home/pranav/mihir/algonauts_challenge/algonauts_2025.competitors/fmri/"
subject = 1

fmri_data = load_fmri(fmri_dir, subject)
data_keys = fmri_data.keys()

# Pick a few parcels to analyze
parcels_to_check = [0, 100, 200, 500, 999]

for key in list(data_keys)[:2]:  # Check first two episodes
    raw_data = fmri_data[key]  # Shape: (num_clips, 1000)
    norm_data = normalize(raw_data)  # Per-episode normalization
    # For global normalization, use the previously normalized fmri_data
    global_norm_data = fmri_data[key]  # Assuming fmri_data is already globally normalized

    for parcel in parcels_to_check:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.hist(raw_data[:, parcel], bins=50, alpha=0.5, label='Raw')
        plt.title(f'Episode {key}, Parcel {parcel}: Raw')
        plt.subplot(1, 3, 2)
        plt.hist(norm_data[:, parcel], bins=50, alpha=0.5, label='Per-Episode Norm')
        plt.title(f'Per-Episode Norm')
        plt.subplot(1, 3, 3)
        plt.hist(global_norm_data[:, parcel], bins=50, alpha=0.5, label='Global Norm')
        plt.title(f'Global Norm')
        plt.legend()
        # os.makedirs('./plots', exist_ok=True)
        plt.savefig(f'./plots/episode_{key}_parcel_{parcel}.png')
        plt.close()

    # Compute statistics
    raw_vars = np.var(raw_data, axis=0)
    norm_vars = np.var(norm_data, axis=0)
    global_norm_vars = np.var(global_norm_data, axis=0)
    print(f"Episode {key} - Raw variances (min, max): {np.min(raw_vars)}, {np.max(raw_vars)}")
    print(f"Episode {key} - Per-Episode Norm variances (min, max): {np.min(norm_vars)}, {np.max(norm_vars)}")
    print(f"Episode {key} - Global Norm variances (min, max): {np.min(global_norm_vars)}, {np.max(global_norm_vars)}")
