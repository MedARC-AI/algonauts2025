# import glob
# import os
# import shutil
# from tqdm import tqdm

# source_dir = "/home/pranav/mihir/algonauts_challenge/AlgonautsDS-features/features/whisper"
# target_dir = "/home/pranav/mihir/algonauts_challenge/whisper"

# for dirpath, dirnames, filenames in os.walk(source_dir):
#     relative_path = os.path.relpath(dirpath, source_dir)
#     parts = relative_path.split(os.sep)
#     if parts[0] in ["friends", "movie10"]:
#         h5_files = [f for f in filenames if f.endswith(".h5")]
#         for filename in tqdm(h5_files, desc=f"Copying files in {dirpath}"):
#             original_file = os.path.join(dirpath, filename)
#             base, ext = os.path.splitext(filename)
#             new_filename = base + "_features_audio" + ext
#             os.makedirs(target_dir, exist_ok=True)
#             target_file = os.path.join(target_dir, new_filename)
#             shutil.copy(original_file, target_file)


import numpy as np
from utils import load_fmri



def normalize(data):
    assert data.shape[1] == 1000, "Data does not have 1000 parcels"
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)

    # Avoid division by zero
    stds = np.where(stds == 0, 1, stds)

    # Standardize
    normalized_data = (data - means) / stds

    # Verify
    new_means = np.mean(normalized_data, axis=0)
    new_stds = np.std(normalized_data, axis=0)
    max_abs_mean = np.max(np.abs(new_means))
    max_std_diff = np.max(np.abs(new_stds - 1))
    # Check for problematic parcels
    if max_std_diff > 1e-5 or max_abs_mean > 1e-5:
        print("  Warning: Normalization not perfect. Check data.")
    else:
        return normalized_data






fmri_dir = "/home/pranav/mihir/algonauts_challenge/algonauts_2025.competitors/fmri/"
subject = 1

fmri_data = load_fmri(fmri_dir, subject)
data_keys = fmri_data.keys()
x = []
for i in data_keys:
    fmri_data[i] = normalize(fmri_data[i])
    x.append(fmri_data[i])

data = np.concatenate(x)


# Ensure the data has 1000 parcels (columns)
assert data.shape[1] == 1000, "Data does not have 1000 parcels"

# Compute mean and standard deviation for each parcel
means = np.mean(data, axis=0)
stds = np.std(data, axis=0)

# Set tolerance for floating-point comparisons
epsilon = 1e-5

# Check if all means are close to 0 and all stds are close to 1
if np.all(np.abs(means) < epsilon) and np.all(np.abs(stds - 1) < epsilon):
    print("Data is normalized.")
else:
    print("Data is not normalized.")
    print(f"Max absolute mean: {np.max(np.abs(means))}")
    print(f"Max std difference: {np.max(np.abs(stds - 1))}")
    print(f"Number of parcels with |mean| >= {epsilon}: {np.sum(np.abs(means) >= epsilon)}")
    print(f"Number of parcels with |std - 1| >= {epsilon}: {np.sum(np.abs(stds - 1) >= epsilon)}")