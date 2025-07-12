from pathlib import Path
import json
import os
import numpy as np
import zipfile

def find_scores_json_parents(root_dir="."):
    root_path = Path(root_dir)
    # Find all 'scores.json' files recursively
    matches = list(root_path.glob("**/scores.json"))
    # Extract parent folders
    parent_folders = [str(file.parent) for file in matches]
    return parent_folders

all_submission_dirs = find_scores_json_parents(".")
npys = []
acc = {}
index = {}
for ii, dir in enumerate(all_submission_dirs):
    print(f"Submission{ii}:{dir}")
    json_path = os.path.join(dir, "scores.json")
    npy_files = [f for f in os.listdir(dir) if f.endswith('.npy')]
    npys.append(np.load(os.path.join(dir, npy_files[0]),allow_pickle=True).item())

    with open(json_path,'r', encoding='utf-8') as file:
        data = json.load(file)  # Returns a Python dict/list
        for key,values in data.items():
            if '_' in key:
                # Split into ['sub-01', 'movie-chaplin']
                subject, movie_with_prefix = key.split('_')
                
                # Remove 'movie-' prefix to get 'chaplin'
                movie = movie_with_prefix.replace('movie-', '')
                
                # Initialize subject if not exists
                if subject not in acc:
                    acc[subject] = {}
                    index[subject] = {}
                # Initialize movie if not exists
                if movie not in acc[subject]:
                    acc[subject][movie] = 0

                # keep the best accuracy
                if values>acc[subject][movie]:
                    acc[subject][movie] = values
                    index[subject][movie] = ii
grand_mean = []
for subject,subject_acc in acc.items():
    movie_mean = np.mean(list(subject_acc.values()))
    print(f"Subject: {subject}, Expected OOD Mean Accuracy: {movie_mean:.4f}")
    grand_mean.append(movie_mean)

grand_mean = np.mean(grand_mean)
print(f"Across-subject Expected OOD Mean Accuracy: {grand_mean:.4f}")
print(acc)
print(index)

# Now we get the actual submission file
final_submission = npys[0].copy()
print(f"Stiching files from submissions...")
for subject, episodes_dict in final_submission.items():
    for episode, values in episodes_dict.items():
        matching_key = next(
            (key for key in index[subject].keys() if episode.startswith(key)),
            None
        )
        if matching_key is not None:
            curr_index = index[subject][matching_key]
        else:
            ValueError(f"No matching key found for episode '{episode}'.")
        
        final_submission[subject][episode] = npys[curr_index][subject][episode]

print(f"Saving final submission file...")
os.makedirs(f"./output/bestof{len(all_submission_dirs)}submissions", exist_ok=True)
save_path = os.path.join(f"./output/bestof{len(all_submission_dirs)}submissions", "fmri_predictions_ood_combined.npy")
np.save(save_path, final_submission, allow_pickle=True)
print(f'Saving final submission to {save_path}')
zip_file = save_path.replace("npy","zip")
with zipfile.ZipFile(zip_file, 'w') as zipf:
    zipf.write(save_path, os.path.basename(save_path))
print(f"Submission file successfully zipped as: {zip_file}")
print(final_submission['sub-01']['chaplin1'][:5])