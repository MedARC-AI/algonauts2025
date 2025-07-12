from pathlib import Path
import json
import os

def find_scores_json_parents(root_dir="."):
    root_path = Path(root_dir)
    # Find all 'scores.json' files recursively
    matches = list(root_path.glob("**/scores.json"))
    # Extract parent folders
    parent_folders = [str(file.parent) for file in matches]
    return parent_folders

all_submission_dirs = find_scores_json_parents(".")
for ii, dir in enumerate(all_submission_dirs):
    print(f"Submission{ii}:{dir}")
    json_path = os.path.join(dir, "scores.json")
    with open(json_path,'r', encoding='utf-8') as file:
        data = json.load(file)  # Returns a Python dict/list
        if ii == 0:
            acc = {}
            index = {}
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
