#!/usr/bin/env python3
"""
Ensemble averaging of fMRI predictions from multiple models.
Usage: 
    python ensemble_predictions.py --target_filename "fmri_predictions_friends_s7.npy" --foldernames "folder1,folder2"
    OR (to auto-detect folders containing the target file):
    python ensemble_predictions.py --target_filename "fmri_predictions_friends_s7.npy"
"""
import numpy as np
import os
import zipfile
from datetime import datetime
import argparse
import pytz

def main(target_filename, foldernames=None):
    # Timestamp for output folder
    ny_time = datetime.now(pytz.timezone("America/New_York"))
    now = ny_time.strftime("%y%m%d%H%M") 
    ensemble_output_path = f'./output/ensemble_output_{now}'
    
    # Find all folders containing the target file (if foldernames not provided)
    matching_folders = []
    if not foldernames:
        for root, _, files in os.walk("./output/"):
            if target_filename in files:
                matching_folders.append(os.path.basename(root))
        foldernames = matching_folders
    
    if not foldernames:
        raise FileNotFoundError(f"No folders found containing {target_filename} under ./output/")
    
    print(f"Using {len(foldernames)} models for ensemble:")
    print("\n".join(f"  - {folder}" for folder in foldernames))

    # Create output directory
    os.makedirs(ensemble_output_path, exist_ok=True)
    
    # Load and average predictions
    all_submissions = None
    for foldername in foldernames:
        filepath = os.path.join("./output", foldername, target_filename)
        submission_predictions = np.load(filepath, allow_pickle=True).item()
        
        if all_submissions is None:
            all_submissions = submission_predictions
        else:
            for subject, episodes_dict in submission_predictions.items():
                for episode, values in episodes_dict.items():
                    all_submissions[subject][episode] += values
    
    # Normalize by number of models
    for subject, episodes_dict in all_submissions.items():
        for episode, values in episodes_dict.items():
            all_submissions[subject][episode] /= len(foldernames)
    
    # Save results
    output_file = os.path.join(ensemble_output_path, target_filename)
    np.save(output_file, all_submissions, allow_pickle=True)
    
    # Zip for submission
    zip_file = output_file.replace(".npy", ".zip")
    with zipfile.ZipFile(zip_file, 'w') as zipf:
        zipf.write(output_file, os.path.basename(output_file))
    
    print(f"\nEnsemble saved to: {output_file}")
    print(f"Zipped submission: {zip_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Average fMRI predictions from multiple models")
    parser.add_argument(
        "--target_filename",
        type=str,
        default="fmri_predictions_friends_s7.npy",
        help="Filename of predictions to ensemble (default: fmri_predictions_friends_s7.npy)"
    )
    parser.add_argument(
        "--foldernames",
        type=str,
        default=None,
        help="Comma-separated folder names (optional). If not provided, auto-detects folders containing target_filename."
    )
    args = parser.parse_args()
    
    # Convert comma-separated foldernames to list (if provided)
    foldernames = args.foldernames.split(",") if args.foldernames else None
    
    main(args.target_filename, foldernames)