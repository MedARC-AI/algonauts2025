import os
import sys
import io
import json
import base64
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import h5py
from nilearn import plotting
from nilearn.maskers import NiftiLabelsMasker
import matplotlib.pyplot as plt
import matplotlib


####################################
######## Helping Functions #########
####################################


# Function to compute encoding accuracy (R² scores) for fMRI predictions
def compute_encoding_accuracy(y_test, y_test_pred):
    correlation = np.zeros((y_test.shape[1]), dtype=np.float32)
    for p in range(len(correlation)):
        correlation[p] = pearsonr(y_test[:, p], y_test_pred[:, p])[0]
    return correlation


def generate_zero_scores():
    """
    Generates a dictionary with zero scores for all subjects and average.
    
    Returns
    -------
    dict
        Dictionary containing zero scores for all subjects and the average.
    """
    zero_scores = {
        "sub-average": 0,  # Overall mean accuracy across all subjects

        "sub-01_movie-chaplin": 0,
        "sub-01_movie-mononoke": 0,
        "sub-01_movie-passepartout": 0,
        "sub-01_movie-planetearth": 0,
        "sub-01_movie-pulpfiction": 0,
        "sub-01_movie-wot": 0,
        "sub-01": 0,  # Mean accuracy for sub-01 across all movies

        "sub-02_movie-chaplin": 0,
        "sub-02_movie-mononoke": 0,
        "sub-02_movie-passepartout": 0,
        "sub-02_movie-planetearth": 0,
        "sub-02_movie-pulpfiction": 0,
        "sub-02_movie-wot": 0,
        "sub-02": 0,  # Mean accuracy for sub-02 across all movies

        "sub-03_movie-chaplin": 0,
        "sub-03_movie-mononoke": 0,
        "sub-03_movie-passepartout": 0,
        "sub-03_movie-planetearth": 0,
        "sub-03_movie-pulpfiction": 0,
        "sub-03_movie-wot": 0,
        "sub-03": 0,  # Mean accuracy for sub-03 across all movies

        "sub-05_movie-chaplin": 0,
        "sub-05_movie-mononoke": 0,
        "sub-05_movie-passepartout": 0,
        "sub-05_movie-planetearth": 0,
        "sub-05_movie-pulpfiction": 0,
        "sub-05_movie-wot": 0,
        "sub-05": 0,  # Mean accuracy for sub-05 across all movies
    }

    return zero_scores


def visualize_encoding_accuracy(subject, movie, r, atlas_path, average, title, output_dir):
    """
    Visualizes encoding accuracy (R² scores) on a glass brain plot and saves it.

    Parameters
    ----------
    subject : str
        Identifier for the subject.
    movie : str
        Identifier for the movie.
    r : numpy array
        Correlation values representing encoding accuracy for brain parcels.
    atlas_path : str
        Path to the brain atlas file corresponding to the subject.
    average : float
        Average Pearson's r encoding accuracy across all parcels.
    title : float
        Plot title.
    output_dir : str
        Directory where the visualization plot will be saved.

    """

    # Prepare the masker with the atlas path
    atlas_masker = NiftiLabelsMasker(labels_img=atlas_path)
    atlas_masker.fit()

    # Transform R² values to nifti format
    r_nii = atlas_masker.inverse_transform(r)
        
    display = plotting.plot_glass_brain(
        r_nii, display_mode="lyrz", colorbar=True, title=title
    )

    # Save the plot
    plot_path = os.path.join(output_dir, f"encoding_accuracy_{subject}_movie-{movie}.png")
    display.savefig(plot_path)
    display.close()

    print(f"Saved encoding accuracy plot for {subject} at {plot_path}")


def image_to_base64(image_path):
    """Convert an image file to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("ascii")    


def generate_detailed_results_html(output_dir, ood_movies_dict):
    """
    Generates an HTML file to display detailed results with encoding accuracy plots for each subject and movie,
    embedding the images in base64 format.

    Parameters
    ----------
    output_dir : str
        Directory where the visualization plots are saved.
    """

    # Path to the detailed results HTML file
    html_file = os.path.join(output_dir, 'detailed_results.html')

    # Start the HTML file
    with open(html_file, 'w', encoding='utf-8') as f:

        # Add the accuracy plots averaged across movies and subjects
        overall_plot_path = os.path.join(output_dir, "encoding_accuracy_sub-average_movie-average.png")
        if os.path.exists(overall_plot_path):
            overall_img_b64 = image_to_base64(overall_plot_path)
            f.write('<h2>Mean encoding accuracy across movies and subjects</h2>\n')
            f.write(f'<img src="data:image/png;base64,{overall_img_b64}" alt="Mean encoding accuracy across movies and subjects" style="width:100%; max-width:800px;">\n')

        # Add the accuracy plots of individual subjects
        for subject in ["sub-01", "sub-02", "sub-03", "sub-05"]:
            f.write(f'<h2>Encoding accuracy {subject}</h2>\n')

            # Add the accuracy plots of individual subjects, averaged across movies
            subject_plot_path = os.path.join(output_dir, f"encoding_accuracy_{subject}_movie-average.png")
            if os.path.exists(subject_plot_path):
                subject_img_b64 = image_to_base64(subject_plot_path)
                f.write(f'<h3>Encoding accuracy {subject}, mean across movies</h3>\n')
                f.write(f'<img src="data:image/png;base64,{subject_img_b64}" alt="Encoding accuracy {subject}, movie-average" style="width:100%; max-width:800px;">\n')

            # Add the accuracy plots of individual subjects and movies
            for movie in ood_movies_dict.keys():
                plot_path = os.path.join(output_dir, f"encoding_accuracy_{subject}_movie-{movie}.png")
                if os.path.exists(plot_path):
                    img_b64 = image_to_base64(plot_path)
                    f.write(f'<h3>Encoding accuracy {subject}, movie {movie.capitalize()}</h3>\n')
                    f.write(f'<img src="data:image/png;base64,{img_b64}" alt="Encoding accuracy {subject}, movie {movie.capitalize()}" style="width:100%; max-width:800px;">\n')

        print(f"Generated detailed results HTML at {html_file}")


####################################
######## Checking Format ###########
####################################


# Define input and output directories
submit_dir = os.path.join('/app/input/', 'res')
truth_dir = os.path.join('/app/input/', 'ref')
output_dir = '/app/output'


# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Expected structure for the submission
required_subjects = {"sub-01", "sub-02", "sub-03", "sub-05"} # Required subject keys
n_features = 1000  # Expected number of parcels (features) per TR
required_episodes = ['chaplin1','chaplin2','mononoke1','mononoke2',
                      'passepartout1','passepartout2','planetearth1',
                      'planetearth2','pulpfiction1','pulpfiction2',
                      'wot1','wot2']

ood_movies_dict = {"chaplin": ['chaplin1','chaplin2'],
                   "mononoke": ['mononoke1','mononoke2'],
                   "passepartout": ['passepartout1','passepartout2'],
                   "planetearth": ['planetearth1','planetearth2'],
                   "pulpfiction": ["pulpfiction1", "pulpfiction2"],
                   "wot": ['wot1','wot2']}

# Initialize a list to collect format error messages
format_errors = []

# Check if there is exactly one .npy file in the submission directory
submission_files = [f for f in os.listdir(submit_dir) if f.endswith(".npy")]
if len(submission_files) == 0:
    format_errors.append("Submission Format ERROR: No .npy file found in the submission directory.")
elif len(submission_files) > 1:
    format_errors.append("Submission Format ERROR: Multiple .npy files found in the submission directory. Only one is allowed.")
else:
    # Load the submission file with numpy version compatibility check
    submission_path = os.path.join(submit_dir, submission_files[0])
    try:
        fmri_predicted = np.load(submission_path, allow_pickle=True).item()
    except Exception as e:
        # Check if it's a numpy version compatibility issue
        if 'numpy._core' in str(e) or isinstance(e, ModuleNotFoundError):
            format_errors.append(
                "Submission Format ERROR: Your submission was saved using NumPy 2.x, but the evaluation server runs NumPy 1.x.\n"
                "Please recreate your .npy files using numpy<2.0.\n\n"
                "To fix this:\n"
                " pip install 'numpy<2.0'\n"
                " Then re-save your submission using np.save(..., allow_pickle=True)."
            )
        else:
            format_errors.append(f"Submission Format ERROR: Failed to load .npy file: {str(e)}")
        fmri_predicted = None  # Set to None so we can skip format validation

    # Only proceed with format validation if file was loaded successfully
    if fmri_predicted is not None:
        # Check if loaded object is a dictionary
        if not isinstance(fmri_predicted, dict):
            format_errors.append("Submission Format ERROR: The .npy file does not contain a dictionary.")
        else:
            # Check if all required subjects are present
            missing_subjects = required_subjects.difference(fmri_predicted.keys())
            if missing_subjects:
                format_errors.append(f"Submission Format ERROR: Missing required subjects: {', '.join(sorted(missing_subjects))}")

            # Check each subject's format and episodes
            for subject in required_subjects:
                if subject in fmri_predicted:
                    episodes = fmri_predicted[subject]
                    
                    # Check for missing episodes in submission
                    submission_episodes = set(episodes.keys())
                    missing_episodes = set(required_episodes) - submission_episodes
                    if missing_episodes:
                        format_errors.append(f"Submission Format ERROR: Missing episodes for subject {subject}: {', '.join(sorted(missing_episodes))}")

                    # Load the target sample number
                    data_dir = os.path.join(truth_dir, subject,
                        'target_sample_number', subject+'_ood_fmri_samples.npy')
                    target_samples = np.load(data_dir, allow_pickle=True).item()

                    # Validate that each episode has correct structure, dimensions, and no NaN values
                    for episode, data in episodes.items():
                        if episode not in target_samples:
                            format_errors.append(f"Submission Format ERROR: Unexpected episode key '{episode}' found in predictions for {subject}.")
                            continue
                        if not isinstance(data, np.ndarray):
                            format_errors.append(f"Submission Format ERROR: Predictions for {subject}, episode {episode} are not in ndarray format.")
                        elif data.shape[0] != target_samples[episode]:
                            format_errors.append(f"Submission Format ERROR: Predictions for {subject}, episode {episode} should have {target_samples[episode]} samples (rows), but has {data.shape[0]} instead.")
                        elif data.shape[1] != n_features:
                            format_errors.append(f"Submission Format ERROR: Predictions for {subject}, episode {episode} should have {n_features} features (columns), but has {data.shape[1]} instead.")
                        elif np.isnan(data).any():
                            format_errors.append(f"Submission Format ERROR: NaN values detected in predictions for {subject}, episode {episode}.")


# Print errors if any
if format_errors:
    for error in format_errors:
        print(error)
        print("Will stop evaluation...")
    # Generate and save zero scores
    zero_scores = generate_zero_scores()
    results_json_path = os.path.join(output_dir, "scores.json")
    with open(results_json_path, 'w') as f:
        json.dump(zero_scores, f)
    print(f"Saved zero scores JSON at {results_json_path} due to format errors")
else:
    print("Submission format is correct.")
    print("Will continue with evaluation...")


    ####################################
    ######## Evaluate Results ##########
    ####################################


    # Initialize results dictionaries
    results_parcels_avg = {} # Correlation scores averaged across parcels
    results_parcels_all = {} # Correlation scores of all parcels

    for subject in ["sub-01", "sub-02", "sub-03", "sub-05"]:
        print(f"Evaluating subject {subject}...")

        # Load ground truth data for the subject
        fmri_recorded = {}
        truth_path = os.path.join(truth_dir, f"{subject}/func/{subject}_task-ood_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_bold.h5")
        fmri_all = h5py.File(truth_path, 'r')
        for key, val in fmri_all.items():
            fmri_recorded[key[13:]] = val[:]

        # Loop across OOD movies
        for movie, episodes in ood_movies_dict.items():

            # Append the fMRI responses for the two episodes of each OOD movie,
            # while removing the first 5 and last 5 samples of each episode
            y_test = []
            y_test_pred = []
            for episode in episodes:
                y_test.append(fmri_recorded[episode][5:-5])
                y_test_pred.append(fmri_predicted[subject][episode][5:-5])
            y_test = np.concatenate(y_test, 0)
            y_test_pred = np.concatenate(y_test_pred, 0)

            # Compute the correlation scores for each parcel
            movie_correlation = compute_encoding_accuracy(y_test, y_test_pred)

            # Store the individual subject correlation array for whole-brain visualizations
            results_parcels_all[subject+"_movie-"+movie] = movie_correlation

            # Store the results for individual movies and subjects
            results_parcels_avg[f"{subject}_movie-{movie}"] = float(np.mean(movie_correlation))

            # Visualize the results for individual movies and subjects
            atlas_path = f"{truth_dir}/{subject}/atlas/{subject}_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-dseg_parcellation.nii.gz"
            average = np.round(np.mean(movie_correlation), 4)
            title = 'Encoding accuracy OOD, ' + subject + ', movie-' + movie + ', mean accuracy: ' + str(average)
            visualize_encoding_accuracy(subject, movie, movie_correlation, atlas_path, average=average, title=title, output_dir=output_dir)

        # Store the results for individual subjects averaged across movies
        average = np.mean([results_parcels_avg[subject+'_movie-'+movie] for movie in ["chaplin", "mononoke", "passepartout", "planetearth", "pulpfiction", "wot"]])
        results_parcels_avg[subject] = float(average)

        # Visualize the results for individual subjects averaged across movies
        average = np.round(average, 4)
        subject_correlation = np.mean([results_parcels_all[subject+'_movie-'+movie] for movie in ["chaplin", "mononoke", "passepartout", "planetearth", "pulpfiction", "wot"]], 0)
        results_parcels_all[subject] = subject_correlation
        atlas_path = f"{truth_dir}/{subject}/atlas/{subject}_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-dseg_parcellation.nii.gz"
        title = 'Encoding accuracy OOD, ' + subject + ', movie-average, mean accuracy: ' + str(average)
        visualize_encoding_accuracy(subject, 'average', subject_correlation, atlas_path, average=average, title=title, output_dir=output_dir)

    # Store the results averaged across movies and subjects
    average = np.mean([results_parcels_avg[subject] for subject in ["sub-01", "sub-02", "sub-03", "sub-05"]])
    results_parcels_avg['sub-average'] = float(average)

    # Visualize the results averaged across movies and subjects
    average = np.round(average, 4)
    subject_correlation = np.mean([results_parcels_all[subject] for subject in ["sub-01", "sub-02", "sub-03", "sub-05"]], 0)
    atlas_path = f"{truth_dir}/sub-01/atlas/sub-01_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-dseg_parcellation.nii.gz"
    title = 'Encoding accuracy OOD, subject-average, movie-average, mean accuracy: ' + str(average)
    visualize_encoding_accuracy('sub-average', 'average', subject_correlation, atlas_path, average=average, title=title, output_dir=output_dir)

    # Save results in JSON format for Codabench
    results_json_path = os.path.join(output_dir, "scores.json")
    with open(results_json_path, 'w') as f:
        json.dump(results_parcels_avg, f)
    print(f"Saved results JSON at {results_json_path}")

    # Generate .html file
    print("Generating Detailed results...")
    generate_detailed_results_html(output_dir, ood_movies_dict)
