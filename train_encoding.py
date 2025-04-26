"""Train encoding models of fMRI responses to movies. The models consist of
a ridge regression that linearly maps the PCA-downsampled multimodal features
(visual, audio, language) onto the fMRI responses.

The models are trained using stimulus features and fMRI responses for the first
6 seasons of Friends plus Movie10.

The fMRI volumes were acquired with a repetition time (TR) of 1.49 seconds
(i.e., one volume was acquired every 1.49 seconds). To allow for pairing
between stimulus features and fMRI data, in the feature extraction step we also
divided the stimulus features into chunks of 1.49 seconds.

Parameters
----------
subject : int
	Integer indicating the subject for which the encoding model is trained and
	tested. The four challenge subjects are ['1', '2', '3', '5'].
modality : str
	String indicating the stimulus feature modality used to train and test the
	encoding model. Available modalities are ['visual', 'audio', 'language',
	'all'].
excluded_samples_start : int
	Integer indicating the first N fMRI samples that will be excluded and not
	used for model training. The reason for excluding these samples is that due
	to the latency of the hemodynamic response the fMRI responses of first few
	fMRI samples do not yet contain stimulus-related information.
excluded_samples_end : int
	Integer indicating the last N fMRI samples that will be excluded and not
	used for model training. The reason for excluding these samples is that
	stimulus feature samples can be shorter than the fMRI samples, since in
	some cases the fMRI run ran longer than the actual movie. However, note
	that the fMRI timeseries onset is ALWAYS SYNCHRONIZED with movie onset
	(i.e., the first fMRI sample is always synchronized with the first stimulus
	sample).
hrf_delay : int
	fMRI detects the BOLD (Blood Oxygen Level Dependent) response, a signal
	that reflects changes in blood oxygenation levels in response to activity
	in the brain. Blood flow increases to a given brain region in response to
	its activity. This vascular response, which follows the hemodynamic
	response function (HRF), takes time. Typically, the HRF peaks around 5–6
	seconds after a neural event: this delay reflects the time needed for blood
	oxygenation changes to propagate and for the fMRI signal to capture them.
	Therefore, this parameter introduces a delay between stimulus chunks and
	fMRI samples for a better correspondence between input stimuli and the
	brain response. For example, with a hrf_delay of 3, if the fMRI sample of
	interest is 20, the corresponding stimulus sample will be 17.
stimulus_window : int
	Integer indicating how many stimulus feature samples are used to model each
	fMRI sample, starting from the stimulus sample corresponding to the fMRI
	sample of interest, minus the hrf_delay, and going back in time. For
	example, with a 'stimulus_window' of 5, if the fMRI sample of interest is
	20, it will be modeled with stimulus samples [16, 17, 18, 19, 20]. Note
	that this only applies to visual and audio features, since the language
	features were already extracted using transcript words spanning several
	movie samples (thus, each fMRI sample will only be modeled using the
	corresponding language feature sample). Also note that a larger stimulus
	window will increase compute time, since it increases the amount of
	stimulus features used to train and test the fMRI encoding models.
project_dir : str
	Directory of the Algonauts 2025 folder.

"""

import os
import argparse
import numpy as np

from train_encoding_utils import load_fmri
from train_encoding_utils import load_stimulus_features
from train_encoding_utils import train_encoding

parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=int, default=1) # '1' '2' '3' '5'
parser.add_argument('--modality', type=str, default='all') # 'visual' 'audio' 'language' 'all'
parser.add_argument('--excluded_samples_start', type=int, default=5)
parser.add_argument('--excluded_samples_end', type=int, default=5)
parser.add_argument('--hrf_delay', type=int, default=3)
parser.add_argument('--stimulus_window', type=int, default=5)
parser.add_argument('--project_dir', default='../algonauts_2025/', type=str)

args = parser.parse_args()

print('>>> Train encoding models <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Load the fMRI responses for the first 6 Friends seasons and Movie10
# => regression training targets (y)
# =============================================================================
y_train, movie_split_names, movie_split_samples = load_fmri(args)


# =============================================================================
# Load the stimulus features for the first 6 Friends seasons and Movie10
# => regression training predictors (X)
# =============================================================================
X_train = load_stimulus_features(args, movie_split_names, movie_split_samples)


# =============================================================================
# Train the regression-based encoding model
# =============================================================================
model = train_encoding(X_train, y_train)


# =============================================================================
# Save the trained encoding model weights
# =============================================================================
trained_model = {
	'args': args,
	'coef_': model.coef_,
	'intercept_': model.intercept_,
	'alpha_': model.alpha_,
	'n_features_in_': model.n_features_in_,
	}

save_dir = os.path.join(args.project_dir, 'results', 'trained_encoding_models')

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = 'trained_encoding_model_sub-0' + str(args.subject) + \
	'_modality-' + args.modality

np.save(os.path.join(save_dir, file_name), trained_model)
