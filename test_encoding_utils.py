import os
import numpy as np
import h5py
from sklearn.linear_model import Ridge


def load_trained_model(args, subject):
	"""Load the trained encoding model.

	Parameters
	----------
	args : Namespace
		Input arguments.
	subject : int
		Used subject.

	Returns
	-------
	model : object
		Trained ridge regression model.

	"""

	### Load the trained encoding model weights ###
	data_dir = os.path.join(args.project_dir, 'results',
		'trained_encoding_models', 'trained_encoding_model_sub-0'+str(subject)+
		'_modality-'+args.modality+'.npy')
	model_weights = np.load(data_dir, allow_pickle=True).item()

	### Initialize a Ridge regression object ###
	model = Ridge()

	### Add the trained weights to the Ridge regression object ###
	model.coef_ = model_weights['coef_']
	model.intercept_ = model_weights['intercept_']
	model.n_features_in_ = model_weights['n_features_in_']

	### Output ###
	return model


def load_fmri_samples(args, subject):
	"""Load the fMRI sample number for Friends season 7.

	Parameters
	----------
	args : Namespace
		Input arguments.
	subject : int
		Used subject.

	Returns
	-------
	fmri_samples : dict
		Dictionary with the amount of fMRI samples for each Friends season 7
		movie split.

	"""

	### Load the fMRI TR number ###
	data_dir = os.path.join(args.project_dir, 'data',
		'algonauts_2025.competitors', 'fmri', 'sub-0'+str(subject),
		'target_sample_number', 'sub-0'+str(subject)+
		'_friends-s7_fmri_samples.npy')
	fmri_samples = np.load(data_dir, allow_pickle=True).item()

	### Output ###
	return fmri_samples


def load_stimulus_features(args, subject, fmri_samples):
	"""Load the stimulus features for Friends season 7, and prepare them in the
	right format for predicting fMRI responses.

	Parameters
	----------
	args : Namespace
		Input arguments.
	subject : int
		Used subject.
	fmri_samples : dict
		Dictionary with the amount of fMRI samples for each Friends season 7
		movie split.

	Returns
	-------
	stimulus_features : list
		Stimulus features for Friends season 7 episodes.

	"""

	### Load the PCA stimulus features ###
	features = {}
	# Visual
	if args.modality == 'visual' or args.modality == 'all':
		stimuli_dir = os.path.join(args.project_dir, 'results',
			'stimulus_features', 'pca', 'friends_movie10', 'visual',
			'features_test.npy')
		features['visual'] = np.load(stimuli_dir, allow_pickle=True).item()
	# Audio
	if args.modality == 'audio' or args.modality == 'all':
		stimuli_dir = os.path.join(args.project_dir, 'results',
			'stimulus_features', 'pca', 'friends_movie10', 'audio',
			'features_test.npy')
		features['audio'] = np.load(stimuli_dir, allow_pickle=True).item()
	# Language
	if args.modality == 'language' or args.modality == 'all':
		stimuli_dir = os.path.join(args.project_dir, 'results',
			'stimulus_features', 'pca', 'friends_movie10', 'language',
			'features_test.npy')
		features['language'] = np.load(stimuli_dir, allow_pickle=True).item()

	### Empty features variable ###
	stimulus_features = {}

	### Loop over Friends season 7 episodes ###
	for epi, samples in fmri_samples.items():
		stim_features_epi = []

		### Loop over fMRI samples ###
		for s in range(samples):
			# Empty variable containing the stimulus features of all
			# modalities for each sample
			f_all = np.empty(0)

			### Loop across modalities ###
			for mod in features.keys():

				### Visual and audio features ###
				# If visual or audio modality, model each fMRI sample using the
				# N stimulus feature samples up to the fMRI sample of interest
				# minus the hrf_delay (where N is defined by the
				# 'stimulus_window' variable)
				if mod == 'visual' or mod == 'audio':
					# In case there are not N stimulus feature samples up to
					# the fMRI sample of interest minus the hrf_delay (where N
					# is defined by the 'stimulus_window' variable), model the
					# fMRI sample using the first N stimulus feature samples
					if s < (args.stimulus_window + args.hrf_delay):
						idx_start = 0
						idx_end = idx_start + args.stimulus_window
					else:
						idx_start = s - args.hrf_delay - args.stimulus_window + 1
						idx_end = idx_start + args.stimulus_window
					# In case there are less visual/audio feature samples than
					# fMRI samples minus the hrf_delay, use the last N
					# visual/audio feature samples available (where N is
					# defined by the 'stimulus_window' variable)
					if idx_end > len(features[mod][epi]):
						idx_end = len(features[mod][epi])
						idx_start = idx_end - args.stimulus_window
					f = features[mod][epi][idx_start:idx_end]
					f_all = np.append(f_all, f.flatten())

				### Language features ###
				# Since language features already consist of embeddings
				# spanning several samples, only model each fMRI sample using
				# the corresponding stimulus feature sample minus the hrf_delay
				elif mod == 'language':
					# In case there are no language features for the fMRI
					# sample of interest minus the hrf_delay, model the fMRI
					# sample using the first language feature sample
					if s < args.hrf_delay:
						idx = 0
					else:
						idx = s - args.hrf_delay
					# In case there are fewer language feature samples than
					# fMRI samples minus the hrf_delay, use the last language
					# feature sample available
					if idx >= (len(features[mod][epi]) - args.hrf_delay):
						f = features[mod][epi][-1,:]
					else:
						f = features[mod][epi][idx]
					f_all = np.append(f_all, f.flatten())

			### Append the stimulus features of all modalities for this sample ###
			stim_features_epi.append(f_all)

		### Format the stimulus features ###
		stimulus_features[epi] = np.asarray(stim_features_epi,
			dtype=np.float32)

	### Output ###
	return stimulus_features


def load_fmri(args, subject):
	"""Load and format the recorded fMRI responses for Friends season 7.

	Parameters
	----------
	args : Namespace
		Input arguments.
	subject : int
		Used subject.

	Returns
	-------
	fmri : dict
		Dictionary containing the fMRI responses for Friends season 7.

	"""

	### Load the fMRI responses ###
	fmri_file = 'sub-0' + str(subject) + '_task-friends_' + \
		'space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_' + \
		'desc-s7_bold.h5'
	fmri_dir = os.path.join(args.project_dir, 'data', 'algonauts_2025.admin',
		'fmri', 'sub-0'+str(subject), 'func', fmri_file)
	fmri_all = h5py.File(fmri_dir, 'r')

	### Extract the fMRI responses ###
	fmri = {}
	for key, val in fmri_all.items():
		fmri[key[13:]] = val[:]

	### Output ###
	return fmri
