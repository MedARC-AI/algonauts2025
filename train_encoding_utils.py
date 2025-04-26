import os
import numpy as np
import h5py
from sklearn.linear_model import RidgeCV


def load_fmri(args):
	"""Load and format the fMRI responses for the first 6 seasons of Friends
	and Movie10, used for training the encoding models.

	Parameters
	----------
	args : Namespace
		Input arguments.

	Returns
	-------
	fmri : float
		fMRI responses for the first 6 seasons of Friends and Movie10.
	movie_split_names : list
		List with the movie split names.
	movie_split_samples : list
		List with the amount of retained fMRI samples per movie split.

	"""

	### Load the fMRI responses ###
	fmri_dir = os.path.join(args.project_dir, 'data',
		'algonauts_2025.competitors', 'fmri', 'sub-0'+str(args.subject),
		'func')
	fmri_file_friends = 'sub-0' + str(args.subject) + \
		'_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_' + \
		'parcel-1000Par7Net_desc-s123456_bold.h5'
	fmri_file_movie10 = 'sub-0' + str(args.subject) + \
		'_task-movie10_space-MNI152NLin2009cAsym_atlas-Schaefer18_' + \
		'parcel-1000Par7Net_bold.h5'
	fmri_friends = h5py.File(os.path.join(fmri_dir, fmri_file_friends), 'r')
	fmri_movie10 = h5py.File(os.path.join(fmri_dir, fmri_file_movie10), 'r')

	### Empty data lists ###
	fmri = []
	movie_split_names = []
	movie_split_samples = []

	### Extract the fMRI responses for Friends ###
	for key, val in fmri_friends.items():
		# Exclude first and last fMRI samples
		fmri_part = val[args.excluded_samples_start:-args.excluded_samples_end]
		fmri.append(fmri_part)
		movie_split_names.append(key[13:])
		movie_split_samples.append(len(fmri_part))

	### Extract the fMRI responses for Movie10 ###
	for key, val in fmri_movie10.items():
		# Exclude first and last fMRI samples
		fmri_part = fmri_movie10[key]\
			[args.excluded_samples_start:-args.excluded_samples_end]
		fmri.append(fmri_part)
		if key[13:20] == 'figures':
			# Omit the '_run-*' ending in the episode name
			movie_split_names.append(key[13:22])
		elif key[13:17] == 'life':
			# Omit the '_run-*' ending in the episode name
			movie_split_names.append(key[13:19])
		else:
			movie_split_names.append(key[13:])
		movie_split_samples.append(len(fmri_part))

	### Format the fMRI responses ###
	fmri = np.concatenate(fmri, axis=0)

	### Output ###
	return fmri, movie_split_names, movie_split_samples


def load_stimulus_features(args, movie_split_names, movie_split_samples):
	"""Load the stimulus features for the first 6 seasons of Friends and
	Movie10, and align them to the fMRI responses for training encoding
	models.

	The goal is to bring the fMRI responses to a (Train Samples x Parcels)
	format, and the stimulus features to a (Train Samples x Features) format,
	with the Train samples dimension matching between both arrays. You will use
	these formatted data later to train encoding models.

	While selecting the fMRI responses for the train set, the first and last
	fMRI samples are excluded based on the 'excluded_samples_start' and
	'excluded_samples_end' variables.

	First, the stimulus features for the train set are selected based on
	available fMRI responses for a given subject: if a subject is missing fMRI
	responses for a given movie split, the corresponding stimulus features for
	that split will not be loaded. Next, the stimulus feature samples are
	aligned with the fMRI response samples (using the 'excluded_samples_start'
	and 'hrf_delay' variables). Since fMRI responses are influenced by
	stimulation up to several seconds in the past, for the visual and audio
	modalities the N stimulus feature samples up to the fMRI sample of interest
	minus the 'hrf_delay' are appended and used to model this fMRI sample,
	where N is defined by the 'stimulus_window' variable. Since the language
	features were already extracted using transcript words spanning the
	duration of several movie samples, each fMRI sample will only be modeled
	using the corresponding language feature sample minus the 'hrf_delay'.
	Finally, the features of different modalities are concatenated together to
	model fMRI responses using a multi-modal stimulus feature space.

	Parameters
	----------
	args : Namespace
		Input arguments.
	movie_split_names : list
		List with the movie split names.
	movie_split_samples : list
		List with the amount of retained fMRI samples per movie split.

	Returns
	-------
	stim_features : float
		Stimulus features for the first 6 Friends seasons and Movie10.

	"""

	### Load the PCA-downsampled stimulus features ###
	features = {}
	# Visual
	if args.modality == 'visual' or args.modality == 'all':
		stimuli_dir = os.path.join(args.project_dir, 'results',
			'stimulus_features','pca', 'friends_movie10', 'visual',
			'features_train.npy')
		features['visual'] = np.load(stimuli_dir, allow_pickle=True).item()
	# Audio
	if args.modality == 'audio' or args.modality == 'all':
		stimuli_dir = os.path.join(args.project_dir, 'results',
			'stimulus_features','pca', 'friends_movie10', 'audio',
			'features_train.npy')
		features['audio'] = np.load(stimuli_dir, allow_pickle=True).item()
	# Language
	if args.modality == 'language' or args.modality == 'all':
		stimuli_dir = os.path.join(args.project_dir, 'results',
			'stimulus_features','pca', 'friends_movie10', 'language',
			'features_train.npy')
		features['language'] = np.load(stimuli_dir, allow_pickle=True).item()

	### Loop over movie splits ###
	stim_features = []
	# Loop over fMRI movie splits, to ensure that we only select stimulus
	# features for movie splits for which there actually are fMRI responses
	for m, split in enumerate(movie_split_names):

		### Loop over fMRI samples ###
		# Loop over fMRI samples, to ensure that the selected stimulus features
		# are aligned with the corresponding fMRI TRs
		for s in range(movie_split_samples[m]):
			# Empty variable containing the stimulus features of all modalities
			# for each fMRI sample
			f_all = np.empty(0)

			### Loop across modalities ###
			for mod in features.keys():

				### Visual and audio features ###
				# If visual or audio modality, model each fMRI sample using
				# the N stimulus feature samples up to the fMRI sample of
				# interest minus the hrf_delay (where N is defined by the
				# 'stimulus_window' variable)
				if mod == 'visual' or mod == 'audio':
					# In case there are not N stimulus feature samples up to
					# the fMRI sample of interest minus the hrf_delay (where N
					# is defined by the 'stimulus_window' variable), model the
					# fMRI sample using the first N stimulus feature samples
					if s < (args.stimulus_window + args.hrf_delay):
						idx_start = args.excluded_samples_start
						idx_end = idx_start + args.stimulus_window
					else:
						idx_start = s + args.excluded_samples_start - \
							args.hrf_delay - args.stimulus_window + 1
						idx_end = idx_start + args.stimulus_window
					# In case there are less visual/audio feature samples than
					# fMRI samples minus the hrf_delay, use the last N
					# visual/audio feature samples available (where N is
					# defined by the 'stimulus_window' variable)
					if idx_end > len(features[mod][split]):
						idx_end = len(features[mod][split])
						idx_start = idx_end - args.stimulus_window
					f = features[mod][split][idx_start:idx_end]
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
					if s < args.hrf_delay:
						idx = args.excluded_samples_start
					else:
						idx = s + args.excluded_samples_start - args.hrf_delay
					# In case there are fewer language feature samples than
					# fMRI samples minus the hrf_delay, use the last language
					# feature sample available
					if idx >= (len(features[mod][split]) - args.hrf_delay):
						f = features[mod][split][-1,:]
					else:
						f = features[mod][split][idx]
					f_all = np.append(f_all, f.flatten())

			### Append the stimulus features of all modalities for this sample ###
			stim_features.append(f_all)

	### Convert the aligned features to a numpy array ###
	stim_features = np.asarray(stim_features, dtype=np.float32)

	### Output ###
	return stim_features


def train_encoding(X_train, y_train):
	"""Train encoding models of fMRI responses to movies. The models consist of
	a ridge regression (with built-in cross-validation), that linearly maps the
	PCA-downsampled multimodal features (visual, audio, language) onto the fMRI
	responses.

	https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html

	Parameters
	----------
	stim_features : float
		Stimuli features for the first 6 Friends seasons, used as the regression
		target.
	fmri : float
		fMRI responses for the first 6 Friends seasons, used as regression
		predictors.

	Returns
	-------
	model : object
		Trained ridge regression model.

	"""

	### Define the Ridge regression alphas ###
	alphas = np.asarray((1000000, 100000, 10000, 1000, 100, 10, 1, 0.5, 0.1,
		0.05, 0.01, 0.005, 0.001, 0.0001, 0.00001))

	### Train the Ridge regression ###
	model = RidgeCV(alphas=alphas, cv=None, alpha_per_target=True)
	model.fit(X_train, y_train)
	model.score(X_train, y_train)

	### Output ###
	return model
