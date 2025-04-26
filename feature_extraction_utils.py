import os
import glob
from pathlib import Path

import h5py
import numpy as np
import librosa
import string
import pandas as pd
from moviepy.editor import VideoFileClip

import torch
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchvision.transforms import Compose, Lambda, CenterCrop
from pytorchvideo.transforms import Normalize, UniformTemporalSubsample, ShortSideScale


def define_frames_transform(args):
	"""Define the transform of the video frames for later input to the DNN.
	https://pytorch.org/hub/facebookresearch_pytorchvideo_resnet/

	Parameters
	----------
	args : Namespace
		Input arguments.

	Returns
	-------
	transform : object
		Video frames transform.

	"""

	side_size = 256
	mean = [0.45, 0.45, 0.45]
	std = [0.225, 0.225, 0.225]
	crop_size = 256
	num_frames = 8

	# Note that this transform is specific to the slow_R50 model.
	transform = Compose(
		[
			UniformTemporalSubsample(num_frames),
			Lambda(lambda x: x/255.0),
			Normalize(mean, std),
			ShortSideScale(size=side_size),
			CenterCrop(crop_size)
		]
	)

	### Output ###
	return transform


def get_vision_model(args, device):
	"""Load the pre-trained video model.
	https://pytorch.org/hub/facebookresearch_pytorchvideo_resnet/

	# https://pytorch.org/vision/stable/models.html
	# https://pytorch.org/hub/facebookresearch_pytorchvideo_resnet/
	# https://pytorch.org/hub/facebookresearch_pytorchvideo_slowfast/
	# https://pytorch.org/vision/stable/models/generated/torchvision.models.video.mvit_v2_s.html#torchvision.models.video.mvit_v2_s
	# https://pytorch.org/vision/stable/models/generated/torchvision.models.video.swin3d_s.html#torchvision.models.video.swin3d_s
	# The videos from Kinetics-400 are 10s

	Parameters
	----------
	args : Namespace
		Input arguments.
	device : str
		Whether to compute on 'cpu' or 'gpu'.

	Returns
	-------
	feature_extractor : object
		Video model feature extractor object.
	model_layer : str
		Used model layer.

	"""

	### Load the DNN model ###
	model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
	model = model.eval()
	model = model.to(device)
	train_nodes, _ = get_graph_node_names(model)
#	model_layer = 'blocks.4.res_blocks.2.activation'
	model_layer = 'blocks.5.pool'
	feature_extractor = create_feature_extractor(model,
		return_nodes=[model_layer])

	### Output ###
	return feature_extractor, model_layer


def list_movie_splits(args):
	"""List the available movies splits for the selected movie type, for which
	the stimulus features will be extracted.

	Parameters
	----------
	args : Namespace
		Input arguments.

	Returns
	-------
	movie_splits_list : list
		List of movie splits for which the stimulus features are extracted.

	"""

	### List movie splits ###
	# Movie directories
	if args.modality == 'language':
		movie_dir = os.path.join(args.project_dir, 'data',
			'algonauts_2025.competitors', 'stimuli', 'transcripts',
			args.movie_type, args.stimulus_type)
		file_type = 'tsv'
	else:
		movie_dir = os.path.join(args.project_dir, 'data',
			'algonauts_2025.competitors', 'stimuli', 'movies', args.movie_type,
			args.stimulus_type)
		file_type = 'mkv'
	# List the movie splits
	if args.movie_type == 'friends':
		movie_splits_list = [
			x.split("/")[-1].split(".")[0][8:]
			for x in sorted(glob.glob(f"{movie_dir}/*.{file_type}"))
		]
	elif args.movie_type == 'movie10':
		if args.modality != 'language':
			movie_splits_list = [
				x.split("/")[-1].split(".")[0]
				for x in sorted(glob.glob(f"{movie_dir}/*.{file_type}"))
			]
		else:
			movie_splits_list = [
				x.split("/")[-1].split(".")[0][8:]
				for x in sorted(glob.glob(f"{movie_dir}/*.{file_type}"))
			]

	### Output ###
	return movie_splits_list


def extract_visual_features(args, movie_split, feature_extractor, model_layer,
	transform, device, save_dir):
	"""Extract and save the visual features from the .mkv file of the selected
	movie split.

	Parameters
	----------
	args : Namespace
		Input arguments.
	movie_split : str
		Movie split for which the features are extracted and saved.
	feature_extractor : object
		Video model feature extractor object.
	model_layer : str
		Used model layer.
	transform : object
		Video frames transform.
	device : str
		Whether to compute on 'cpu' or 'gpu'.
	save_dir : str
		Save directory.

	"""

	### Temporary directory ###
	temp_dir = os.path.join(save_dir, 'temp')
	if os.path.isdir(temp_dir) == False:
		os.makedirs(temp_dir)

	### Stimulus path ###
	if args.movie_type == 'friends':
		stim_path = os.path.join(args.project_dir, 'data',
			'algonauts_2025.competitors', 'stimuli', 'movies', args.movie_type,
			args.stimulus_type, 'friends_'+movie_split+'.mkv')
	elif args.movie_type == 'movie10':
		stim_path = os.path.join(args.project_dir, 'data',
			'algonauts_2025.competitors', 'stimuli', 'movies', args.movie_type,
			args.stimulus_type, movie_split+'.mkv')

	### Divide the movie in chunks of length TR ###
	clip = VideoFileClip(stim_path)
	start_times = [x for x in np.arange(0, clip.duration, args.tr)][:-1]

	### Loop over movie chunks ###
	visual_features = []
	for start in start_times:

		### Save the chunk clips ###
		clip_chunk = clip.subclip(start, start + args.tr)
		chunk_path = os.path.join(temp_dir, 'visual_'+str(args.stimulus_type)+
			'.mp4')
		clip_chunk.write_videofile(chunk_path, verbose=False)

		### Load the video chunk frames ###
		video_clip = VideoFileClip(chunk_path)
		chunk_frames = [chunk_frames for chunk_frames in video_clip.iter_frames()]
	
		### Format the frames ###
		# Pytorch video models usually require shape:
		# [batch_size, channel, number_of_frame, height, width]
		frames_array = np.transpose(chunk_frames, [3, 0, 1, 2])

		### Transform the frames for DNN feature extraction ###
		inputs = torch.from_numpy(frames_array)
		inputs = transform(inputs)
		inputs = inputs.expand(1, -1, -1, -1, -1)
		inputs = inputs.to(device)

		### Extract the visual features ###
		with torch.no_grad():
			preds = feature_extractor(inputs)
		visual_features.append(
			np.reshape(preds[model_layer].cpu().numpy(), -1))

	### Format the visual features ###
	visual_features = np.array(visual_features, dtype='float32')

	### Save the visual features ###
	out_file = os.path.join(save_dir, args.movie_type+'_'+args.stimulus_type+
		'_features_visual.h5')
	flag = 'a' if Path(out_file).exists() else 'w'
	with h5py.File(out_file, flag) as f:
		group = f.create_group(movie_split)
		group.create_dataset('visual', data=visual_features, dtype=np.float32)


def extract_audio_features(args, movie_split, device, save_dir):
	"""Extract and save the audio features from the .mkv file of the selected
	movie split.

	Parameters
	----------
	args : Namespace
		Input arguments.
	movie_split : str
		Movie split for which the features are extracted and saved.
	device : str
		Whether to compute on 'cpu' or 'gpu'.
	save_dir : str
		Save directory.

	"""

	### Temporary directory ###
	temp_dir = os.path.join(save_dir, 'temp')
	if os.path.isdir(temp_dir) == False:
		os.makedirs(temp_dir)

	### Stimulus path ###
	if args.movie_type == 'friends':
		stim_path = os.path.join(args.project_dir, 'data',
			'algonauts_2025.competitors', 'stimuli', 'movies', args.movie_type,
			args.stimulus_type, 'friends_'+movie_split+'.mkv')
	elif args.movie_type == 'movie10':
		stim_path = os.path.join(args.project_dir, 'data',
			'algonauts_2025.competitors', 'stimuli', 'movies', args.movie_type,
			args.stimulus_type, movie_split+'.mkv')

	### Divide the movie in chunks of length TR ###
	clip = VideoFileClip(stim_path)
	start_times = [x for x in np.arange(0, clip.duration, args.tr)][:-1]

	### Loop over movie chunks ###
	audio_features = []
	for start in start_times:

		### Save the chunk clips ###
		clip_chunk = clip.subclip(start, start + args.tr)
		chunk_path = os.path.join(temp_dir, 'audio_'+str(args.stimulus_type)+
			'.wav')
		clip_chunk.audio.write_audiofile(chunk_path, verbose=False)

		### Load the video chunk audio ###
		y, sr = librosa.load(chunk_path, sr=args.sr, mono=True)

		### Extract the audio features ###
		audio_features.append(np.mean(librosa.feature.mfcc(y=y, sr=sr), 1))

	### Format the audio features ###
	audio_features = np.array(audio_features, dtype='float32')

	### Save the audio features ###
	out_file = os.path.join(save_dir, args.movie_type+'_'+args.stimulus_type+
		'_features_audio.h5')
	flag = 'a' if Path(out_file).exists() else 'w'
	with h5py.File(out_file, flag) as f:
		group = f.create_group(movie_split)
		group.create_dataset('audio', data=audio_features, dtype=np.float32)


def extract_language_features(args, movie_split, model, tokenizer, device,
	save_dir):
	"""Extract and save the language features from the .tsv transcripts of the
	selected movie split.

	Parameters
	----------
	args : Namespace
		Input arguments.
	movie_split : str
		Movie split for which the features are extracted and saved.
	model : object
		Language model.
	tokenizer : object
		Tokenizer corresponding to the language model.
	device : str
		Whether to compute on 'cpu' or 'gpu'.
	save_dir : str
		Save directory.

	"""

	### Stimulus path ###
	if args.movie_type == 'friends':
		stim_path = os.path.join(args.project_dir, 'data',
			'algonauts_2025.competitors', 'stimuli', 'transcripts',
			args.movie_type, args.stimulus_type, 'friends_'+movie_split+'.tsv')
	elif args.movie_type == 'movie10':
		stim_path = os.path.join(args.project_dir, 'data',
			'algonauts_2025.competitors', 'stimuli', 'transcripts',
			args.movie_type, args.stimulus_type, 'movie10_'+movie_split+'.tsv')

	### Read the transcripts ###
	df = pd.read_csv(stim_path, sep = '\t')
	df.insert(loc=0, column="is_na", value=df["text_per_tr"].isna())

	### Empty feature lists ###
	tokens = [] # Tokens of the complete transcripts
	np_tokens = [] # Tokens of only words (without punctuation)
	last_hidden_state = []
	pooler_output = []

	### Loop over text chunks ###
	for i in range(df.shape[0]): # each row/sample of the df corresponds to one fMRI TR

		### Tokenize raw text ###
		if not df.iloc[i]["is_na"]: # Only tokenize if words were spoken during a chunk (i.e., if the chunk is not empty)
			# Tokenize raw text with puntuation (for pooler_output features)
			tr_text = df.iloc[i]["text_per_tr"]
			tokens.extend(tokenizer.tokenize(tr_text))
			# Tokenize without punctuation (for last_hidden_state features)
			tr_np_tokens = tokenizer.tokenize(
				tr_text.translate(str.maketrans('', '', string.punctuation)))
			np_tokens.extend(tr_np_tokens)

		### Extract the pooler_output features ###
		if len(tokens) > 0: # Only extract features if there are tokens available
			# Select the number of tokens used from the current and past chunks,
			# and convert them into IDs
			used_tokens = tokenizer.convert_tokens_to_ids(
				tokens[-(args.num_used_tokens):])
			# IDs 101 and 102 are special tokens that indicate the beginning and
			# end of an input sequence, respectively.
			input_ids = [101] + used_tokens + [102]
			tensor_tokens = torch.tensor(input_ids).unsqueeze(0).to(device)
			# Extract and store the pooler_output features
			with torch.no_grad():
				outputs = model(tensor_tokens)
				pooler_output.append(outputs['pooler_output'][0].cpu().numpy())
		else: # Store NaN values if no tokes are available
			pooler_output.append(np.full(768, np.nan, dtype='float32'))

		### Extract the last_hidden_state features ###
		if len(np_tokens) > 0: # Only extract features if there are tokens available
			np_feat = np.full((args.kept_tokens_last_hidden_state, 768),
				np.nan, dtype='float32')
			# Select the number of tokens used from the current and past chunks,
			# and convert them into IDs
			used_tokens = tokenizer.convert_tokens_to_ids(
				np_tokens[-(args.num_used_tokens):])
			# IDs 101 and 102 are special tokens that indicate the beginning and
			# end of an input sequence, respectively.
			np_input_ids = [101] + used_tokens + [102]
			np_tensor_tokens = \
				torch.tensor(np_input_ids).unsqueeze(0).to(device)
			# Extract and store the last_hidden_state features
			with torch.no_grad():
				np_outputs = model(np_tensor_tokens)
				np_outputs = \
					np_outputs['last_hidden_state'][0][1:-1].cpu().numpy()
			tk_idx = min(args.kept_tokens_last_hidden_state, len(np_tokens))
			np_feat[-tk_idx:, :] = np_outputs[-tk_idx:]
			last_hidden_state.append(np_feat)
		else: # Store NaN values if no tokens are available
			last_hidden_state.append(np.full(
				(args.kept_tokens_last_hidden_state, 768), np.nan,
				dtype='float32'))

	### Convert the language features to float32 ###
	pooler_output = np.array(pooler_output, dtype='float32')
	last_hidden_state = np.array(last_hidden_state, dtype='float32')

	### Save the language features ###
	out_file = os.path.join(save_dir, args.movie_type+'_'+args.stimulus_type+
		'_features_language.h5')
	flag = 'a' if Path(out_file).exists() else 'w'
	with h5py.File(out_file, flag) as f:
		group = f.create_group(movie_split)
		group.create_dataset('language_pooler_output', data=pooler_output,
			dtype=np.float32)
		group.create_dataset('language_last_hidden_state',
			data=last_hidden_state, dtype=np.float32)
