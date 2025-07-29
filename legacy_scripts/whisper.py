import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import fnmatch
from typing import Any, Dict, List, Tuple
from torch import nn

class HuggingFaceFeatureExtractor:
    """
    A feature extractor for Hugging Face (or any PyTorch) models that captures
    intermediate activations from any layer specified by exact name or glob pattern.

    Example usage:
        from transformers import BertModel
        model = BertModel.from_pretrained("bert-base-uncased")
        # Specify layers (using glob patterns is supported)
        layers_to_extract = ["encoder.layer.*.output"]

        # Using the extractor as a context manager ensures hooks are removed automatically.
        with HuggingFaceFeatureExtractor(model, layers_to_extract, detach=True) as extractor:
            # Perform a forward pass as usual
            outputs = model(input_ids, attention_mask=mask)
            # Get a copy of the extracted features
            features = extractor.features
            # Now 'features' is a dict mapping layer names to their activation tensors.
    """

    def __init__(self, model: nn.Module, layers: List[str], detach: bool = True):
        self.model = model
        self.detach = detach
        # Expand layer patterns into full module names
        self.layers = self._expand_layers(model, layers)
        self._features: Dict[str, Any] = {}
        self._handles: Dict[str, Any] = {}
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks on each specified layer."""
        for layer in self.layers:
            sub_module = self.model.get_submodule(layer)
            handle = sub_module.register_forward_hook(self._make_hook(layer))
            self._handles[layer] = handle

    def _make_hook(self, layer_name: str):
        def hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any):
            # Optionally detach to break the graph and save memory.
            self._features[layer_name] = output.detach() if self.detach else output
        return hook

    def clear(self):
        """Clear the stored features before a new forward pass."""
        self._features.clear()

    @property
    def features(self) -> Dict[str, Any]:
        """Return a copy of the captured features."""
        return dict(self._features)

    def __call__(self, *args, **kwargs) -> Any:
        """
        Run the model forward. This automatically clears previous features,
        then performs a forward pass, capturing intermediate activations.
        Returns the model's original output.
        """
        self.clear()
        return self.model(*args, **kwargs)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self._handles.values():
            handle.remove()
        self._handles.clear()

    def __enter__(self):
        """Enter context: hooks are already registered."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit context: remove all hooks."""
        self.remove_hooks()

    @staticmethod
    def _expand_layers(model: nn.Module, layers: List[str]) -> List[str]:
        """
        Expand a list of layer names and/or glob patterns to all matching module names
        in the given model. Raises an error if a specified name or pattern doesn't match.
        """
        all_layers = [name for name, _ in model.named_modules() if name]  # skip the root module ''
        all_layers_set = set(all_layers)
        expanded = []
        special_chars = set("*?[]")
        for layer in layers:
            if not any(char in layer for char in special_chars):
                if layer not in all_layers_set:
                    raise ValueError(f"Layer '{layer}' not found in the model.")
                expanded.append(layer)
            else:
                matches = fnmatch.filter(all_layers, layer)
                if not matches:
                    raise ValueError(f"No layers match the pattern '{layer}'.")
                expanded.extend(matches)
        return expanded


import pandas as pd
import torchaudio
import cv2
import torch
import torch.nn.functional as F
from typing import Tuple, List, Callable




def load_transcript(
    path: str
) -> pd.DataFrame:
    """
    Loads a transcript file (TSV) into a pandas DataFrame.

    Parameters:
        path (str): Path to the transcript file.

    Returns:
        pd.DataFrame: DataFrame containing the transcript data.
    """
    try:
        df = pd.read_csv(path, sep='\t')
        return df
    except Exception as e:
        raise RuntimeError(f"Error loading transcript from {path}: {e}")


def load_audio(
    path: str,
    sampling_rate: int = 48000,
    stereo: bool = True
) -> (torch.Tensor, int):
    """
    Loads an audio file using torchaudio, converts the waveform to half precision,
    optionally converts stereo audio to mono, resamples it to the specified sampling_rate
    if needed, and returns the waveform and sample rate.

    Parameters:
        path (str): Path to the audio file.
        sampling_rate (int): Desired sampling rate for the output waveform.
        stereo (bool): If False, converts stereo audio to mono.

    Returns:
        tuple: (waveform_fp16, sampling_rate) where waveform_fp16 is a tensor in float16.
    """
    try:
        # Set the backend to 'ffmpeg' if available
        torchaudio.set_audio_backend("ffmpeg")
        waveform, orig_sr = torchaudio.load(path)

        # Convert to mono if stereo is False and the waveform has multiple channels
        if not stereo and waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if original sample rate is different from the desired sampling rate
        if orig_sr != sampling_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=sampling_rate)
            waveform = resampler(waveform)

        # Convert the waveform to half precision (float16)
        waveform_fp16 = waveform.half()
        del waveform
        return waveform_fp16, sampling_rate
    except Exception as e:
        raise RuntimeError(f"Error loading audio from {path}: {e}")


def load_video(
    path: str,
    resolution: Tuple[int, int] = (224, 224),
    tensor_dtype: torch.dtype = torch.float16,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Loads a video file, reads its frames, converts each frame from BGR to RGB,
    resizes to 224x224, and returns a tensor containing all frames.

    Parameters:
        path (str): Path to the video file.

    Returns:
        torch.Tensor: Tensor of shape [num_frames, 3, 224, 224] containing the video frames.
    """
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        raise IOError("Cannot open video file: {}".format(path))

    # Get video FPS and calculate number of frames for 10 seconds
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames_to_read = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if verbose:
        print("Total number of frames in the video:", cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("Original Resolution:", (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print("FPS:", fps)
        print("Duration (seconds):", num_frames_to_read / fps)
        print("Target Resolution:", resolution)

    frames = torch.zeros(num_frames_to_read, 3, 224, 224, dtype=tensor_dtype)

    for i in range(num_frames_to_read):
        ret, frame = cap.read()

        if not ret:
            break

        # Optionally, convert the frame from BGR to RGB (if needed)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the frame (numpy array) to a torch tensor and permute dimensions to [C, H, W]
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1) 
        # Resize the frame to 224x224
        frame_tensor = torch.nn.functional.interpolate(frame_tensor.unsqueeze(0), size=224, mode='bilinear', align_corners=False)
        frames[i] = frame_tensor

    cap.release()

    if verbose:
        print(f"Read {len(frames)} frames.")
        print(f"Frames shape: {frames.shape}")

    return frames

audio_test, sampling_rate = load_audio('/data/algonauts_2025.competitors/stimuli/movies/ood/pulpfiction/task-pulpfiction1_video.mkv', sampling_rate = 16000, stereo = False)

import torch
from transformers import WhisperModel, AutoProcessor, pipeline
from datasets import load_dataset
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
cache_dir = "/home/mihir/projects/hf_cache"
os.makedirs(cache_dir, exist_ok=True)

model_id = "openai/whisper-large-v3" #"openai/whisper-small"

model = WhisperModel.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, cache_dir=cache_dir
)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)


from transformers import BertModel, BertTokenizer
import torch


layers_to_extract = ["layer_norm", 'layers.31.fc2', 'layers.25.fc2', 'layers.12.fc2']



from pathlib import Path
import h5py
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def extract_features(
    parts: List[str],
    movies_base: str,
    transcripts_base: str,
    output_dir: str,
    extraction_fn: Callable,
    interval: int = 1.49,
    verbose: bool = True,
    modality: str = 'all',
    ood: bool = False
):
    """
    Extracts features from video/audio stimuli and saves them to HDF5 files.

    This function iterates through specified movie files, loads the necessary data
    (video, audio, and optionally transcripts), processes the data in intervals,
    and uses a provided extraction function to generate features, which are then
    saved. It is robust to missing transcript files.
    """
    movies_base = Path(movies_base)
    transcripts_base = Path(transcripts_base)

    # 1. --- Initial Setup and Verification ---
    if not movies_base.exists():
        raise FileNotFoundError(f"Movies directory not found: {movies_base}")
    if not transcripts_base.exists():
        raise FileNotFoundError(f"Transcripts directory not found: {transcripts_base}")

    # 2. --- Iterate Through Movie Files ---
    for folder in movies_base.rglob('*'):
        if not (folder.is_dir() and folder.name in parts):
            continue

        for movie_file in folder.glob('*.mkv'):
            try:
                rel_folder = folder.relative_to(movies_base)
            except ValueError:
                continue

            # 3. --- Determine Transcript Path ---
            if ood:
                ood_suffix = str(movie_file.with_suffix('.tsv').name)[5:].replace('_video', '')
                if not os.path.splitext(ood_suffix)[0][-1].isdigit():
                    print(f"skipping {ood_suffix}")
                    continue
                transcript_file = transcripts_base / rel_folder / f"ood_{ood_suffix}.tsv"
                # print(ood_suffix)
                # import sys; sys.exit()

            else:
                transcript_file = transcripts_base / rel_folder / movie_file.with_suffix('.tsv').name

            if verbose:
                print(f"Movie:      {movie_file}")
                print(f"Transcript: {transcript_file}")

            # 4. --- Load Stimuli Data ---
            video, audio, transcript, sample_rate, fps_video = None, None, None, None, 30.0
            if modality in ['all', 'video']:
                # Assumes load_video returns a tuple: (video_tensor, fps)
                video, fps_video = load_video(movie_file, verbose=verbose)
            if modality in ['all', 'audio', 'video', 'transcript']:
                audio, sample_rate = load_audio(movie_file)

            if transcript_file.exists() and (modality == 'all' or modality == 'transcript'):
                transcript = load_transcript(transcript_file)
            else:
                transcript = None
                if verbose:
                    print(f"Transcript file not found or not requested for {movie_file.name}, proceeding without it.")

            # 5. --- Process and Save Features ---
            total_duration = audio.shape[1] / sample_rate
            num_intervals = int(total_duration // interval)

            if verbose:
                print(f"Total duration: {total_duration:.2f} seconds")
                print(f"Number of intervals: {num_intervals}")

            output_folder = Path(output_dir) / rel_folder
            output_folder.mkdir(parents=True, exist_ok=True)
            output_file = output_folder / movie_file.with_suffix('.h5').name

            if verbose:
                print(f"Output file: {output_file}")

            with h5py.File(output_file, 'w') as f:
                features_datasets = {}
                for i in tqdm(range(num_intervals), desc=f"Processing {movie_file.name}"):
                    # Extract a single time-based interval from the full media
                    video_section, audio_section, transcript_section = extract_section(
                        video, audio, transcript, interval, i, sample_rate, modality
                    )

                    if audio_section is None or audio_section.shape[1] == 0:
                        continue

                    # Get features from the model for this interval
                    output_features = extraction_fn(video_section, audio_section, transcript_section, verbose)

                    # 6. --- Save Features to HDF5 ---
                    for layer_name, tensor in output_features.items():
                        # The tensor from the model is likely (1, feature_dim), so we squeeze it
                        feature_vector = tensor.cpu().numpy().squeeze(0)

                        if layer_name not in features_datasets:
                            # Create the HDF5 dataset on the first interval
                            features_datasets[layer_name] = f.create_dataset(
                                layer_name,
                                shape=(1,) + feature_vector.shape,
                                maxshape=(None,) + feature_vector.shape, # Allow resizing
                                dtype=np.float16,
                                chunks=True,
                            )
                            features_datasets[layer_name][0] = feature_vector
                        else:
                            # Append to the existing dataset
                            ds = features_datasets[layer_name]
                            ds.resize(ds.shape[0] + 1, axis=0)
                            ds[-1] = feature_vector


def extract_section(
    video: torch.Tensor,
    audio: torch.Tensor,
    transcript: pd.DataFrame,
    interval: float,
    index: int,
    sample_rate: int,
    modality: str = 'all',
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Extracts a section of audio, video, and transcript data based on the interval and index.

    Parameters:
        audio (torch.Tensor): Tensor containing audio waveform (2, num_samples).
        video (torch.Tensor): Tensor containing video frames (num_frames, 3, 224, 224).
        transcript (pd.DataFrame): DataFrame containing transcript data.
        interval (float): Interval (in seconds) at which to extract features.
        index (int): Index of the interval to extract.
        sample_rate (int): Sample rate of the audio waveform.
        modality (str): Modality to extract features from. Default is 'all'. Options are 'video', 'audio', 'transcript'. The function will return None for the other modalities.

    Returns:
        tuple: (audio_section, video_section, transcript_section) where:
            - audio_section is a torch.Tensor containing the audio data for the section.
            - video_section is a torch.Tensor containing the video data for the section.
            - transcript_section is a list of strings containing the transcript data for the section.
    """
    audio_section = None
    video_section = None
    transcript_section = None

    # Compute the start and end times for the section.
    start_time = index * interval
    end_time = (index + 1) * interval

    if modality == 'all' or modality == 'audio':
        # Extract audio data for the section.
        audio_start = int(start_time * sample_rate)
        audio_end = int(end_time * sample_rate)
        audio_section = audio[:, audio_start:audio_end]

    if modality == 'all' or modality == 'video':
        # Extract video data for the section.
        fps_video = 30 # 29.97
        frame_start = round(start_time * fps_video)
        frame_end = round(end_time * fps_video)
        video_section = video[frame_start:frame_end]

    if modality == 'all' or modality == 'transcript':
        # Since transcript is split into rows each row corresponding to an interval of 1.49 seconds
        # we need to extract the corresponding row for the current interval
        # TODO: Write logic to extract precisely with other intervals as well
        transcript_section = transcript['words_per_tr'].iloc[index]

    return video_section, audio_section,  transcript_section

# Example usage:
parts = ["chaplin"]
movies_base = "/data/algonauts_2025.competitors/stimuli/movies"
transcripts_base = "/data/algonauts_2025.competitors/stimuli/transcripts"
out_dir = '/home/mihir/projects/datasets/whisper_OOD'


from functools import wraps

# Define the extractor.
extractor = HuggingFaceFeatureExtractor(model.encoder, layers_to_extract, detach=True)

def extract_fn(video, audio, transcript, verbose):
    with torch.no_grad():
        inputs = processor(audio[0][:int(sampling_rate*1.49)], sampling_rate = sampling_rate, return_tensors = 'pt')
        
        # Perform feature extraction here
        outputs = extractor(inputs['input_features'].half().to(device))

    # Retrieve the extracted features.
    features = extractor.features
    dict_return = {}
    for layer_name, activation in features.items():
        # Taking the mean bcz the features are so large
        dict_return[layer_name] = activation.mean(1)
        
    extractor.clear()
    return dict_return

extract_features(parts = parts, movies_base = movies_base, transcripts_base = transcripts_base, output_dir = out_dir, extraction_fn = extract_fn, verbose = True, modality = 'audio', ood=True)
