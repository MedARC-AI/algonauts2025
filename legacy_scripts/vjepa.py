
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os


import fnmatch
from typing import Any, Dict, List, Tuple
from torch import nn
from PIL import Image
import ast 

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

    def __init__(self, model: nn.Module, layers: List[str], detach: bool = True, call_fn: str = "forward"):
        self.model = model
        self.detach = detach
        # Expand layer patterns into full module names
        self.layers = self._expand_layers(model, layers)
        self._features: Dict[str, Any] = {}
        self._handles: Dict[str, Any] = {}
        self._register_hooks()
        self._call_fn = call_fn  

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
        fn = getattr(self.model, self._call_fn, None)
        if fn is None or not callable(fn):
            raise AttributeError(f"Model has no callable '{self._call_fn}' method.")
        return fn(*args, **kwargs)

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


# ### Define functions for loading video/audio/transcript

# In[3]:


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
    resolution: Tuple[int, int] = None, # = (224, 224),
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

    if resolution is None:
        H, W = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    else:
        H, W = resolution

    frames = torch.zeros(num_frames_to_read, 3, H, W, dtype=tensor_dtype)

    for i in range(num_frames_to_read):
        ret, frame = cap.read()

        if not ret:
            break

        # Optionally, convert the frame from BGR to RGB (if needed)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the frame (numpy array) to a torch tensor and permute dimensions to [C, H, W]
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1) 
        # Resize the frame to 224x224
        frame_tensor = torch.nn.functional.interpolate(frame_tensor.unsqueeze(0), size=(H,W), mode='bilinear', align_corners=False)
        frames[i] = frame_tensor

    cap.release()

    if verbose:
        print(f"Read {len(frames)} frames.")
        print(f"Frames shape: {frames.shape}")

    return frames, fps


# ### Define main function for iterating over the files and write .h5 for each one 

# In[ ]:


from pathlib import Path
import h5py
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import math


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
    Extracts features from the specified parts of the dataset using the provided extraction function.

    Parameters:
        parts (List[str]): List of parts to extract features from. This is the subdirectory name under friends and movie10 folders.
        movies_base (str): Path to the base directory containing movie files.
        transcripts_base (str): Path to the base directory containing transcript files.
        output_dir (str): Path to the output directory where features will be saved.
        interval (int): Interval (in seconds) at which to extract features. Default is 1.49 seconds (the TR for the dataset).
        extraction_fn (function): Function that extracts features from the stimuli. The function should take the following arguments:
            - video: torch.Tensor containing video frames (num_frames, 3, 224, 224)
            - audio: torch.Tensor containing audio waveform (2, num_samples)
            - transcript: array containing strings of words (num_words,)
            - verbose: bool indicating whether to print verbose output.
            and should return a dictionary mapping layer names to extracted features as torch.Tensor.
        verbose (bool): Whether to print verbose output.
        modality (str): Modality to extract features from. Default is 'all'. Options are 'video', 'audio', 'transcript'.


    """
    global video_section_g, audio_section_g, transcript_section_g

    movies_base = Path(movies_base)
    transcripts_base = Path(transcripts_base)

    # Verify that the base directories exist.
    if not movies_base.exists():
        raise FileNotFoundError(f"Movies directory not found: {movies_base}")
    if not transcripts_base.exists():
        raise FileNotFoundError(f"Transcripts directory not found: {transcripts_base}")

    # Iterate through all directories under movies_base.
    for folder in movies_base.rglob('*'):
        if folder.is_dir() and folder.name in parts:
            # Iterate through mkv files in the matched directory.
            for movie_file in folder.glob('*.mkv'):
                # Compute the relative path from movies_base.
                try:
                    rel_folder = folder.relative_to(movies_base)
                except ValueError:
                    # Skip directories that are not under movies_base.
                    continue

                print(rel_folder)
                if "friends" in str(rel_folder):
                    # Build the corresponding transcript file path.
                    transcript_file = transcripts_base / rel_folder / movie_file.with_suffix('.tsv').name
                elif ood:
                    ood_suffix = str(movie_file.with_suffix('.tsv').name)[5:].replace('_video', '')
                    if not os.path.splitext(ood_suffix)[0][-1].isdigit():
                        print(f"skipping {ood_suffix}")
                        continue
                    transcript_file = transcripts_base / rel_folder / f"ood_{ood_suffix}"
                else:
                    transcript_file = transcripts_base / rel_folder / f"movie10_{movie_file.with_suffix('.tsv').name}"

                print(f"Movie:      {movie_file}")
                print(f"Transcript: {transcript_file}")

                # Load video frames, audio waveform, and transcript.
                video, audio, transcript, sample_rate, fps_video = None, None, None, None, None
                if modality == 'all' or modality == 'video':
                    video, fps_video = load_video(movie_file, verbose=verbose, resolution = (224,224))
                if modality == 'all' or modality == 'audio' or modality == 'video' or modality == 'transcript':
                    audio, sample_rate = load_audio(movie_file)
                if modality == 'all' or modality == 'transcript':
                    transcript = load_transcript(transcript_file)

                # round fps video
                # if fps_video:
                #     fps_video = round(fps_video)

                if transcript is not None:
                    transcript = resample_transcript(transcript, interval)

                total_duration = audio.shape[1] / sample_rate
                num_intervals = math.ceil(total_duration / interval)

                if verbose:
                    print(f"Total duration: {total_duration:.2f} seconds")
                    print(f"Number of intervals: {num_intervals}")

                # Create the output directory if it doesn't exist.
                output_folder = Path(output_dir) / rel_folder
                output_folder.mkdir(parents=True, exist_ok=True)

                # Create a h5 file to store the features.
                output_file = output_folder / movie_file.with_suffix('.h5').name

                if verbose:
                    print(f"Output file: {output_file}")

                # Create a HDF5 file to store the features.
                with h5py.File(output_file, 'w') as f:
                    features_datasets = {} 
                    # Extract features at each interval.
                    for i in tqdm(range(num_intervals)):
                        video_section, audio_section, transcript_section = extract_section(
                            video, audio, transcript, interval, i, sample_rate, modality, fps_video
                        )

                        # if i == 100:
                        #     video_section_g = video_section
                        #     audio_section_g = audio_section
                        #     transcript_section_g = transcript_section


                        #     # convert the video_section from int8 to float32
                        #     video_section = video_section.int()
                        #     # plot the first and the last frame of the video
                        #     plt.imshow(video_section[0].permute(1, 2, 0).cpu().numpy())
                        #     plt.show()
                        #     plt.imshow(video_section[-1].permute(1, 2, 0).cpu().numpy())    
                        #     plt.show()


                        #     # display the audio section as an html audio element
                        #     torchaudio.save("audio.wav", audio_section.float(), sample_rate)
                        #     from IPython.display import Audio
                        #     Audio("audio.wav")


                        #     # print the transcript section
                        #     print(transcript_section)

                        #     # break for testing
                        #     assert False



                        output_features = extract_fn(video_section, audio_section, transcript_section, verbose)

                        for layer_name, tensor in output_features.items():
                            # Convert the tensor to a numpy array (on CPU) before storing.
                            tensor_np = tensor.cpu().numpy()
                            if layer_name not in features_datasets:
                                # Create a new dataset and initialize it with the first interval's data.
                                features_datasets[layer_name] = f.create_dataset(
                                    layer_name,
                                    data=tensor_np[np.newaxis, ...],
                                    maxshape=(None,) + tensor_np.shape,
                                    dtype=np.float16,
                                    chunks=True,
                                )
                            else:
                                ds = features_datasets[layer_name]
                                ds.resize(ds.shape[0] + 1, axis=0)
                                ds[-1] = tensor_np

                        # if features_dataset is None:
                        #     features_max_shape = (None,) + output_features.shape
                        #     print(features_max_shape, output_features.shape)
                        #     features_dataset = f.create_dataset(
                        #         'features', 
                        #         shape= output_features.unsqueeze(0).shape,
                        #         maxshape=features_max_shape,
                        #         dtype=np.float16,
                        #         chunks=True,    
                        #     )
                        # else:
                        #     features_dataset.resize(features_dataset.shape[0] + 1, axis=0)
                        #     features_dataset[-1] = output_features

def resample_transcript(transcript: pd.DataFrame, new_interval: float) -> pd.DataFrame:
    """
    Pre-aggregates transcript data into new time intervals.

    Parameters:
        transcript (pd.DataFrame): DataFrame with columns 'words_per_tr', 'onsets_per_tr', and 'durations_per_tr'.
        new_interval (float): Desired interval in seconds for grouping.

    Returns:
        pd.DataFrame: New DataFrame where each row aggregates words whose end time (onset + duration)
                      falls within the same new interval.
    """
    all_words = []
    all_onsets = []
    all_durations = []

    for _, row in transcript.iterrows():
        # Skip rows without valid onsets.
        if not row['onsets_per_tr'] or row['onsets_per_tr'] == []:
            continue

        # Convert string representations if needed.
        onsets = row['onsets_per_tr']
        words = row['words_per_tr']
        durations = row['durations_per_tr']
        if isinstance(onsets, str):
            onsets = ast.literal_eval(onsets)
        if isinstance(words, str):
            words = ast.literal_eval(words)
        if isinstance(durations, str):
            durations = ast.literal_eval(durations)

        all_words.extend(words)
        all_onsets.extend(onsets)
        all_durations.extend(durations)

    # Create a DataFrame with one row per word.
    df = pd.DataFrame({
        'word': all_words,
        'onset': all_onsets,
        'duration': all_durations
    })
    df['word_end'] = df['onset'] + df['duration']

    # Determine the new interval index.
    df['new_index'] = (df['word_end'] // new_interval).astype(int)

    # Group by the new interval index.
    grouped = df.groupby('new_index').agg({
        'word': list,
        'onset': list,
        'duration': list,
        'word_end': list
    }).reset_index(drop=True)

    return grouped


def extract_section(
    video: torch.Tensor,
    audio: torch.Tensor,
    transcript: pd.DataFrame,
    interval: float,
    index: int,
    sample_rate: int,
    modality: str = 'all',
    fps_video: float = 30
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
    transcript_section = []

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
        # fps_video = 30 # 29.97
        frame_start = round(start_time * fps_video)
        frame_end = round(end_time * fps_video)
        video_section = video[frame_start:frame_end]

    if modality == 'all' or modality == 'transcript':
        transcript_section = transcript['word'].iloc[index]

        print(transcript_section)

    return video_section, audio_section,  transcript_section



# ### Load the model and experiment to see it's inputs and layers

# In[ ]:


import torch
from torchcodec.decoders import VideoDecoder
import numpy as np
from transformers import AutoVideoProcessor, AutoModel
device = torch.device("cuda:1")
hf_repo = "facebook/vjepa2-vitg-fpc64-256"
cache_dir = "/home/mihir/projects/hf_cache"
os.makedirs(cache_dir, exist_ok=True)
model = AutoModel.from_pretrained(hf_repo, cache_dir=cache_dir).to(device).eval()
processor = AutoVideoProcessor.from_pretrained(hf_repo, cache_dir=cache_dir)

print(model)

# In[ ]:


layers_to_extract = layers_to_extract = [
    "encoder.layer.15.norm1",
    "encoder.layer.15.norm2",
    "encoder.layer.15.mlp.fc2",
    "encoder.layer.25.norm1",
    "encoder.layer.25.norm2",
    "encoder.layer.25.mlp.fc2",
    "encoder.layer.35.norm1",
    "encoder.layer.35.norm2",
    "encoder.layer.35.mlp.fc2",
    "encoder.layer.5.norm1",
    "encoder.layer.5.norm2",
    "encoder.layer.5.mlp.fc2",
    "encoder.layernorm"
]


# In[12]:


def select_frames(video_section, target_frames=16):
    num_frames = video_section.shape[0]
    if num_frames >= target_frames:
        # Uniformly sample 16 indices from 0 to num_frames - 1
        indices = torch.linspace(0, num_frames - 1, steps=target_frames).long()
        selected_frames = video_section[indices]
    else:
        # Repeat frames to reach exactly 16 frames
        repeats = target_frames // num_frames
        remainder = target_frames % num_frames
        repeated_frames = video_section.repeat(repeats, 1, 1, 1)
        if remainder > 0:
            extra_frames = video_section[:remainder]
            repeated_frames = torch.cat([repeated_frames, extra_frames], dim=0)
        selected_frames = repeated_frames
    return selected_frames


# In[13]:


import math
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms import InterpolationMode
from transformers import AutoConfig, AutoModel, AutoTokenizer
from typing import Dict, List, Tuple, Union
from pathlib import Path
from torchvision.transforms.functional import to_pil_image

# random_reduction
sparse_random_projection = False
eps_reduction = 0.1

from sklearn.random_projection import johnson_lindenstrauss_min_dim
import numpy as np
from sklearn.random_projection import SparseRandomProjection
from einops import rearrange

if sparse_random_projection:
    rng = np.random.RandomState(42)

    def get_min_dims():
        num_seasons = 7
        average_episodes_per_season = 45
        average_TRs_per_episode = 470
        min_dim = johnson_lindenstrauss_min_dim(n_samples=num_seasons * average_episodes_per_season * average_TRs_per_episode, eps=eps_reduction)
        print(min_dim)  # Should print around 1000–1200
        return min_dim

    def create_transformations(features):
        all_transformator = {}
        min_dim = get_min_dims()

        for layer_name, feature in features.items():
            feature = rearrange(feature, 'b t d -> b (t d)').to('cpu')
            print('Creating projection for layer:', layer_name, 'with shape:', feature.shape)
            if feature.shape[1] > min_dim:
                transformer = SparseRandomProjection(n_components=min_dim, random_state=rng)
                transformer = transformer.fit(feature)
            else:
                raise ValueError("Feature dimension is too low for transformation.")

            all_transformator[layer_name] = transformer

        return all_transformator

    def apply_transformations(features, all_transformator):
        transformed_features = {}
        for layer_name, feature in features.items():
            feature = rearrange(feature, 'b t d -> b (t d)').to('cpu')
            if layer_name in all_transformator:
                transformer = all_transformator[layer_name]
                transformed_feature = transformer.transform(feature)
                transformed_features[layer_name] = torch.tensor(transformed_feature, dtype=torch.float16)
            else:
                transformed_features[layer_name] = feature
        return transformed_features

    all_sparse_transformator = create_transformations(features)


# ### Define the extract_fn function

# In[14]:


from functools import wraps

# Define the extractor.
extractor = HuggingFaceFeatureExtractor(model, layers_to_extract, detach=True, call_fn="get_vision_features")
num_frames_to_extract = 15

def extract_fn(
    video: torch.Tensor, 
    audio: torch.Tensor, 
    transcript: List[str], 
    verbose: bool
) -> Dict[str, torch.Tensor]:
    # Modify this function using the feature extractor
    # video is a tensor with shape [fps * interval, 3, heigth, width] on fp16 from 0-255
    # audio is a tensor with shape [1 if mono 2 if stereo, sampling_rate * interval] on fp16
    # transcript is list of strings of words.

    with torch.no_grad():                   
        if video.shape[0] < 1:
            print("Alert, one of the intervals is empty.")
            return{
                layer: torch.zeros_like(features_sample[layer]) for layer in layers_to_extract
            }
        # if video.shape[0] < num_frames_to_extract:
        #     last = video[-1]                    # shape (H,W)
        #     pad = last.unsqueeze(0)         # shape (1,H,W)
        #     pad = pad.repeat(num_frames_to_extract - video.size(0), 1, 1) 
        #     video = torch.cat([video, pad], dim=0)

        if video.shape[0] < num_frames_to_extract:
            video = video.to(torch.uint8)
        else:
            video = select_frames(video, num_frames_to_extract).to(torch.uint8)

        video_processed = processor(video, return_tensors="pt").to(model.device)
        video_embeddings = extractor(**video_processed)

        # with torch.no_grad():
        #     _ = extractor(full_batch.to('cuda'))

        dict_return = {}

        # for layer_name, activation in extractor.features.items():
        #     # Taking the mean bcz the features are so large
        #     # dict_return[layer_name] = activation.mean(1)
        #     if activation.dim() > 2:
        #         dict_return[layer_name + 'gp'] = activation.max(dim=1)[0].unsqueeze(0) # save both global pooling 
        #         dict_return[layer_name + 'avg'] = activation.mean(dim=1).unsqueeze(0) # and save average
        #     else: 
        #         dict_return[layer_name] = activation.unsqueeze(0)

        # extractor.clear()

        features = extractor.features


        if sparse_random_projection:
            # Apply the sparse random projection transformation.
            proj_features = apply_transformations(features, all_sparse_transformator)

        # Iterate over the extracted features and print their shapes.
        for layer_name, activation in features.items():
            avg_activation = activation.mean(dim=1)
            dict_return[layer_name + '_avg'] = avg_activation.to(torch.float16).to('cpu')
            # print(f"Layer: {layer_name}, Feature shape: {activation.shape}, Averaged feature shape: {avg_activation.shape}, Sample: {avg_activation[0,0:5]}")

        if sparse_random_projection:
            for layer_name, activation in proj_features.items():
                dict_return[layer_name + '_srp'] = activation.to(torch.float16).to('cpu')
                # print("Layer:", layer_name, "with shape:", activation.shape, "and sample:", activation[0,0:5])

        # print(f"Extracted features: {list(dict_return.keys())}")
        extractor.clear()
        return dict_return
        # Concatenate features from all chunks for each layer.
        # dict_return = {}
        # for layer, act_list in features_accum.items():
        #     # Each activation has shape [1, chunk_size, ...]; concatenating along the frame dimension (dim=1).
        #     concatenated = torch.cat(act_list, dim=1)
        #     # Taking the mean over the temporal dimension (adjust as needed)
        #     dict_return[layer] = concatenated

        # return dict_return


# ### Start extracting features

# In[ ]:


# Example usage:
parts = ['chaplin', 'mononoke', 'passepartout', 'planetearth', 'wot', 'pulpfiction']
movies_base = "/data/algonauts_2025.competitors/stimuli/movies"
transcripts_base = "/data/algonauts_2025.competitors/stimuli/transcripts"
out_dir = '/home/mihir/projects/datasets/"VJEPA2_OOD'


extract_features(parts = parts, movies_base = movies_base, transcripts_base = transcripts_base, output_dir = out_dir, extraction_fn = extract_fn, verbose = True, modality = 'video', ood=True)

