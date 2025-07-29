import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
from decord import VideoReader, cpu


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


# In[5]:


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


def load_video(path, resolution=(224, 224), dtype=torch.float16, verbose: bool = True):
    vr   = VideoReader(str(path), ctx=cpu(0))              # multi-thread FFmpeg
    fps  = vr.get_avg_fps()
    frames = vr.get_batch(range(len(vr)))             # NDArray, [T,H0,W0,3]
    frames = torch.from_numpy(frames.asnumpy())       \
                 .permute(0,3,1,2)                    # to [T,3,H0,W0]
    print("Frames dtype before interpolation:", frames.dtype)
    frames = F.interpolate(frames,
                           size=resolution, mode="bilinear",
                           align_corners=False)
    frames = frames.to(dtype=dtype)  # cast to float16
    return frames, fps



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
    past_context_in_seconds: int = 30,
    splits_overlap: float = 0.5,
    ignore_done = None, 
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

    done_ = []

    # curret = 0
    # start_ = 35
    # max_ = 100

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
                    #     print("yes digit")
                    # print(ood_suffix)
                    transcript_file = transcripts_base / rel_folder / f"ood_{ood_suffix}"
                else:
                    transcript_file = transcripts_base / rel_folder / f"movie10_{movie_file.with_suffix('.tsv').name}"

                print(f"Movie:      {movie_file}")
                print(f"Transcript: {transcript_file}")

                if str(movie_file).split('/')[-1].split('.')[0] + '.h5' in ignore_done:
                    continue

                # if curret < start_:
                #     curret = curret + 1
                #     continue
                # if curret >= max_:
                #     break

                # curret = curret + 1

                # Load video frames, audio waveform, and transcript.
                video, audio, transcript, sample_rate, fps_video = None, None, None, None, None
                if modality == 'all' or modality == 'video':
                    video, fps_video = load_video(movie_file, verbose=verbose, resolution = (256, 384))
                if modality == 'all' or modality == 'audio' or modality == 'video' or modality == 'transcript':
                    audio, sample_rate = load_audio(movie_file, sampling_rate = 16000, stereo=False)
                if (modality == 'all' or modality == 'transcript' or modality == 'video') and transcript_file.exists():
                    transcript = load_transcript(transcript_file)
                    if transcript is not None:
                        transcript = resample_transcript(transcript, interval)
                else:
                    transcript = None
                    print(f"Transcript file not found or not requested for {movie_file.name}, proceeding without it.")
                # transcript=None

                total_duration = video.shape[0] / fps_video 
                # total_duration = audio.shape[1] / sample_rate
                num_intervals_tr = int(total_duration // interval)

                if verbose:
                    print(f"Total duration: {total_duration:.2f} seconds")
                    print(f"Number of intervals: {num_intervals_tr}")
                    print(f"Sample rate: {sample_rate}")

                # Create the output directory if it doesn't exist.
                output_folder = Path(output_dir) / rel_folder
                output_folder.mkdir(parents=True, exist_ok=True)

                # Create a h5 file to store the features.
                output_file = output_folder / movie_file.with_suffix('.h5').name
                done_.append(movie_file.with_suffix('.h5').name)
                print("Doneee: \n", done_)

                if verbose:
                    print(f"Output file: {output_file}")



                total_trs = math.ceil(total_duration / interval)
                context_in_TRs = math.ceil(past_context_in_seconds / interval)
                overlap_in_TRs = math.ceil(splits_overlap * context_in_TRs)
                real_increment_in_TRs = context_in_TRs - overlap_in_TRs

                total_iterations = math.ceil((total_trs - context_in_TRs) / real_increment_in_TRs) + 1

                if verbose:
                    print(f"Total TRs for this clip: {total_trs}")
                    print(f"Context in TRs: {context_in_TRs}")
                    print(f"Overlap in TRs: {overlap_in_TRs}")
                    print(f"Real increment in TRs: {real_increment_in_TRs}")
                    print(f"Total iterations per clip: {total_iterations}")



                # Create a HDF5 file to store the features.
                with h5py.File(output_file, 'w') as f:
                    features_datasets = {} 
                    for i in tqdm(range(total_iterations)):
                        start_index = i * real_increment_in_TRs
                        end_index = start_index + context_in_TRs

                        if start_index >= total_trs:
                            print(f"Skipping iteration {i} as start_index {start_index} is beyond total TRs {total_trs}. This should not happen.")
                            break

                        # compute future_offset safely
                        future_offset = min(end_index, total_trs) - start_index - 1

                        # print("First ", index, future_offset)
                        video_section, audio_section, transcript_section = extract_section(
                            video, audio, transcript, interval, start_index, sample_rate, modality, fps_video, past_offset = 0, future_offset = future_offset, split_by_tr = True
                        )

                        # print(video_section.shape)
                        # output_features = {}
                        # layer_names = ['abc', 'def']  # Example layer names, replace with actual layer names from your model.
                        # for layer in layer_names:
                        #     output_features[layer] = torch.zeros(video_section.shape[0], 512) + i


                        # plot first and last frame of the video section
                        # if video_section.shape[0] > 0:
                        #     first_frame = video_section[0][0].permute(1, 2, 0).cpu().numpy()/256
                        #     last_frame = video_section[-1][-1].permute(1, 2, 0).cpu().numpy()/256
                        #     plt.figure(figsize=(10, 5))
                        #     plt.subplot(1, 2, 1)
                        #     plt.imshow(first_frame)
                        #     plt.title("First Frame")
                        #     plt.axis('off')
                        #     plt.subplot(1, 2, 2)
                        #     plt.imshow(last_frame)
                        #     plt.title("Last Frame")
                        #     plt.axis('off')
                        #     plt.show()

                        output_features = extraction_fn(video_section, audio_section, transcript_section, verbose, clip_fps = fps_video)

                        for layer_name, tensor in output_features.items():
                            assert tensor.shape[0] == video_section.shape[0], f"Error on layer: {layer_name}, the number of TRs of the output features should be the same as the number of TRs of the video section. Got {tensor.shape[0]} and {video_section.shape[0]}"

                        for layer_name, tensor in output_features.items():
                            # Convert the tensor to a numpy array (on CPU) before storing.
                            tensor_np = tensor.cpu().numpy() # shape [batch_size, feature_dim1, feature_dim2, ...]
                            batch_size = tensor_np.shape[0]
                            if layer_name not in features_datasets:
                                # Create a new dataset and initialize it with the first interval's data.
                                features_datasets[layer_name] = f.create_dataset(
                                    layer_name,
                                    # data=tensor_np[np.newaxis, ...],
                                    data=tensor_np,
                                    maxshape=(None,) + tensor_np.shape[1::],
                                    dtype=np.float16,
                                    chunks=True,
                                )
                                # print("Max_shape: ", features_datasets[layer_name].maxshape)
                                # print("Shape: ", features_datasets[layer_name].shape)
                            else:
                                ds = features_datasets[layer_name]
                                # ds.resize(ds.shape[0] + 1, axis=0)
                                last_shape = ds.shape[0]
                                # print("Resizing dataset from shape: ", ds.shape, " to ", min(end_index, total_trs))
                                t_end_index = min(end_index, total_trs)
                                ds.resize(t_end_index, axis=0)
                                # ds[-1] = tensor_np
                                ds[last_shape:t_end_index] = tensor_np[-(t_end_index - last_shape)::]
                                # print("Max_shape: ", features_datasets[layer_name].maxshape)
                                # print("Shape: ", features_datasets[layer_name].shape)


def resample_transcript(transcript: pd.DataFrame, new_interval: float) -> pd.DataFrame:
    """
    Pre-aggregates transcript data into new time intervals.

    Parameters:
        transcript (pd.DataFrame): DataFrame with columns 'words_per_tr', 'onsets_per_tr', and 'durations_per_tr'.
        new_interval (float): Desired interval in seconds for grouping.

    Returns:
        pd.DataFrame: New DataFrame where each row aggregates words whose end time (onset + duration)
                      falls within the same new interval. Intervals with no transcript words
                      are represented with empty text and empty arrays.
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

    # Determine the new interval index for each word (based on word_end)
    df['new_index'] = (df['word_end'] // new_interval).astype(int)

    # Group by the new interval index.
    grouped = df.groupby('new_index').agg({
        'word': list,
        'onset': list,
        'duration': list,
        'word_end': list
    }).reset_index()

    # Ensure max_index is an integer. If df is empty, set max_index to 0.
    max_index = df['new_index'].max()
    if pd.isna(max_index):
        max_index = 0
    else:
        max_index = int(max_index)

    # Create a complete DataFrame with all interval indices from 0 up to the maximum.
    complete_intervals = pd.DataFrame({'new_index': range(max_index + 1)})

    # Merge the complete intervals with the grouped data so that empty intervals are kept.
    result = complete_intervals.merge(grouped, on='new_index', how='left')

    # Replace any missing values with empty lists.
    for col in ['word', 'onset', 'duration', 'word_end']:
        result[col] = result[col].apply(lambda x: x if isinstance(x, list) else [])

    # (Optional) Create a text column that joins the words, resulting in an empty string for empty intervals.
    result['text'] = result['word'].apply(lambda x: ' '.join(x))

    return result


import math
import torch
import pandas as pd
from typing import Tuple, List
import einops

def extract_section(
    video: torch.Tensor,
    audio: torch.Tensor,
    transcript: pd.DataFrame,
    interval: float,
    index: int,
    sample_rate: int,
    modality: str = 'all',
    fps_video: float = 30,
    past_offset: int = 0,   # number of intervals (including current) to include from the past
    future_offset: int = 0,  # number of intervals after the current one to include
    split_by_tr: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Extracts a section of audio, video, and transcript data based on the interval and index,
    with optional offsets for past and future intervals. If part of the requested window is
    out of bounds, the missing parts are padded with zeros (or empty strings for transcript).

    Parameters:
        video (torch.Tensor): Tensor containing video frames (num_frames, 3, 224, 224).
        audio (torch.Tensor): Tensor containing audio waveform (channels, num_samples).
        transcript (pd.DataFrame): DataFrame containing transcript data (assumed one row per interval).
        interval (float): Duration (in seconds) of one segment/interval.
        index (int): Index (zero-indexed) of the current interval.
        sample_rate (int): Sample rate of the audio waveform.
        modality (str): Modality to extract features from. Options are 'video', 'audio', 'transcript', or 'all'.
        fps_video (float): Frames per second of the video.
        past_offset (int): Number of intervals to include from the past (including the current one).
            For example, past_offset=5 with index=100 returns intervals 96-100.
        future_offset (int): Number of intervals after the current one to include.
            For example, future_offset=2 with index=100 returns intervals 100-102.

    Returns:
        tuple: (video_section, audio_section, transcript_section) where:
            - video_section is a torch.Tensor of shape (requested_frames, *video.shape[1:]),
              padded with zeros if necessary.
            - audio_section is a torch.Tensor of shape (channels, requested_samples),
              padded with zeros if necessary.
            - transcript_section is a list of strings of length (number of requested intervals),
              where missing intervals are filled with empty strings.
    """
    # Determine the range of intervals to extract.
    # If past_offset > 0, we include the current interval and the (past_offset - 1) preceding intervals.
    extraction_start_index = index - past_offset + 1 if past_offset > 0 else index
    extraction_end_index = index + future_offset  # inclusive
    total_intervals = extraction_end_index - extraction_start_index + 1

    # Determine the corresponding time boundaries.
    # Note that a given interval i spans [i * interval, (i+1) * interval).
    requested_start_time = extraction_start_index * interval
    requested_end_time = (extraction_end_index + 1) * interval  # exclusive end

    # ---- Audio Extraction ----
    audio_section = None
    if modality in ['all', 'audio']:
        # Total samples requested
        total_requested_samples = int(round(total_intervals * interval * sample_rate))
        # Create output tensor filled with zeros.
        audio_section = torch.zeros(audio.shape[0], total_requested_samples)

        # Compute the global sample indices corresponding to the requested time window.
        requested_start_sample = int(round(requested_start_time * sample_rate))
        requested_end_sample = int(round(requested_end_time * sample_rate))

        # Determine the part available from the source audio.
        source_start = max(0, requested_start_sample)
        source_end = min(audio.shape[1], requested_end_sample)

        # Determine where to paste the available audio in the output tensor.
        target_offset = 0
        if requested_start_sample < 0:
            target_offset = -requested_start_sample  # number of samples to pad at beginning

        # Compute the number of samples to copy.
        num_samples_to_copy = source_end - source_start
        if num_samples_to_copy > 0:
            audio_section[:, target_offset:target_offset + num_samples_to_copy] = audio[:, source_start:source_end]
        if split_by_tr:
            audio_section = einops.rearrange(audio_section, 'c (tr t) -> tr c t', tr=total_intervals)

    # ---- Video Extraction ----
    video_section = None
    if modality in ['all', 'video']:
        # Compute requested frame indices.
        requested_video_start = int(round(requested_start_time * fps_video))
        requested_video_end = int(round(requested_end_time * fps_video))  # exclusive end
        total_requested_frames = requested_video_end - requested_video_start

        # Create output tensor filled with zeros.
        video_section = torch.zeros(total_requested_frames, *video.shape[1:])

        # Determine the available frames.
        source_frame_start = max(0, requested_video_start)
        source_frame_end = min(video.shape[0], requested_video_end)

        # Determine target offset in frames.
        target_offset_frames = 0
        if requested_video_start < 0:
            target_offset_frames = -requested_video_start

        num_frames_to_copy = source_frame_end - source_frame_start
        if num_frames_to_copy > 0:
            video_section[target_offset_frames:target_offset_frames + num_frames_to_copy] = video[source_frame_start:source_frame_end]
        if split_by_tr:
            B, C, H, W = video_section.shape
            tr = total_intervals            # 137
            f  = B // tr                    # 35
            new_len = f * tr                # 35 * 137 = 4795

            # pick new_len indices that span 0 … B-1 evenly
            # torch.linspace gives floats; .round().long() makes them integer
            inds = torch.linspace(0, B-1, steps=new_len).round().long()

            # gather those frames
            vs = video_section[inds]

            # reshape into (35, 3, 137, H, W)
            video_section = einops.rearrange(vs, '(tr f) c w h -> tr f c w h', tr=tr)

    # ---- Transcript Extraction ----
    transcript_section = []
    # --- FIX ---
    # Check if transcript is not None before trying to access it.
    if modality in ['all', 'transcript'] and transcript is not None:
        for i in range(total_intervals):
            global_idx = extraction_start_index + i
            if global_idx < 0 or global_idx >= len(transcript):
                # Append an empty list for intervals with no words, matching the expected type.
                transcript_section.append([])
            else:
                # Use .append() for efficiency and clarity.
                transcript_section.append(transcript['word'].iloc[global_idx])
    # If transcript is None, transcript_section remains an empty list, which is a valid iterable.

    return video_section, audio_section, transcript_section


import torch
import soundfile as sf
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniThinkerForConditionalGeneration
from qwen_omni_utils import process_mm_info
path = "Qwen/Qwen2.5-Omni-7B"
cache_dir = "/home/mihir/projects/hf_cache"
os.makedirs(cache_dir, exist_ok=True)
model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    # enable_audio_output=True,
    attn_implementation="sdpa",
    cache_dir=cache_dir
).eval()
device = torch.device("cuda")
model = model.to(device)
print(model)
# model = torch.compile(model, mode="reduce-overhead")

processor = Qwen2_5OmniProcessor.from_pretrained(path, cache_dir = cache_dir)


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

USE_AUDIO_IN_VIDEO = True
processor.max_pixels = 128 * 28 * 28


def build_inputs_and_intervals(
    frames: torch.Tensor,
    *,
    processor,
    audio: torch.Tensor,
    model_device,
    use_audio_in_video: bool = True,
    model_dtype = torch.float16,
    clips_fps = int
) -> Tuple[dict, List[Tuple[int, int]]]:
    """
    Parameters
    ----------
    frames : torch.Tensor
        Shape (N, T, C, H, W) **or** (N, 2, H, W) – N clips stacked on the first axis.
        If you already have a python `List[List[torch.Tensor]]` of clips you can pass that
        instead and skip the `frames[i]` → `clip_frames` conversion below.
    processor : transformers.QwenProcessor (or compatible)
    audio : torch.Tensor with shape (N, sr)
        1‑D mono waveform already at `processor.sampling_rate`.
    use_audio_in_video : bool
        Whether to fuse the audio into the video tokens (matches your  `USE_AUDIO_IN_VIDEO` flag).
    model_device, model_dtype
        Forwarded to `.to(...)` just like in your snippet.

    Returns
    -------
    inputs  : dict
        The same structure that `processor(...)` returns, moved to the requested
        device/dtype – ready to feed into the model.
    intervals : List[Tuple[int, int]]
        One `(start, end)` pair **per clip**, inclusive, in the order they appeared
        in `frames`.
    """
    FPS_qwen = 2
    N = len(frames)
    video_chunks = ["<|vision_bos|><|VIDEO|><|vision_eos|>" for _ in range(N)]
    text_input = ["".join(video_chunks)]         # processor expects a list of strings

    #     to the distinct <|VIDEO|> placeholders
    videos = [select_frames(frames[i], target_frames=round(frames[i].shape[0] // clips_fps * FPS_qwen)) for i in range(N)]

    # print(text_input, videos[0].shape, videos[1].shape)
    inputs = processor(
        text=text_input,
        audio=[tuple(audio_i) for audio_i in audio],   # keep the outer list so batch size == 1
        images=None,
        videos=videos,
        return_tensors="pt",
        padding=False,
        use_audio_in_video=use_audio_in_video,
    ).to(model_device).to(model_dtype)

    # 4️⃣  Locate the token spans for each clip
    ids = inputs["input_ids"][0].tolist()                  # flatten to python list
    bos_id = processor.tokenizer.convert_tokens_to_ids("<|vision_bos|>")
    eos_id = processor.tokenizer.convert_tokens_to_ids("<|vision_eos|>")

    intervals: List[Tuple[int, int]] = []
    active_start = None
    for idx, tok in enumerate(ids):
        if tok == bos_id and active_start is None:
            active_start = idx
        elif tok == eos_id and active_start is not None:
            intervals.append((active_start, idx))          # inclusive interval
            active_start = None

    assert len(intervals) == N, "Failed to find all clip spans – tokenisation drifted?"

    return inputs, intervals


# In[31]:


import math
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms import InterpolationMode
from transformers import AutoConfig, AutoModel, AutoTokenizer
from typing import Dict, List, Tuple, Union
from pathlib import Path
from torchvision.transforms.functional import to_pil_image


# In[ ]:


layers_to_extract = layers_to_extract = [
    # "model.layers.10.post_attention_layernorm",
    # "model.layers.25.post_attention_layernorm",
    "model.layers.15.post_attention_layernorm",
    "model.layers.20.post_attention_layernorm",
    # # "model.layers.25.mlp.down_proj",
    # # "model.layers.10.mlp.down_proj",
    # # "model.layers.15.mlp.down_proj",
    # "model.layers.20.mlp.down_proj",
    # "model.norm"
]


# ### Write the extract_fn

# In[34]:


from functools import wraps

# Define the extractor.
extractor = HuggingFaceFeatureExtractor(model, layers_to_extract, detach=True)


# @torch.compile
def extract_fn(
    video: torch.Tensor, 
    audio: torch.Tensor, 
    transcript: List[List[str]], 
    verbose: bool,
    clip_fps: None
) -> Dict[str, torch.Tensor]:
    # Modify this function using the feature extractor
    # video is a tensor with shape [fps * interval, 3, heigth, width] on fp16 from 0-255
    # audio is a tensor with shape [1 if mono 2 if stereo, sampling_rate * interval] on fp16
    # transcript is list of strings of words.

    dict_return = {}
    with torch.no_grad():
        print(len([vid_i for vid_i in video]), [vid_i for vid_i in video][0].shape, len([audio_i[0] for audio_i in audio]), [audio_i[0] for audio_i in audio][0].shape, audio.shape)
        inputs, intervals = build_inputs_and_intervals(
            [vid_i for vid_i in video],
            processor=processor,
            audio=[audio_i[0] for audio_i in audio],
            use_audio_in_video=True,
            model_device=model.device,
            model_dtype=model.dtype,
            clips_fps=clip_fps
        )
        outputs = extractor(**inputs, use_audio_in_video = True)
        print(outputs['logits'].shape)        # [1, seq_len, 151674]
        print(intervals)              # e.g. [(0,160), (161,280), ...]

        features = extractor.features

        # Iterate over the extracted features and print their shapes.
        for layer_name, activation in features.items():
            avg_activation = torch.stack([
                activation[0, s:e+1].mean(dim=0)              # mean over sequence dimension
                for (s, e) in intervals
            ], dim=0)  
            # print(f"Layer: {layer_name}, Feature shape: {activation.shape}, Averaged feature shape: {avg_activation.shape}, Sample: {avg_activation[0,0:5]}")
            dict_return[layer_name] = avg_activation.to(torch.float16).cpu()

        # pixel_values = video[:,0] # select the first frame of each chunk
        # pixel_values = load_image(pixel_values).to(torch.bfloat16).cuda()
        # sentences = transcript

        # logits, ranges, avg_logits = logits_by_pair(
        #     extractor, tokenizer,
        #     pixel_values, sentences,
        #     use_template=False   # or True
        # )

        # # print(logits.shape)        # [1, seq_len, 151674]
        # # print(ranges)              # e.g. [(0,160), (161,280), ...]
        # # print(avg_logits.shape)    # [n, 151674]

        # features = extractor.features

        # # Iterate over the extracted features and print their shapes.
        # for layer_name, activation in features.items():
        #     avg_activation = torch.stack([
        #         activation[0, s:e+1].mean(dim=0)              # mean over sequence dimension
        #         for (s, e) in ranges
        #     ], dim=0)  
        #     print(f"Layer: {layer_name}, Feature shape: {activation.shape}, Averaged feature shape: {avg_activation.shape}, Sample: {avg_activation[0,0:5]}")
        #     dict_return[layer_name] = avg_activation.to(torch.float16).cpu()
    return dict_return


# Example usage:
# parts = ['chaplin', 'mononoke', 'passepartout', 'planetearth', 'wot', 'pulpfiction']
parts = ['s2', 's3', 's4', 's5','wolf', 'bourne']
movies_base = "/data/algonauts_2025.competitors/stimuli/movies"
transcripts_base = "/data/algonauts_2025.competitors/stimuli/transcripts"
out_dir = '/home/mihir/projects/datasets/qwen_7B_30sec'
ignore_done = []




extract_features(parts = parts, movies_base = movies_base, transcripts_base = transcripts_base, output_dir = out_dir, extraction_fn = extract_fn, verbose = True, modality = 'all', past_context_in_seconds = 30, splits_overlap=0.5, ignore_done = ignore_done, ood=False)