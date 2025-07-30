import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from decord import VideoReader, cpu
import argparse
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import fnmatch
from typing import Any, Dict, List, Tuple, Callable, Union
from torch import nn
from PIL import Image
import ast 
import torchaudio
import cv2
import torch
import torch.nn.functional as F
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import einops
import torchvision.transforms as T
from transformers import AutoConfig, AutoModel, AutoTokenizer
import dataclasses
from enum import IntEnum, auto

ROOT = Path(__file__).parent.parent
DEFAULT_DATA_DIR = ROOT / "datasets"
DEFAULT_CONFIG = ROOT / "config/default_internvl3_features.yaml"



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

    # Iterate through all directories under movies_base.
    # print("starting")
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

                # print(rel_folder)
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
                    # import sys; sys.exit()
                    transcript_file = transcripts_base / rel_folder / f"ood_{ood_suffix}"
                else:
                    transcript_file = transcripts_base / rel_folder / f"movie10_{movie_file.with_suffix('.tsv').name}"

                print(f"Movie:      {movie_file}")
                print(f"Transcript: {transcript_file}")

                if str(movie_file).split('/')[-1].split('.')[0] + '.h5' in ignore_done:
                    continue

                # Load video frames, audio waveform, and transcript.
                video, audio, transcript, sample_rate, fps_video = None, None, None, None, None
                if modality == 'all' or modality == 'video':
                    video, fps_video = load_video(movie_file, verbose=verbose)
                if modality == 'all' or modality == 'audio' or modality == 'video' or modality == 'transcript':
                    audio, sample_rate = load_audio(movie_file)

                # Only load and process the transcript if the file exists.
                if (modality == 'all' or modality == 'transcript') and transcript_file.exists():
                    transcript = load_transcript(transcript_file)
                    if transcript is not None:
                        transcript = resample_transcript(transcript, interval)
                else:
                    transcript = None
                    print(f"Transcript file not found or not requested for {movie_file.name}, proceeding without it.")
                # --- END CORRECTION ---

                total_duration = audio.shape[1] / sample_rate
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

                if verbose:
                    print(f"Output file: {output_file}")

                seconds_duration = int(audio.shape[1] / sample_rate)
                num_splits = max(1, int(seconds_duration / past_context_in_seconds))
                print(f"Num splits: {num_splits}")

                total_iterations = math.ceil((num_splits / (1 - splits_overlap)) - 1)
                # Create a HDF5 file to store the features.
                with h5py.File(output_file, 'w') as f:
                    features_datasets = {} 
                    # Extract features at each interval.
                    fixed_distance_interval = math.ceil(num_intervals_tr / num_splits)
                    for i in tqdm(range(total_iterations)):
                        index = math.ceil((i * fixed_distance_interval) - (i * splits_overlap * fixed_distance_interval))

                        if index >= num_intervals_tr:        # ← guard clause
                            break

                        # compute future_offset safely
                        future_offset = min(fixed_distance_interval - 1,
                                            num_intervals_tr - index - 1)

                        end_index = index + future_offset

                        # print("First ", index, future_offset)
                        video_section, audio_section, transcript_section = extract_section(
                            video, audio, transcript, interval, index, sample_rate, modality, fps_video, past_offset = 0, future_offset = future_offset, split_by_tr = True
                        )

                        output_features = extraction_fn(video_section, audio_section, transcript_section, verbose)

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
                            else:
                                ds = features_datasets[layer_name]
                                # ds.resize(ds.shape[0] + 1, axis=0)
                                last_shape = ds.shape[0]
                                ds.resize(end_index + 1, axis=0)
                                # ds[-1] = tensor_np
                                ds[last_shape:end_index] = tensor_np[-(end_index - last_shape)::]


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



def extract_section(
    video: torch.Tensor,
    audio: torch.Tensor,
    transcript: pd.DataFrame,
    interval: float,
    index: int,
    sample_rate: int,
    modality: str = 'all',
    fps_video: float = 30,
    past_offset: int = 0,
    future_offset: int = 0,
    split_by_tr: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, List[List[str]]]: # Corrected return type annotation
    """
    Extracts a section of audio, video, and transcript data.
    This version safely handles cases where the transcript is None.
    """
    # Determine the range of intervals to extract.
    extraction_start_index = index - past_offset + 1 if past_offset > 0 else index
    extraction_end_index = index + future_offset
    total_intervals = extraction_end_index - extraction_start_index + 1

    requested_start_time = extraction_start_index * interval
    requested_end_time = (extraction_end_index + 1) * interval

    # ---- Audio Extraction ----
    audio_section = None
    if modality in ['all', 'audio'] and audio is not None:
        total_requested_samples = int(round(total_intervals * interval * sample_rate))
        audio_section = torch.zeros(audio.shape[0], total_requested_samples)
        requested_start_sample = int(round(requested_start_time * sample_rate))
        requested_end_sample = int(round(requested_end_time * sample_rate))
        source_start = max(0, requested_start_sample)
        source_end = min(audio.shape[1], requested_end_sample)
        target_offset = -requested_start_sample if requested_start_sample < 0 else 0
        num_samples_to_copy = source_end - source_start
        if num_samples_to_copy > 0:
            audio_section[:, target_offset:target_offset + num_samples_to_copy] = audio[:, source_start:source_end]
        if split_by_tr:
            audio_section = einops.rearrange(audio_section, 'c (tr t) -> tr c t', tr=total_intervals)

    # ---- Video Extraction ----
    video_section = None
    if modality in ['all', 'video'] and video is not None:
        requested_video_start = int(round(requested_start_time * fps_video))
        requested_video_end = int(round(requested_end_time * fps_video))
        total_requested_frames = requested_video_end - requested_video_start
        video_section = torch.zeros(total_requested_frames, *video.shape[1:])
        source_frame_start = max(0, requested_video_start)
        source_frame_end = min(video.shape[0], requested_video_end)
        target_offset_frames = -requested_video_start if requested_video_start < 0 else 0
        num_frames_to_copy = source_frame_end - source_frame_start
        if num_frames_to_copy > 0:
            video_section[target_offset_frames:target_offset_frames + num_frames_to_copy] = video[source_frame_start:source_frame_end]
        if split_by_tr:
            B, C, H, W = video_section.shape
            tr = total_intervals
            f = B // tr
            new_len = f * tr
            inds = torch.linspace(0, B - 1, steps=new_len).round().long()
            vs = video_section[inds]
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




IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def find_closest_aspect_ratio(ar, target_ratios, width, height, image_size):
    best_diff = float('inf')
    best = (1, 1)
    area = width * height
    for (w, h) in target_ratios:
        target_ar = w / h
        diff = abs(ar - target_ar)
        # pick the ratio with smallest diff; on tie prefer larger original image area
        if diff < best_diff or (diff == best_diff and area > 0.5 * image_size**2 * w * h):
            best_diff = diff
            best = (w, h)
    return best

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_w, orig_h = image.size
    ar = orig_w / orig_h

    # build all (i,j) pairs whose product ∈ [min_num, max_num]
    target_ratios = sorted(
        {(i, j)
         for n in range(min_num, max_num + 1)
         for i in range(1, n + 1)
         for j in range(1, n + 1)
         if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1]
    )

    w_mul, h_mul = find_closest_aspect_ratio(ar, target_ratios, orig_w, orig_h, image_size)
    target_w, target_h = image_size * w_mul, image_size * h_mul
    blocks = w_mul * h_mul

    resized = image.resize((target_w, target_h))
    cols = target_w // image_size

    crops = []
    for idx in range(blocks):
        row = idx // cols
        col = idx % cols
        box = (
            col * image_size,
            row * image_size,
            (col + 1) * image_size,
            (row + 1) * image_size
        )
        crops.append(resized.crop(box))

    if use_thumbnail and len(crops) != 1:
        crops.append(image.resize((image_size, image_size)))

    return crops

def _preprocess_single_pil(pil_img, transform, input_size, max_num):
    crops = dynamic_preprocess(
        pil_img,
        image_size=input_size,
        use_thumbnail=True,
        max_num=max_num
    )
    return [transform(c) for c in crops]   # list of tensors


def load_image(
    imgs: Union[str, Path, List[Union[str, Path]], torch.Tensor],
    input_size: int = 448,
    max_num: int = 12,
) -> torch.Tensor:
    """
    Parameters
    ----------
    imgs :  • str/Path : path to one image file
            • list/tuple of paths
            • torch.Tensor  [n,3,H,W] or [3,H,W]  (values 0‑255, dtype int / fp16)
    input_size : target side length (square patch size)
    max_num    : maximum #crops per original image

    Returns
    -------
    pixel_values : Tensor  [total_crops, 3, input_size, input_size]
                   normalized to ImageNet mean/std  (dtype = float32)
    """

    transform = build_transform(input_size=input_size)

    # --------------------------------------------------------
    # Phase 1: collect all PIL images --------------------------------
    # --------------------------------------------------------
    pil_images = []

    # 1) path or list‑of‑paths
    if isinstance(imgs, (str, Path)):
        pil_images.append(Image.open(imgs).convert('RGB'))

    elif isinstance(imgs, (list, tuple)) and imgs and isinstance(imgs[0], (str, Path)):
        for p in imgs:
            pil_images.append(Image.open(p).convert('RGB'))

    # 2) tensor input
    elif isinstance(imgs, torch.Tensor):
        if imgs.ndim == 3:                 # [3,H,W]  -> add batch dim
            imgs = imgs.unsqueeze(0)
        assert imgs.ndim == 4 and imgs.shape[1] == 3, \
            "Expect tensor shape [n,3,H,W] or [3,H,W]"

        # move to CPU, ensure uint8
        imgs_cpu = imgs.detach().to('cpu')
        if imgs_cpu.dtype != torch.uint8:
            imgs_cpu = imgs_cpu.round().clamp(0, 255).to(torch.uint8)

        for i in range(imgs_cpu.size(0)):
            pil_images.append(T.functional.to_pil_image(imgs_cpu[i]))

    else:
        raise TypeError("`imgs` must be a path, list of paths, or a [n,3,H,W] tensor")

    # --------------------------------------------------------
    # Phase 2: dynamic tiling + transforms ------------------
    # --------------------------------------------------------
    pixel_tensors = []
    for pil in pil_images:
        pixel_tensors.extend(
            _preprocess_single_pil(pil, transform, input_size, max_num)
        )

    # --------------------------------------------------------
    # Phase 3: stack to one tensor  --------------------------
    # --------------------------------------------------------
    if not pixel_tensors:
        raise RuntimeError("No images found after preprocessing")

    return torch.stack(pixel_tensors)      # [total_crops, 3, input_size, input_size]



class SeparatorStyle(IntEnum):
    """Separator styles."""

    ADD_COLON_SINGLE = auto()
    ADD_COLON_TWO = auto()
    ADD_COLON_SPACE_SINGLE = auto()
    NO_COLON_SINGLE = auto()
    NO_COLON_TWO = auto()
    ADD_NEW_LINE_SINGLE = auto()
    LLAMA2 = auto()
    CHATGLM = auto()
    CHATML = auto()
    CHATINTERN = auto()
    DOLLY = auto()
    RWKV = auto()
    PHOENIX = auto()
    ROBIN = auto()
    FALCON_CHAT = auto()
    CHATGLM3 = auto()
    INTERNVL_ZH = auto()
    MPT = auto()


@dataclasses.dataclass
class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""

    # The name of this template
    name: str
    # The template of the system prompt
    system_template: str = '{system_message}'
    # The system message
    system_message: str = ''
    # The names of two roles
    roles: Tuple[str] = ('USER', 'ASSISTANT')
    # All messages. Each item is (role, message).
    messages: List[List[str]] = ()
    # The number of few shot examples
    offset: int = 0
    # The separator style and configurations
    sep_style: SeparatorStyle = SeparatorStyle.ADD_COLON_SINGLE
    sep: str = '\n'
    sep2: str = None
    # Stop criteria (the default one is EOS token)
    stop_str: Union[str, List[str]] = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        system_prompt = self.system_template.format(system_message=self.system_message)
        if self.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ': ' + message + self.sep
                else:
                    ret += role + ':'
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = system_prompt + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ': ' + message + seps[i % 2]
                else:
                    ret += role + ':'
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_SPACE_SINGLE:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ': ' + message + self.sep
                else:
                    ret += role + ': '  # must be end with a space
            return ret
        elif self.sep_style == SeparatorStyle.ADD_NEW_LINE_SINGLE:
            ret = '' if system_prompt == '' else system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + '\n' + message + self.sep
                else:
                    ret += role + '\n'
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_SINGLE:
            ret = system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + message + seps[i % 2]
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.RWKV:
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += (
                        role
                        + ': '
                        + message.replace('\r\n', '\n').replace('\n\n', '\n')
                    )
                    ret += '\n\n'
                else:
                    ret += role + ':'
            return ret
        elif self.sep_style == SeparatorStyle.LLAMA2:
            seps = [self.sep, self.sep2]
            if self.system_message:
                ret = system_prompt
            else:
                ret = '[INST] '
            for i, (role, message) in enumerate(self.messages):
                tag = self.roles[i % 2]
                if message:
                    if i == 0:
                        ret += message + ' '
                    else:
                        ret += tag + ' ' + message + seps[i % 2]
                else:
                    ret += tag
            return ret
        elif self.sep_style == SeparatorStyle.CHATGLM:
            # source: https://huggingface.co/THUDM/chatglm-6b/blob/1d240ba371910e9282298d4592532d7f0f3e9f3e/modeling_chatglm.py#L1302-L1308
            # source2: https://huggingface.co/THUDM/chatglm2-6b/blob/e186c891cf64310ac66ef10a87e6635fa6c2a579/modeling_chatglm.py#L926
            round_add_n = 1 if self.name == 'chatglm2' else 0
            if system_prompt:
                ret = system_prompt + self.sep
            else:
                ret = ''

            for i, (role, message) in enumerate(self.messages):
                if i % 2 == 0:
                    ret += f'[Round {i//2 + round_add_n}]{self.sep}'

                if message:
                    ret += f'{role}：{message}{self.sep}'
                else:
                    ret += f'{role}：'
            return ret
        elif self.sep_style == SeparatorStyle.CHATML:
            ret = '' if system_prompt == '' else system_prompt + self.sep + '\n'
            for role, message in self.messages:
                if message:
                    ret += role + '\n' + message + self.sep + '\n'
                else:
                    ret += role + '\n'
            return ret
        elif self.sep_style == SeparatorStyle.CHATGLM3:
            ret = ''
            if self.system_message:
                ret += system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + '\n' + ' ' + message
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.CHATINTERN:
            # source: https://huggingface.co/internlm/internlm-chat-7b-8k/blob/bd546fa984b4b0b86958f56bf37f94aa75ab8831/modeling_internlm.py#L771
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                # if i % 2 == 0:
                #     ret += "<s>"
                if message:
                    ret += role + ':' + message + seps[i % 2] + '\n'
                else:
                    ret += role + ':'
            return ret
        elif self.sep_style == SeparatorStyle.DOLLY:
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ':\n' + message + seps[i % 2]
                    if i % 2 == 1:
                        ret += '\n\n'
                else:
                    ret += role + ':\n'
            return ret
        elif self.sep_style == SeparatorStyle.PHOENIX:
            ret = system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + ': ' + '<s>' + message + '</s>'
                else:
                    ret += role + ': ' + '<s>'
            return ret
        elif self.sep_style == SeparatorStyle.ROBIN:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ':\n' + message + self.sep
                else:
                    ret += role + ':\n'
            return ret
        elif self.sep_style == SeparatorStyle.FALCON_CHAT:
            ret = ''
            if self.system_message:
                ret += system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ': ' + message + self.sep
                else:
                    ret += role + ':'

            return ret
        elif self.sep_style == SeparatorStyle.INTERNVL_ZH:
            seps = [self.sep, self.sep2]
            ret = self.system_message + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ': ' + message + seps[i % 2]
                else:
                    ret += role + ':'
            return ret
        elif self.sep_style == SeparatorStyle.MPT:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        else:
            raise ValueError(f'Invalid style: {self.sep_style}')

    def set_system_message(self, system_message: str):
        """Set the system message."""
        self.system_message = system_message

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def update_last_message(self, message: str):
        """Update the last output.
        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.
        """
        self.messages[-1][1] = message

    def to_gradio_chatbot(self):
        """Convert the conversation to gradio chatbot format."""
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def to_openai_api_messages(self):
        """Convert the conversation to OpenAI chat completion format."""
        ret = [{'role': 'system', 'content': self.system_message}]

        for i, (_, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append({'role': 'user', 'content': msg})
            else:
                if msg is not None:
                    ret.append({'role': 'assistant', 'content': msg})
        return ret

    def copy(self):
        return Conversation(
            name=self.name,
            system_template=self.system_template,
            system_message=self.system_message,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )

    def dict(self):
        return {
            'template_name': self.name,
            'system_message': self.system_message,
            'roles': self.roles,
            'messages': self.messages,
            'offset': self.offset,
        }


# A global registry for all conversation templates
conv_templates: Dict[str, Conversation] = {}


def register_conv_template(template: Conversation, override: bool = False):
    """Register a new conversation template."""
    if not override:
        assert (
            template.name not in conv_templates
        ), f'{template.name} has been registered.'

    conv_templates[template.name] = template


def get_conv_template(name: str) -> Conversation:
    """Get a conversation template."""
    return conv_templates[name].copy()



# register_conv_template(
#     Conversation(
#         name='Hermes-2',
#         system_template='<|im_start|>system\n{system_message}',
#         # note: The new system prompt was not used here to avoid changes in benchmark performance.
#         # system_message='我是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。',
#         system_message='你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。',
#         roles=('<|im_start|>user\n', '<|im_start|>assistant\n'),
#         sep_style=SeparatorStyle.MPT,
#         sep='<|im_end|>',
#         stop_str='<|endoftext|>',
#     )
# )


# register_conv_template(
#     Conversation(
#         name='internlm2-chat',
#         system_template='<|im_start|>system\n{system_message}',
#         # note: The new system prompt was not used here to avoid changes in benchmark performance.
#         # system_message='我是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。',
#         system_message='你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。',
#         roles=('<|im_start|>user\n', '<|im_start|>assistant\n'),
#         sep_style=SeparatorStyle.MPT,
#         sep='<|im_end|>',
#     )
# )


# register_conv_template(
#     Conversation(
#         name='phi3-chat',
#         system_template='<|system|>\n{system_message}',
#         # note: The new system prompt was not used here to avoid changes in benchmark performance.
#         # system_message='我是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。',
#         system_message='你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。',
#         roles=('<|user|>\n', '<|assistant|>\n'),
#         sep_style=SeparatorStyle.MPT,
#         sep='<|end|>',
#     )
# )


# register_conv_template(
#     Conversation(
#         name='internvl2_5',
#         system_template='<|im_start|>system\n{system_message}',
#         system_message='你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。',
#         roles=('<|im_start|>user\n', '<|im_start|>assistant\n'),
#         sep_style=SeparatorStyle.MPT,
#         sep='<|im_end|>\n',
#     )
# )


def build_prompt(model, tokenizer, question, num_patches_list,
                 IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                 IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'):
    """
    Returns prompt text + (img_context_token_id already set in model)
    """
    if '<image>' not in question:
        question = '<image>\n' + question
    template = get_conv_template(model.template)
    template.system_message = model.system_message
    template.append_message(template.roles[0], question)   # user
    template.append_message(template.roles[1], None)       # assistant (to be filled)
    prompt = template.get_prompt()

    # replace each <image> placeholder with the correct number of IMG_CONTEXT_TOKENs
    for n_patches in num_patches_list:
        image_tokens = (
            IMG_START_TOKEN
            + IMG_CONTEXT_TOKEN * model.num_image_token * n_patches
            + IMG_END_TOKEN
        )
        prompt = prompt.replace('<image>', image_tokens, 1)

    # store the special‑token id inside the model (needed by forward)
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    return prompt


# ------------------------------------------------------------------
# Main helper: run forward() and return logits
# ------------------------------------------------------------------


@torch.no_grad()
def get_logits(model, tokenizer, pixel_values, question, num_patches_list, device):
    """
    pixel_values       : (total_image_crops, 3, H, W)  — concatenated crops
    num_patches_list   : e.g. [12, 12]  (one entry per *original* image)
    returns logits     : shape (B, seq_len, vocab)
    """
    prompt = build_prompt(model, tokenizer, question, num_patches_list)
    print(prompt)
    tokenizer.padding_side = 'left'
    model_inputs = tokenizer(prompt, return_tensors='pt')
    input_ids      = model_inputs['input_ids'     ].to(device)
    attention_mask = model_inputs['attention_mask'].to(device)

    # forward() wants an image‑level flag; 1 = this crop is used
    image_flags = torch.ones(pixel_values.size(0), 1,
                             dtype=torch.long, device=device)
    global am
    am = attention_mask.detach()
    outputs = model(
        pixel_values=pixel_values.to(device),
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_flags=image_flags,
        return_dict=True
    )
    return outputs.logits        # (1, seq_len, vocab)


def make_pair_chunk(words: List[str],
                    num_img_tokens: int,
                    IMG_START_TOKEN='<img>',
                    IMG_END_TOKEN='</img>',
                    IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'):
    """
    Creates a string chunk for an image-text pair.
    If `words` is empty or None, it creates an image-only chunk.
    """
    # Create the image token part of the chunk
    image_chunk = (
        IMG_START_TOKEN
        + IMG_CONTEXT_TOKEN * num_img_tokens
        + IMG_END_TOKEN
    )
    # If words are provided, append them. Otherwise, return the image chunk alone.
    if words:
        return image_chunk + ' ' + ' '.join(words)
    else:
        return image_chunk

@torch.no_grad()
def logits_by_pair(extractor, tokenizer,
                   pixel_values: torch.Tensor,           # [n, 3, H, W]
                   sentences: List[List[str]],         # Can be None
                   use_template: bool = False           # wrap in chat template?
                   ) -> Tuple[torch.Tensor,
                              List[Tuple[int,int]],
                              torch.Tensor]:
    """
    Returns:
        logits         : [1, seq_len, vocab]
        token_ranges   : list of (start, end) indices per pair
        avg_logits     : [n, vocab]  (avg over sequence tokens of each pair)
    """
    # --- CORRECTED SECTION ---
    # The number of pairs is driven by the number of images, not sentences.
    n = len(pixel_values)
    device = next(extractor.model.parameters()).device

    # ------------------------------------------------------------------
    # 1) Build the text prompt
    # ------------------------------------------------------------------
    num_img_tokens = extractor.model.num_image_token  # K: one image -> K tokens
    
    pair_chunks = []
    for i in range(n):
        # If sentences exist, use the corresponding one. Otherwise, use an empty list.
        # This makes the logic robust to `sentences` being None.
        current_words = sentences[i] if sentences is not None and i < len(sentences) else []
        pair_chunks.append(
            make_pair_chunk(current_words, num_img_tokens)
        )
    
    body = '\n'.join(pair_chunks)

    if use_template:
        tpl = get_conv_template(extractor.model.template)
        tpl.system_message = extractor.model.system_message
        tpl.append_message(tpl.roles[0], body)
        tpl.append_message(tpl.roles[1], None)   # assistant placeholder
        prompt = tpl.get_prompt()
    else:
        prompt = body

    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    extractor.model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    # ------------------------------------------------------------------
    # 2) Tokenise
    # ------------------------------------------------------------------
    tokenizer.padding_side = 'left'
    enc = tokenizer(prompt, return_tensors='pt')
    input_ids      = enc['input_ids'      ].to(device)
    attention_mask = enc['attention_mask'].to(device)

    # ------------------------------------------------------------------
    # 3) Build image_flags (all ones, one per image)
    # ------------------------------------------------------------------
    image_flags = torch.ones(pixel_values.size(0), 1, dtype=torch.long, device=device)

    # ------------------------------------------------------------------
    # 4) Forward pass  -> logits
    # ------------------------------------------------------------------
    logits = extractor(
        pixel_values=pixel_values.to(device),
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_flags=image_flags,
        return_dict=True
    ).logits                                               # [1, seq_len, vocab]

    # ------------------------------------------------------------------
    # 5) Find token-range for each pair  (start index of every <img>)
    # ------------------------------------------------------------------
    img_start_id = tokenizer.convert_tokens_to_ids('<img>')
    token_ids = input_ids[0]                                   # (seq_len,)
    start_idxs = (token_ids == img_start_id).nonzero(as_tuple=False).flatten().tolist()
    assert len(start_idxs) == n, "didn't find <img> marker for every image"

    token_ranges = []
    for i in range(n):
        s = start_idxs[i]
        e = start_idxs[i+1]-1 if i < n-1 else len(token_ids)-1
        token_ranges.append((s, e))

    # ------------------------------------------------------------------
    # 6) Average logits over each range -> [n, vocab]
    # ------------------------------------------------------------------
    avg_logits = torch.stack([
        logits[0, s:e+1].mean(dim=0)           # mean over sequence dimension
        for (s, e) in token_ranges
    ], dim=0)                                          # [n, vocab]

    return logits, token_ranges, avg_logits

def select_16_frames(video_section, target_frames=16):
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

def extract_fn(
    video: torch.Tensor, 
    audio: torch.Tensor, 
    transcript: List[List[str]], 
    verbose: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Extracts features from video and optional transcript using a pre-trained model.

    Args:
        video (torch.Tensor): A tensor of video frames.
        audio (torch.Tensor): A tensor of audio waveforms.
        transcript (List[List[str]]): A list of word lists from the transcript. Can be None.
        verbose (bool): If True, prints status messages.

    Returns:
        Dict[str, torch.Tensor]: A dictionary mapping layer names to extracted feature tensors.
    """
    dict_return = {}
    with torch.no_grad():
        # Select the first frame of each chunk in the video batch
        pixel_values = video[:, 0] 
        # Preprocess the images for the model
        pixel_values = load_image(pixel_values).to(torch.bfloat16).to(device)

        # --- CORRECTED SECTION ---
        # Explicitly handle the case where the transcript is missing or empty.
        # If it's invalid, `sentences` becomes None.
        sentences = transcript if transcript and all(transcript) else None
        
        # Call the feature extraction helper. It is designed to handle `sentences` being None.
        print("getting logits")
        logits, ranges, avg_logits = logits_by_pair(
            extractor, tokenizer,
            pixel_values, sentences,
            use_template=False
        )

        # Retrieve the intermediate activations captured by the hooks
        features = extractor.features

        # Process the features from each requested layer
        for layer_name, activation in features.items():
            # Average the features across the token sequence for each image-text pair
            avg_activation = torch.stack([
                activation[0, s:e+1].mean(dim=0)  # Mean over the sequence dimension
                for (s, e) in ranges
            ], dim=0)  
            
            if verbose:
                print(f"Layer: {layer_name}, Feature shape: {activation.shape}, Averaged feature shape: {avg_activation.shape}")
            
            # Store the final, averaged features for the layer
            dict_return[layer_name] = avg_activation.to(torch.float16).cpu()
            
    return dict_return


def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed)

    parts = cfg.feature_extraction_parts
    movies_base = DEFAULT_DATA_DIR / cfg.dataset_dir / 'stimuli' / 'movies'
    transcripts_base = DEFAULT_DATA_DIR / cfg.dataset_dir / 'stimuli' / 'movies'
    out_dir = cfg.out_dir
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda")
    model = AutoModel.from_pretrained(
        cfg.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    ).eval()
    model= model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model,
        trust_remote_code=True,
        use_fast=False
    )
    # print("loaded the model")
    layers_to_extract = cfg.layers
    # print("extractor")
    extractor = HuggingFaceFeatureExtractor(model, layers_to_extract, detach=True)
    # print("extract features")
    extract_features(parts = parts, movies_base = movies_base, transcripts_base = transcripts_base, output_dir = out_dir, extraction_fn = extract_fn, verbose = True, modality = 'all', past_context_in_seconds = cfg.past_context, splits_overlap=cfg.splits_overlap, ignore_done =[], ood=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", type=str, default=None)
    parser.add_argument("--overrides", type=str, default=None, nargs="+")
    args = parser.parse_args()
    cfg = OmegaConf.load(DEFAULT_CONFIG)
    if args.cfg_path:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.load(args.cfg_path))
    if args.overrides:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.from_dotlist(args.overrides))
    main(cfg)