import numpy as np
import pandas as pd
import os
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
from transformers import WhisperModel, AutoProcessor

ROOT = Path(__file__).parent.parent
DEFAULT_DATA_DIR = ROOT / "datasets"
DEFAULT_CONFIG = ROOT / "config/default_whisper_features.yaml"

class HuggingFaceFeatureExtractor:
    def __init__(self, model: nn.Module, layers: List[str], detach: bool = True):
        self.model = model
        self.detach = detach
        self.layers = self._expand_layers(model, layers)
        self._features: Dict[str, Any] = {}
        self._handles: Dict[str, Any] = {}
        self._register_hooks()

    def _register_hooks(self):
        for layer in self.layers:
            sub_module = self.model.get_submodule(layer)
            handle = sub_module.register_forward_hook(self._make_hook(layer))
            self._handles[layer] = handle

    def _make_hook(self, layer_name: str):
        def hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any):
            self._features[layer_name] = output.detach() if self.detach else output
        return hook

    def clear(self):
        self._features.clear()

    @property
    def features(self) -> Dict[str, Any]:
        return dict(self._features)

    def __call__(self, *args, **kwargs) -> Any:
        self.clear()
        return self.model(*args, **kwargs)

    def remove_hooks(self):
        for handle in self._handles.values():
            handle.remove()
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.remove_hooks()

    @staticmethod
    def _expand_layers(model: nn.Module, layers: List[str]) -> List[str]:
        all_layers = [name for name, _ in model.named_modules() if name]
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

def load_transcript(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep='\t')
        return df
    except Exception as e:
        raise RuntimeError(f"Error loading transcript from {path}: {e}")

def load_audio(path: str, sampling_rate: int = 48000, stereo: bool = True) -> (torch.Tensor, int):
    try:
        waveform, orig_sr = torchaudio.load(path)
        if not stereo and waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if orig_sr != sampling_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=sampling_rate)
            waveform = resampler(waveform)
        waveform_fp16 = waveform.half()
        del waveform
        return waveform_fp16, sampling_rate
    except Exception as e:
        raise RuntimeError(f"Error loading audio from {path}: {e}")

def load_video(path: str, resolution: Tuple[int, int] = (224, 224), tensor_dtype: torch.dtype = torch.float16, verbose: bool = True) -> torch.Tensor:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames_to_read = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if verbose:
        print(f"Total number of frames in the video: {num_frames_to_read}")
        print(f"Original Resolution: ({cap.get(cv2.CAP_PROP_FRAME_WIDTH)}, {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)})")
        print(f"FPS: {fps}")
        print(f"Duration (seconds): {num_frames_to_read / fps}")
        print(f"Target Resolution: {resolution}")
    frames = torch.zeros(num_frames_to_read, 3, 224, 224, dtype=tensor_dtype)
    for i in range(num_frames_to_read):
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1)
        frame_tensor = F.interpolate(frame_tensor.unsqueeze(0), size=224, mode='bilinear', align_corners=False)
        frames[i] = frame_tensor
    cap.release()
    if verbose:
        print(f"Read {len(frames)} frames.")
        print(f"Frames shape: {frames.shape}")
    return frames

def extract_features(parts: List[str], movies_base: str, transcripts_base: str, output_dir: str, extraction_fn: Callable, interval: int = 1.49, verbose: bool = True, modality: str = 'all', ood: bool = False):
    movies_base = Path(movies_base)
    transcripts_base = Path(transcripts_base)
    if not movies_base.exists():
        raise FileNotFoundError(f"Movies directory not found: {movies_base}")
    if not transcripts_base.exists():
        raise FileNotFoundError(f"Transcripts directory not found: {transcripts_base}")
    for folder in movies_base.rglob('*'):
        if not (folder.is_dir() and folder.name in parts):
            continue
        for movie_file in folder.glob('*.mkv'):
            try:
                rel_folder = folder.relative_to(movies_base)
            except ValueError:
                continue
            if ood:
                ood_suffix = str(movie_file.with_suffix('.tsv').name)[5:].replace('_video', '')
                if not os.path.splitext(ood_suffix)[0][-1].isdigit():
                    print(f"skipping {ood_suffix}")
                    continue
                transcript_file = transcripts_base / rel_folder / f"ood_{ood_suffix}.tsv"
            else:
                transcript_file = transcripts_base / rel_folder / movie_file.with_suffix('.tsv').name
            if verbose:
                print(f"Movie:      {movie_file}")
                print(f"Transcript: {transcript_file}")
            video, audio, transcript, sample_rate, fps_video = None, None, None, None, 30.0
            if modality in ['all', 'video']:
                video, fps_video = load_video(movie_file, verbose=verbose)
            if modality in ['all', 'audio', 'video', 'transcript']:
                audio, sample_rate = load_audio(movie_file)
            if transcript_file.exists() and (modality in ['all', 'transcript']):
                transcript = load_transcript(transcript_file)
            else:
                transcript = None
                if verbose:
                    print(f"Transcript file not found or not requested for {movie_file.name}, proceeding without it.")
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
                    video_section, audio_section, transcript_section = extract_section(video, audio, transcript, interval, i, sample_rate, modality)
                    if audio_section is None or audio_section.shape[1] == 0:
                        continue
                    output_features = extraction_fn(video_section, audio_section, transcript_section, verbose)
                    for layer_name, tensor in output_features.items():
                        feature_vector = tensor.cpu().numpy().squeeze(0)
                        if layer_name not in features_datasets:
                            features_datasets[layer_name] = f.create_dataset(layer_name, shape=(1,) + feature_vector.shape, maxshape=(None,) + feature_vector.shape, dtype=np.float16, chunks=True)
                            features_datasets[layer_name][0] = feature_vector
                        else:
                            ds = features_datasets[layer_name]
                            ds.resize(ds.shape[0] + 1, axis=0)
                            ds[-1] = feature_vector

def extract_section(video: torch.Tensor, audio: torch.Tensor, transcript: pd.DataFrame, interval: float, index: int, sample_rate: int, modality: str = 'all') -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    audio_section, video_section, transcript_section = None, None, None
    start_time, end_time = index * interval, (index + 1) * interval
    if modality in ['all', 'audio']:
        audio_start, audio_end = int(start_time * sample_rate), int(end_time * sample_rate)
        audio_section = audio[:, audio_start:audio_end]
    if modality in ['all', 'video']:
        fps_video = 30
        frame_start, frame_end = round(start_time * fps_video), round(end_time * fps_video)
        video_section = video[frame_start:frame_end]
    if modality in ['all', 'transcript']:
        transcript_section = transcript['words_per_tr'].iloc[index]
    return video_section, audio_section, transcript_section

def extract_fn(video, audio, transcript, verbose, extractor, processor, model, device, sampling_rate):
    with torch.no_grad():
        inputs = processor(audio[0][:int(sampling_rate*1.49)], sampling_rate=sampling_rate, return_tensors='pt')
        outputs = extractor(inputs['input_features'].half().to(device))
    features = extractor.features
    dict_return = {}
    for layer_name, activation in features.items():
        dict_return[layer_name] = activation.mean(1)
    extractor.clear()
    return dict_return

def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    parts = cfg.feature_extraction_parts
    movies_base = DEFAULT_DATA_DIR / cfg.dataset_dir / 'stimuli' / 'movies'
    transcripts_base = DEFAULT_DATA_DIR / cfg.dataset_dir / 'stimuli' / 'movies'
    out_dir = Path(cfg.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(cfg.device)
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = WhisperModel.from_pretrained(cfg.model, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
    model.to(device)
    processor = AutoProcessor.from_pretrained(cfg.model)
    layers_to_extract = cfg.layers
    extractor = HuggingFaceFeatureExtractor(model.encoder, layers_to_extract, detach=True)
    sampling_rate = 16000
    extraction_fn_wrapper = lambda video, audio, transcript, verbose: extract_fn(video, audio, transcript, verbose, extractor, processor, model, device, sampling_rate)
    extract_features(parts=parts, movies_base=str(movies_base), transcripts_base=str(transcripts_base), output_dir=str(out_dir), extraction_fn=extraction_fn_wrapper, verbose=True, modality='audio', ood=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--overrides", type=str, default=None, nargs="+")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_path)
    if args.overrides:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.from_dotlist(args.overrides))
    main(cfg)