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
import einops
import torchvision.transforms as T
from transformers import AutoVideoProcessor, AutoModel
from feature_extractor import FeatureExtractor

ROOT = Path(__file__).parent.parent
DEFAULT_DATA_DIR = ROOT / "datasets"
DEFAULT_CONFIG = ROOT / "config/default_vjepa2_features.yaml"


def load_transcript(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep='\t')
        return df
    except Exception as e:
        raise RuntimeError(f"Error loading transcript from {path}: {e}")

def load_audio(path: str, sampling_rate: int = 48000, stereo: bool = True) -> tuple[torch.Tensor, int]:
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

def load_video(path: str, resolution: Tuple[int, int] = None, tensor_dtype: torch.dtype = torch.float16, verbose: bool = True) -> torch.Tensor:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError("Cannot open video file: {}".format(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames_to_read = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if verbose:
        print(f"Total number of frames in the video: {num_frames_to_read}")
        print(f"Original Resolution: ({cap.get(cv2.CAP_PROP_FRAME_WIDTH)}, {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)})")
        print(f"FPS: {fps}")
        print(f"Duration (seconds): {num_frames_to_read / fps}")
        print(f"Target Resolution: {resolution}")
    if resolution is None:
        H, W = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    else:
        H, W = resolution
    frames = torch.zeros(num_frames_to_read, 3, H, W, dtype=tensor_dtype)
    for i in range(num_frames_to_read):
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1)
        frame_tensor = F.interpolate(frame_tensor.unsqueeze(0), size=(H,W), mode='bilinear', align_corners=False)
        frames[i] = frame_tensor
    cap.release()
    if verbose:
        print(f"Read {len(frames)} frames.")
        print(f"Frames shape: {frames.shape}")
    return frames, fps

def extract_features(parts: List[str], movies_base: str, transcripts_base: str, output_dir: str, extraction_fn: Callable, interval: int = 1.49, verbose: bool = True, modality: str = 'all', ood: bool = False):
    movies_base = Path(movies_base)
    transcripts_base = Path(transcripts_base)
    if not movies_base.exists():
        raise FileNotFoundError(f"Movies directory not found: {movies_base}")
    if not transcripts_base.exists():
        raise FileNotFoundError(f"Transcripts directory not found: {transcripts_base}")
    for folder in movies_base.rglob('*'):
        if folder.is_dir() and folder.name in parts:
            for movie_file in folder.glob('*.mkv'):
                try:
                    rel_folder = folder.relative_to(movies_base)
                except ValueError:
                    continue
                print(rel_folder)
                if "friends" in str(rel_folder):
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
                video, audio, transcript, sample_rate, fps_video = None, None, None, None, None
                if modality == 'all' or modality == 'video':
                    video, fps_video = load_video(movie_file, verbose=verbose, resolution=(224,224))
                if modality in ['all', 'audio', 'video', 'transcript']:
                    audio, sample_rate = load_audio(movie_file)
                if modality in ['all', 'transcript']:
                    transcript = load_transcript(transcript_file)
                if transcript is not None:
                    transcript = resample_transcript(transcript, interval)
                total_duration = audio.shape[1] / sample_rate
                num_intervals = math.ceil(total_duration / interval)
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
                    for i in tqdm(range(num_intervals)):
                        video_section, audio_section, transcript_section = extract_section(video, audio, transcript, interval, i, sample_rate, modality, fps_video)
                        output_features = extraction_fn(video_section, audio_section, transcript_section, verbose)
                        for layer_name, tensor in output_features.items():
                            tensor_np = tensor.cpu().numpy()
                            if layer_name not in features_datasets:
                                features_datasets[layer_name] = f.create_dataset(layer_name, data=tensor_np[np.newaxis, ...], maxshape=(None,) + tensor_np.shape, dtype=np.float16, chunks=True)
                            else:
                                ds = features_datasets[layer_name]
                                ds.resize(ds.shape[0] + 1, axis=0)
                                ds[-1] = tensor_np

def resample_transcript(transcript: pd.DataFrame, new_interval: float) -> pd.DataFrame:
    all_words, all_onsets, all_durations = [], [], []
    for _, row in transcript.iterrows():
        if not row['onsets_per_tr'] or row['onsets_per_tr'] == []:
            continue
        onsets = ast.literal_eval(row['onsets_per_tr']) if isinstance(row['onsets_per_tr'], str) else row['onsets_per_tr']
        words = ast.literal_eval(row['words_per_tr']) if isinstance(row['words_per_tr'], str) else row['words_per_tr']
        durations = ast.literal_eval(row['durations_per_tr']) if isinstance(row['durations_per_tr'], str) else row['durations_per_tr']
        all_words.extend(words)
        all_onsets.extend(onsets)
        all_durations.extend(durations)
    df = pd.DataFrame({'word': all_words, 'onset': all_onsets, 'duration': all_durations})
    df['word_end'] = df['onset'] + df['duration']
    df['new_index'] = (df['word_end'] // new_interval).astype(int)
    grouped = df.groupby('new_index').agg({'word': list, 'onset': list, 'duration': list, 'word_end': list}).reset_index(drop=True)
    return grouped

def extract_section(video: torch.Tensor, audio: torch.Tensor, transcript: pd.DataFrame, interval: float, index: int, sample_rate: int, modality: str = 'all', fps_video: float = 30) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    audio_section, video_section, transcript_section = None, None, []
    start_time, end_time = index * interval, (index + 1) * interval
    if modality in ['all', 'audio']:
        audio_start, audio_end = int(start_time * sample_rate), int(end_time * sample_rate)
        audio_section = audio[:, audio_start:audio_end]
    if modality in ['all', 'video']:
        frame_start, frame_end = round(start_time * fps_video), round(end_time * fps_video)
        video_section = video[frame_start:frame_end]
    if modality in ['all', 'transcript']:
        transcript_section = transcript['word'].iloc[index]
    return video_section, audio_section, transcript_section

def select_frames(video_section, target_frames=16):
    num_frames = video_section.shape[0]
    if num_frames >= target_frames:
        indices = torch.linspace(0, num_frames - 1, steps=target_frames).long()
        selected_frames = video_section[indices]
    else:
        repeats = target_frames // num_frames
        remainder = target_frames % num_frames
        repeated_frames = video_section.repeat(repeats, 1, 1, 1)
        if remainder > 0:
            extra_frames = video_section[:remainder]
            repeated_frames = torch.cat([repeated_frames, extra_frames], dim=0)
        selected_frames = repeated_frames
    return selected_frames

def extract_fn(video: torch.Tensor, audio: torch.Tensor, transcript: List[str], verbose: bool, extractor: FeatureExtractor, processor: AutoVideoProcessor, model: AutoModel, num_frames_to_extract: int) -> Dict[str, torch.Tensor]:
    with torch.no_grad():
        if video.shape[0] < 1:
            return {layer: None for layer in extractor.layers}
        if video.shape[0] < num_frames_to_extract:
            video = video.to(torch.uint8)
        else:
            video = select_frames(video, num_frames_to_extract).to(torch.uint8)
        video_processed = processor(video, return_tensors="pt").to(model.device)
        video_embeddings = extractor(**video_processed)
        features = extractor.features
        dict_return = {}
        for layer_name, activation in features.items():
            avg_activation = activation.mean(dim=1)
            dict_return[layer_name + '_avg'] = avg_activation.to(torch.float16).to('cpu')
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
    model = AutoModel.from_pretrained(cfg.model).to(device).eval()
    processor = AutoVideoProcessor.from_pretrained(cfg.model)
    layers_to_extract = cfg.layers
    extractor = FeatureExtractor(model, layers_to_extract, call_fn="get_vision_features")
    num_frames_to_extract = cfg.num_frames
    extraction_fn_wrapper = lambda video, audio, transcript, verbose: extract_fn(video, audio, transcript, verbose, extractor, processor, model, num_frames_to_extract)
    extract_features(parts=parts, movies_base=str(movies_base), transcripts_base=str(transcripts_base), output_dir=str(out_dir), extraction_fn=extraction_fn_wrapper, verbose=True, modality='video', ood=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--overrides", type=str, default=None, nargs="+")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_path)
    if args.overrides:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.from_dotlist(args.overrides))
    main(cfg)
