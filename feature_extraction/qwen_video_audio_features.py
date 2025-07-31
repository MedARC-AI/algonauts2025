import numpy as np 
import pandas as pd 
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
import soundfile as sf
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniThinkerForConditionalGeneration
from feature_extractor import FeatureExtractor

ROOT = Path(__file__).parent.parent
DEFAULT_DATA_DIR = ROOT / "datasets"
DEFAULT_CONFIG = ROOT / "config/default_qwen_features.yaml"



# class HuggingFaceFeatureExtractor:
#     def __init__(self, model: nn.Module, layers: List[str], detach: bool = True):
#         self.model = model
#         self.detach = detach
#         self.layers = self._expand_layers(model, layers)
#         self._features: Dict[str, Any] = {}
#         self._handles: Dict[str, Any] = {}
#         self._register_hooks()

#     def _register_hooks(self):
#         for layer in self.layers:
#             sub_module = self.model.get_submodule(layer)
#             handle = sub_module.register_forward_hook(self._make_hook(layer))
#             self._handles[layer] = handle

#     def _make_hook(self, layer_name: str):
#         def hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any):
#             self._features[layer_name] = output.detach() if self.detach else output
#         return hook

#     def clear(self):
#         self._features.clear()

#     @property
#     def features(self) -> Dict[str, Any]:
#         return dict(self._features)

#     def __call__(self, *args, **kwargs) -> Any:
#         self.clear()
#         return self.model(*args, **kwargs)

#     def remove_hooks(self):
#         for handle in self._handles.values():
#             handle.remove()
#         self._handles.clear()

#     def __enter__(self):
#         return self

#     def __exit__(self, exc_type, exc_value, traceback):
#         self.remove_hooks()

#     @staticmethod
#     def _expand_layers(model: nn.Module, layers: List[str]) -> List[str]:
#         all_layers = [name for name, _ in model.named_modules() if name]
#         all_layers_set = set(all_layers)
#         expanded = []
#         special_chars = set("*?[]")
#         for layer in layers:
#             if not any(char in layer for char in special_chars):
#                 if layer not in all_layers_set:
#                     raise ValueError(f"Layer '{layer}' not found in the model.")
#                 expanded.append(layer)
#             else:
#                 matches = fnmatch.filter(all_layers, layer)
#                 if not matches:
#                     raise ValueError(f"No layers match the pattern '{layer}'.")
#                 expanded.extend(matches)
#         return expanded



def load_transcript(
    path: str
) -> pd.DataFrame:
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


def load_video(path, resolution=(224, 224), dtype=torch.float16, verbose: bool = True):
    vr   = VideoReader(str(path), ctx=cpu(0))
    fps  = vr.get_avg_fps()
    frames = vr.get_batch(range(len(vr)))
    frames = torch.from_numpy(frames.asnumpy()).permute(0,3,1,2)
    frames = F.interpolate(frames,
                           size=resolution, mode="bilinear",
                           align_corners=False)
    frames = frames.to(dtype=dtype)
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

                if str(movie_file).split('/')[-1].split('.')[0] + '.h5' in ignore_done:
                    continue

                video, audio, transcript, sample_rate, fps_video = None, None, None, None, None
                if modality == 'all' or modality == 'video':
                    video, fps_video = load_video(movie_file, verbose=verbose, resolution=(256, 384))
                if modality in ['all', 'audio', 'video', 'transcript']:
                    audio, sample_rate = load_audio(movie_file, sampling_rate=16000, stereo=False)

                if (modality in ['all', 'transcript', 'video']) and transcript_file.exists():
                    transcript = load_transcript(transcript_file)
                    if transcript is not None:
                        transcript = resample_transcript(transcript, interval)
                else:
                    transcript = None
                    print(f"Transcript file not found or not requested for {movie_file.name}, proceeding without it.")

                total_duration = video.shape[0] / fps_video
                num_intervals_tr = int(total_duration // interval)

                if verbose:
                    print(f"Total duration: {total_duration:.2f} seconds")
                    print(f"Number of intervals: {num_intervals_tr}")
                    print(f"Sample rate: {sample_rate}")

                output_folder = Path(output_dir) / rel_folder
                output_folder.mkdir(parents=True, exist_ok=True)
                output_file = output_folder / movie_file.with_suffix('.h5').name

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

                with h5py.File(output_file, 'w') as f:
                    features_datasets = {} 
                    for i in tqdm(range(total_iterations)):
                        start_index = i * real_increment_in_TRs
                        end_index = start_index + context_in_TRs

                        if start_index >= total_trs:
                            break

                        future_offset = min(end_index, total_trs) - start_index - 1

                        video_section, audio_section, transcript_section = extract_section(
                            video, audio, transcript, interval, start_index, sample_rate, modality, fps_video, past_offset = 0, future_offset = future_offset, split_by_tr = True
                        )

                        output_features = extraction_fn(video_section, audio_section, transcript_section, verbose, clip_fps = fps_video)

                        for layer_name, tensor in output_features.items():
                            assert tensor.shape[0] == video_section.shape[0], f"Error on layer: {layer_name}, the number of TRs of the output features should be the same as the number of TRs of the video section. Got {tensor.shape[0]} and {video_section.shape[0]}"

                        for layer_name, tensor in output_features.items():
                            tensor_np = tensor.cpu().numpy()
                            if layer_name not in features_datasets:
                                features_datasets[layer_name] = f.create_dataset(
                                    layer_name,
                                    data=tensor_np,
                                    maxshape=(None,) + tensor_np.shape[1::],
                                    dtype=np.float16,
                                    chunks=True,
                                )
                            else:
                                ds = features_datasets[layer_name]
                                last_shape = ds.shape[0]
                                t_end_index = min(end_index, total_trs)
                                ds.resize(t_end_index, axis=0)
                                ds[last_shape:t_end_index] = tensor_np[-(t_end_index - last_shape)::]


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
    grouped = df.groupby('new_index').agg({'word': list, 'onset': list, 'duration': list, 'word_end': list}).reset_index()
    max_index = 0 if pd.isna(df['new_index'].max()) else int(df['new_index'].max())
    complete_intervals = pd.DataFrame({'new_index': range(max_index + 1)})
    result = complete_intervals.merge(grouped, on='new_index', how='left')
    for col in ['word', 'onset', 'duration', 'word_end']:
        result[col] = result[col].apply(lambda x: x if isinstance(x, list) else [])
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
) -> Tuple[torch.Tensor, torch.Tensor, List[List[str]]]:
    extraction_start_index = index - past_offset + 1 if past_offset > 0 else index
    extraction_end_index = index + future_offset
    total_intervals = extraction_end_index - extraction_start_index + 1
    requested_start_time = extraction_start_index * interval
    requested_end_time = (extraction_end_index + 1) * interval

    audio_section = None
    if modality in ['all', 'audio']:
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

    video_section = None
    if modality in ['all', 'video']:
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

    transcript_section = []
    if modality in ['all', 'transcript'] and transcript is not None:
        for i in range(total_intervals):
            global_idx = extraction_start_index + i
            if global_idx < 0 or global_idx >= len(transcript):
                transcript_section.append([])
            else:
                transcript_section.append(transcript['word'].iloc[global_idx])

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
    FPS_qwen = 2
    N = len(frames)
    video_chunks = ["<|vision_bos|><|VIDEO|><|vision_eos|>" for _ in range(N)]
    text_input = ["".join(video_chunks)]
    videos = [select_frames(frames[i], target_frames=round(frames[i].shape[0] // clips_fps * FPS_qwen)) for i in range(N)]
    inputs = processor(
        text=text_input,
        audio=[tuple(audio_i) for audio_i in audio],
        images=None,
        videos=videos,
        return_tensors="pt",
        padding=False,
        use_audio_in_video=use_audio_in_video,
    ).to(model_device).to(model_dtype)

    ids = inputs["input_ids"][0].tolist()
    bos_id = processor.tokenizer.convert_tokens_to_ids("<|vision_bos|>")
    eos_id = processor.tokenizer.convert_tokens_to_ids("<|vision_eos|>")

    intervals: List[Tuple[int, int]] = []
    active_start = None
    for idx, tok in enumerate(ids):
        if tok == bos_id and active_start is None:
            active_start = idx
        elif tok == eos_id and active_start is not None:
            intervals.append((active_start, idx))
            active_start = None

    assert len(intervals) == N, "Failed to find all clip spans"
    return inputs, intervals


def extract_fn(
    video: torch.Tensor, 
    audio: torch.Tensor, 
    transcript: List[List[str]], 
    verbose: bool,
    clip_fps: None,
    extractor: FeatureExtractor,
    processor: Qwen2_5OmniProcessor,
    model: Qwen2_5OmniThinkerForConditionalGeneration
) -> Dict[str, torch.Tensor]:
    dict_return = {}
    with torch.no_grad():
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
        features = extractor.features

        for layer_name, activation in features.items():
            avg_activation = torch.stack([
                activation[0, s:e+1].mean(dim=0)
                for (s, e) in intervals
            ], dim=0)  
            dict_return[layer_name] = avg_activation.to(torch.float16).cpu()
    return dict_return


def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    parts = cfg.feature_extraction_parts
    movies_base = DEFAULT_DATA_DIR / cfg.dataset_dir / 'stimuli' / 'movies'
    transcripts_base = DEFAULT_DATA_DIR / cfg.dataset_dir / 'stimuli' / 'movies'
    out_dir = Path(cfg.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        cfg.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).eval()
    device = torch.device("cuda")
    model = model.to(device)
    processor = Qwen2_5OmniProcessor.from_pretrained(cfg.model)
    processor.max_pixels = 128 * 28 * 28
    
    layers_to_extract = cfg.layers
    extractor = FeatureExtractor(model, layers_to_extract, detach=True)
    
    extraction_fn_wrapper = lambda video, audio, transcript, verbose, clip_fps: extract_fn(
        video, audio, transcript, verbose, clip_fps, extractor, processor, model
    )

    extract_features(
        parts=parts, 
        movies_base=str(movies_base), 
        transcripts_base=str(transcripts_base), 
        output_dir=str(out_dir), 
        extraction_fn=extraction_fn_wrapper, 
        verbose=True, 
        modality='all', 
        past_context_in_seconds=cfg.past_context, 
        splits_overlap=cfg.splits_overlap, 
        ignore_done=[], 
        ood=False
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--overrides", type=str, default=None, nargs="+")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_path)
    if args.overrides:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.from_dotlist(args.overrides))
    main(cfg)
