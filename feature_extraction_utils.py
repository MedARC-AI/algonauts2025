from typing import Optional, Callable, Tuple, Any
from torchvision.datasets.vision import VisionDataset
import os
from pathlib import Path
import glob
import re
import numpy as np
import pandas as pd
import h5py
import torch
import librosa
import ast
import string
import zipfile
from tqdm.notebook import tqdm
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import cv2
import nibabel as nib
from nilearn import plotting
from nilearn.maskers import NiftiLabelsMasker
import ipywidgets as widgets
from ipywidgets import VBox, Dropdown, Button
from IPython.display import Video, display, clear_output
from moviepy.editor import VideoFileClip
from transformers import BertTokenizer, BertModel
from torchvision.transforms import Compose, Lambda, CenterCrop
from torchvision.models.feature_extraction import create_feature_extractor
from pytorchvideo.transforms import Normalize, UniformTemporalSubsample, ShortSideScale
def extract_visual_features(episode_path, tr, feature_extractor, model_layer,
    transform, device, save_dir_temp, save_dir_features):
    """
    Extract visual features from a movie using a pre-trained video model.

    Parameters
    ----------
    episode_path : str
        Path to the movie file for which the visual features are extracted.
    tr : float
        Duration of each chunk, in seconds (aligned with the fMRI repetition
        time, or TR).
    feature_extractor : torch.nn.Module
        Pre-trained feature extractor model.
    model_layer : str
        The model layer from which the visual features are extracted.
    transform : torchvision.transforms.Compose
        Transformation pipeline for processing video frames.
    device : torch.device
        Device for computation ('cpu' or 'cuda').
    save_dir_temp : str
        Directory where the chunked movie clips are temporarily stored for
        feature extraction.
    save_dir_features : str
        Directory where the extracted visual features are saved.

    Returns
    -------
    visual_features : float
        Array containing the extracted visual features.

    """

    # Get the onset time of each movie chunk
    clip = VideoFileClip(episode_path)
    start_times = [x for x in np.arange(0, clip.duration, tr)][:-1]
    # Create the directory where the movie chunks are temporarily saved
    temp_dir = os.path.join(save_dir_temp, 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    # Empty features list
    visual_features = []

    # Loop over chunks
    with tqdm(total=len(start_times), desc="Extracting visual features") as pbar:
        for start in start_times:

            # Divide the movie in chunks of length TR, and save the resulting
            # clips as '.mp4' files
            clip_chunk = clip.subclip(start, start+tr)
            chunk_path = os.path.join(temp_dir, 'visual_chunk.mp4')
            clip_chunk.write_videofile(chunk_path, verbose=False, audio=False,
                logger=None)
            # Load the frames from the chunked movie clip
            video_clip = VideoFileClip(chunk_path)
            chunk_frames = [frame for frame in video_clip.iter_frames()]

            # Format the frames to shape:
            # (batch_size, channels, num_frames, height, width)
            frames_array = np.transpose(np.array(chunk_frames), (3, 0, 1, 2))
            # Convert the video frames to tensor
            inputs = torch.from_numpy(frames_array).float()
            # Preprocess the video frames
            inputs = transform(inputs).unsqueeze(0).to(device)

            # Extract the visual features
            with torch.no_grad():
                preds = feature_extractor(inputs)
            visual_features.append(np.reshape(preds[model_layer].cpu().numpy(), -1))

            # Update the progress bar
            pbar.update(1)

    # Convert the visual features to float32
    visual_features = np.array(visual_features, dtype='float32')

    # Save the visual features
    out_file_visual = os.path.join(
       save_dir_features, f'friends_s01e01a_features_visual.h5')
    with h5py.File(out_file_visual, 'a' if Path(out_file_visual).exists() else 'w') as f:
       group = f.create_group("s01e01a")
       group.create_dataset('visual', data=visual_features, dtype=np.float32)
    print(f"Visual features saved to {out_file_visual}")

    # Output
    return visual_features
def load_mkv_file(movie_path):
    """
    Load video and audio data from the given .mkv movie file, and additionally
    prints related information.

    Parameters
    ----------
    movie_path : str
        Path to the .mkv movie file.

    """

    # Read the .mkv file
    cap = cv2.VideoCapture(movie_path)

    if not cap.isOpened():
        print("Error: Could not open movie.")
        return

    # Get video information
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = video_total_frames / video_fps
    video_duration_minutes = video_duration / 60

    # Print video information
    print(">>> Video Information <<<")
    print(f"Video FPS: {video_fps}")
    print(f"Video Resolution: {video_width}x{video_height}")
    print(f"Total Frames: {video_total_frames}")
    print(f"Video Duration: {video_duration:.2f} seconds or {video_duration_minutes:.2f} minutes")

    # Release the video object
    cap.release()

    # Audio information
    clip = VideoFileClip(movie_path)
    audio = clip.audio
    audio_duration = audio.duration
    audio_fps = audio.fps
    print("\n>>> Audio Information <<<")
    print(f"Audio Duration: {audio_duration:.2f} seconds")
    print(f"Audio FPS (Sample Rate): {audio_fps} Hz")

    # Extract and display the first 20 seconds of the video
    output_video_path = 'first_20_seconds.mp4'
    video_segment = clip.subclip(0, min(20, video_duration))
    print("\nCreating clip of the first 20 seconds of the video...")
    video_segment.write_videofile(output_video_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)

    # Display the video in the notebook
    display(Video(output_video_path, embed=True, width=640, height=480))
    

def load_transcript(transcript_path):
    """
    Loads a transcript file and returns it as a DataFrame.

    Parameters
    ----------
    transcript_path : str
        Path to the .tsv transcript file.

    """
    df = pd.read_csv(transcript_path, sep='\t')
    return df


def get_movie_info(movie_path):
    """
    Extracts the frame rate (FPS) and total duration of a movie.

    Parameters
    ----------
    movie_path : str
        Path to the .mkv movie file.

    """

    cap = cv2.VideoCapture(movie_path)
    fps, frame_count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    return fps, frame_count / fps


def split_movie_into_chunks(movie_path, chunk_duration=1.49):
    """
    Divides a video into fixed-duration chunks.

    Parameters
    ----------
    movie_path : str
        Path to the .mkv movie file.
    chunk_duration : float, optional
        Duration of each chunk in seconds (default is 1.49).

    """

    _, video_duration = get_movie_info(movie_path)
    chunks = []
    start_time = 0.0

    # Create chunks for the specified time
    while start_time < video_duration:
        end_time = min(start_time + chunk_duration, video_duration)
        chunks.append((start_time, end_time))
        start_time += chunk_duration
    return chunks

def extract_movie_segment_with_sound(movie_path, start_time, end_time,
    output_path='output_segment.mp4'):
    """
    Extracts a specific segment of a video with sound and saves it.

    Parameters
    ----------
    movie_path : str
        Path to the .mkv movie file.
    start_time : float
        Start time of the segment in seconds.
    end_time : float
        End time of the segment in seconds.
    output_path : str, optional
        Path to save the output segment (default is 'output_segment.mp4').

    """

    # Create movie segment
    movie_segment = VideoFileClip(movie_path).subclip(start_time, end_time)
    print(f"\nWriting movie file from {start_time}s until {end_time}s")

    # Write video file
    movie_segment.write_videofile(output_path, codec="libx264",
        audio_codec="aac", verbose=False, logger=None)
    return output_path


def display_transcript_and_movie(chunk_index, transcript_df, chunks,
    movie_path):
    """
    Displays transcript, movie, onset, and duration for a selected chunk.

    Parameters
    ----------
    chunk_index : int
        Index of the selected chunk.
    transcript_df : DataFrame
        DataFrame containing transcript data.
    chunks : list
        List of (start_time, end_time) tuples for video chunks.
    movie_path : str
        Path to the .mkv movie file.

    """
    # Retrieve the start and end times for the selected chunk
    start_time, end_time = chunks[chunk_index]

    # Get the corresponding transcript row if it exists in the DataFrame
    transcript_chunk = transcript_df.iloc[chunk_index] if chunk_index < len(transcript_df) else None

    # Display the stimulus chunk number
    print(f"\nChunk number: {chunk_index + 1}")

    # Display transcript details if available; otherwise, indicate no dialogue
    if transcript_chunk is not None and pd.notna(transcript_chunk['text_per_tr']):
        print(f"\nText: {transcript_chunk['text_per_tr']}")
        print(f"Words: {transcript_chunk['words_per_tr']}")
        print(f"Onsets: {transcript_chunk.get('onsets_per_tr', 'N/A')}")
        print(f"Durations: {transcript_chunk.get('durations_per_tr', 'N/A')}")
    else:
        print("<No dialogue in this scene>")

    # Extract and display the video segment
    output_movie_path = extract_movie_segment_with_sound(movie_path, start_time,
        end_time)
    display(Video(output_movie_path, embed=True, width=640, height=480))


def create_dropdown_by_text(transcript_df):
    """
    Creates a dropdown widget for selecting chunks by their text.

    Parameters
    ----------
    transcript_df : DataFrame
        DataFrame containing transcript data.

    """

    options = []

    # Iterate over each row in the transcript DataFrame
    for i, row in transcript_df.iterrows():
        if pd.notna(row['text_per_tr']):  # Check if the transcript text is not NaN
            options.append((row['text_per_tr'], i))
        else:
            options.append(("<No dialogue in this scene>", i))
    return widgets.Dropdown(options=options, description='Select scene:')


def interface_display_transcript_and_movie(movie_path, transcript_path):
    """
    Interactive interface to align movie and transcript chunks.

    Parameters
    ----------
    movie_path : str
        Path to the .mkv movie file.
    transcript_path : str
        Path to the transcript file (.tsv).

    """

    # Load the transcript data from the provided path
    transcript_df = load_transcript(transcript_path)

    # Split the video file into chunks of 1.49 seconds
    chunks = split_movie_into_chunks(movie_path)

    # Create a dropdown widget with transcript text as options
    dropdown = create_dropdown_by_text(transcript_df)

    # Create an output widget to display video and transcript details
    output = widgets.Output()

    # Display the dropdown and output widgets
    display(dropdown, output)

    # Define the function to handle dropdown value changes
    def on_chunk_select(change):
        with output:
            output.clear_output()  # Clears previous content
            chunk_index = dropdown.value
            display_transcript_and_movie(chunk_index, transcript_df, chunks,
                movie_path)

    dropdown.observe(on_chunk_select, names='value')

# == Dataloader writen by AI (Use with Caution)
class VideoDataset(VisionDataset):
    """
    Args:
        root (string): Enclosing folder where videos are located on the local machine.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self, root: str, device: str = 'cpu', transforms: Optional[Callable] = None, 
                 transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.device = device

        # Load video file IDs
        self.ids = [f for f in os.listdir(root) if f.endswith('.mkv')]

    def _load_video(self, video_file: str):
        video_path = os.path.join(self.root, video_file)
        video = torchvision.io.read_video(video_path, pts_unit='sec')
        frames = video[0].permute((0, 3, 1, 2))  # Permute to channel-first format
        return frames

    def _load_target(self, video_file: str) -> dict:
        # This function now just returns the video filename as a target
        return {'id': video_file}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        video_file = self.ids[index]
        video = self._load_video(video_file)
        target = self._load_target(video_file)

        if self.transforms:
            video, target = self.transforms(video, target)

        video.to(device=self.device)

        return video, target

    def __len__(self) -> int:
        return len(self.ids)

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
        path (str): Path to the video.

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
    return frames, fps
