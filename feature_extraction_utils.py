import os
import torchvision
from typing import Optional, Callable, Tuple, Any
from torchvision.datasets.vision import VisionDataset

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
