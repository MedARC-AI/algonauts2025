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
