from pathlib import Path
from typing import Generator
from PIL import Image

import av


def read_video(
    path: str | Path,
    interval: float = 1.0,
    offset: float = 0.0,
) -> Generator[tuple[float, Image.Image], None, None]:
    container = av.open(path)
    video_stream = next(s for s in container.streams if s.type == "video")

    time_base = video_stream.time_base  # Timestamps are in terms of time_base units
    interval_pts = interval / time_base

    idx = 0
    for frame in container.decode(video_stream):
        if frame.pts and frame.pts >= (idx + offset) * interval_pts:
            idx += 1
            timestamp = float(frame.pts * time_base)
            img = frame.to_image()
            yield timestamp, img
