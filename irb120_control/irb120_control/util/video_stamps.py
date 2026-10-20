"""Per-frame timestamp sidecars for recorded videos (no ROS imports; used offline too).

camera_hull_recorder writes <video stem>.frames.csv next to each video:
    frame,stamp_s,receive_s
stamp_s is the camera image header stamp, receive_s the recorder's wall clock.
"""
from pathlib import Path

import numpy as np


def frame_stamps_path(video) -> Path:
    video = Path(video)
    return video.with_name(video.stem + ".frames.csv")


def load_frame_stamps(video):
    """(frame index array, header stamp array) or None when the video has no sidecar."""
    path = frame_stamps_path(video)
    if not path.exists():
        return None
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if len(data) == 0:
        return None
    return data[:, 0].astype(int), data[:, 1]
