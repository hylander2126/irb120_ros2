"""Static transform acquired via run_handeye_calibration.py (headless).

EYE-TO-HAND: base_link -> realsense2_link. Seeded with cv2.calibrateHandEye
(method=park), then refined by minimizing board-corner reprojection
error; 12 samples, RMS reprojection error 1.17 px
(~0.73 mm at the board).

Not wired into bringup automatically. Point this camera's CAMERAS entry in
bringup_cams.launch.py at this file (or copy the values in) once you trust the result, and keep the previous
cam_tf_*.launch.py around as a fallback per this package's README.
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    nodes = [
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            output="log",
            arguments=[
                "--frame-id", "base_link",
                "--child-frame-id", "realsense2_link",
                "--x", "0.631582",
                "--y", "0.326349",
                "--z", "0.797486",
                "--qx", "0.349666",
                "--qy", "0.334732",
                "--qz", "-0.619841",
                "--qw", "0.617645",
            ],
        ),
    ]
    return LaunchDescription(nodes)
