"""Static transform acquired via run_handeye_calibration.py (headless).

EYE-TO-HAND: base_link -> realsense2_link. Seeded with cv2.calibrateHandEye
(method=park), then refined by minimizing board-corner reprojection
error; 18 samples, RMS reprojection error 2.12 px
(~1.60 mm at the board).

Not wired into bringup automatically. Point the matching
bringup_camN.launch.py's cameraN_tf.launch.py include at this file (or copy
the values in) once you trust the result, and keep the previous
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
                "--x", "0.630145",
                "--y", "0.326873",
                "--z", "0.795220",
                "--qx", "0.348765",
                "--qy", "0.335450",
                "--qz", "-0.618309",
                "--qw", "0.619298",
            ],
        ),
    ]
    return LaunchDescription(nodes)
