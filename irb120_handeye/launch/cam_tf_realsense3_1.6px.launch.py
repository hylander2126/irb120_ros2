"""Static transform acquired via run_handeye_calibration.py (headless).

EYE-TO-HAND: base_link -> realsense3_link. Seeded with cv2.calibrateHandEye
(method=park), then refined by minimizing board-corner reprojection
error; 19 samples, RMS reprojection error 1.58 px
(~1.15 mm at the board).

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
                "--child-frame-id", "realsense3_link",
                "--x", "1.196439",
                "--y", "0.001461",
                "--z", "0.113988",
                "--qx", "-0.008602",
                "--qy", "-0.021340",
                "--qz", "0.999444",
                "--qw", "0.024115",
            ],
        ),
    ]
    return LaunchDescription(nodes)
