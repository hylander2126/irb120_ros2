"""Static transform acquired via run_handeye_calibration.py (headless).

EYE-TO-HAND: base_link -> realsense_link. Solved with cv2.calibrateHandEye
(method=park), 16 samples, mean pairwise AX=XB translation
residual ~4.5mm (algebraic solve-consistency metric, not the
MoveIt panel's pixel reprojection error -- see run_handeye_calibration.py).

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
                "--child-frame-id", "realsense_link",
                "--x", "0.056033",
                "--y", "-0.229916",
                "--z", "0.242406",
                "--qx", "-0.008187",
                "--qy", "0.079331",
                "--qz", "0.198977",
                "--qw", "0.976754",
            ],
        ),
    ]
    return LaunchDescription(nodes)
