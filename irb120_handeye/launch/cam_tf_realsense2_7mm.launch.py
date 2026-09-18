"""Static transform acquired via run_handeye_calibration.py (headless).

EYE-TO-HAND: base_link -> realsense2_link. Solved with cv2.calibrateHandEye
(method=park), 16 samples, mean pairwise AX=XB translation
residual ~7.1mm (algebraic solve-consistency metric, not the
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
                "--child-frame-id", "realsense2_link",
                "--x", "0.081169",
                "--y", "0.265997",
                "--z", "0.504083",
                "--qx", "0.653305",
                "--qy", "0.016000",
                "--qz", "-0.297268",
                "--qw", "0.696110",
            ],
        ),
    ]
    return LaunchDescription(nodes)
