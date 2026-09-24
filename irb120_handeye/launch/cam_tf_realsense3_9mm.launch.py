"""Static transform acquired via run_handeye_calibration.py (headless).

EYE-TO-HAND: base_link -> realsense3_link. Solved with cv2.calibrateHandEye
(method=park), 11 samples, mean pairwise AX=XB translation
residual ~9.0mm (algebraic solve-consistency metric, not the
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
                "--child-frame-id", "realsense3_link",
                "--x", "1.068655",
                "--y", "0.019318",
                "--z", "0.129216",
                "--qx", "-0.102305",
                "--qy", "0.001759",
                "--qz", "0.994722",
                "--qw", "-0.007665",
            ],
        ),
    ]
    return LaunchDescription(nodes)
