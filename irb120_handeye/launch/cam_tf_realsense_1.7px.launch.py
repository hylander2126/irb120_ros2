"""Static transform acquired via run_handeye_calibration.py (headless).

EYE-TO-HAND: base_link -> realsense_link. Seeded with cv2.calibrateHandEye
(method=park), then refined by minimizing board-corner reprojection
error; 21 samples, RMS reprojection error 1.71 px
(~0.61 mm at the board).

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
                "--x", "0.126471",
                "--y", "0.017224",
                "--z", "0.106371",
                "--qx", "-0.020317",
                "--qy", "0.025509",
                "--qz", "-0.002386",
                "--qw", "0.999465",
            ],
        ),
    ]
    return LaunchDescription(nodes)
