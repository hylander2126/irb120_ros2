""" Static transform publisher acquired via MoveIt 2 hand-eye calibration """
""" EYE-TO-HAND: base_link -> realsense_link """
from launch import LaunchDescription
from launch_ros.actions import Node

## This was from handeye calibration setting sensor to realsense_link, NOT camera_color_optical_frame.
# Allows for a direct transform from base_link to realsense_link, required for the tf tree 

def generate_launch_description() -> LaunchDescription:
    nodes = [
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            output="log",
            arguments=[
                "--frame-id",
                "base_link",
                "--child-frame-id",
                "realsense_link",
                "--x",
                "-0.167995",
                "--y",
                "-0.504039",
                "--z",
                "0.396781",
                "--qx",
                "-0.0427951",
                "--qy",
                "0.0854905",
                "--qz",
                "0.288532",
                "--qw",
                "0.952685",
                # "--roll",
                # "3.00906",
                # "--pitch",
                # "3.00295",
                # "--yaw",
                # "-2.54422",
            ],
        ),
    ]
    return LaunchDescription(nodes)
