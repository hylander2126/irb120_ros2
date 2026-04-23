"""Static transform publisher acquired via MoveIt 2 hand-eye calibration.

EYE-TO-HAND: base_link -> realsense_link (12mm lens result).
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            output="log",
            arguments=[
                "--frame-id", "base_link",
                "--child-frame-id", "realsense_link",
                "--x", "-0.167995",
                "--y", "-0.504039",
                "--z", "0.396781",
                "--qx", "-0.0427951",
                "--qy", "0.0854905",
                "--qz", "0.288532",
                "--qw", "0.952685",
            ],
        ),
    ])
