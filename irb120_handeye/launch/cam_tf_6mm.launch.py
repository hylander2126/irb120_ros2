""" Static transform publisher acquired via MoveIt 2 hand-eye calibration """
""" EYE-TO-HAND: base -> realsense_link """
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    nodes = [
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            output="log",
            arguments=[
                "--frame-id",
                "base",
                "--child-frame-id",
                "realsense_link",
                "--x",
                "0.07271",
                "--y",
                "-0.220391",
                "--z",
                "0.275143",
                "--qx",
                "-0.0293261",
                "--qy",
                "0.157697",
                "--qz",
                "0.222212",
                "--qw",
                "0.961714",
                # "--roll",
                # "3.00902",
                # "--pitch",
                # "2.84707",
                # "--yaw",
                # "-2.66775",
            ],
        ),
    ]
    return LaunchDescription(nodes)
