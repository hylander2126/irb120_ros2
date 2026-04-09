import os

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _create_nodes(context):
    package_share = get_package_share_directory("irb120_control")
    default_model = os.path.join(package_share, "urdf", "irb120_3_58.xacro")
    model_path = LaunchConfiguration("model").perform(context) or default_model

    if model_path.endswith(".xacro"):
        robot_description_xml = xacro.process_file(model_path).toxml()
    else:
        with open(model_path, "r", encoding="utf-8") as model_file:
            robot_description_xml = model_file.read()

    robot_description = {"robot_description": robot_description_xml}

    return [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="both",
            parameters=[robot_description],
        ),
        Node(
            package="joint_state_publisher_gui",
            executable="joint_state_publisher_gui",
            name="joint_state_publisher_gui",
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            arguments=["-d", os.path.join(package_share, "rviz", "urdf_description.rviz")],
            output="screen",
        ),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "model",
                default_value=os.path.join(
                    get_package_share_directory("irb120_control"), "urdf", "irb120_3_58.xacro"
                ),
                description="Absolute path to robot model (.xacro or .urdf)",
            ),
            OpaqueFunction(function=_create_nodes),
        ]
    )
