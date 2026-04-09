import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    bringup_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("irb120_control"), "launch", "velocity_ctrl_irb120.launch.py"]
            )
        )
    )

    moveit_cfg_pkg = get_package_share_directory("irb120_moveit_config")

    moveit_config = (
        MoveItConfigsBuilder("abb_irb120_3_58", package_name="irb120_moveit_config")
        .robot_description(
            file_path=os.path.join(
                get_package_share_directory("irb120_control"),
                "urdf",
                "irb120_with_finger.xacro",
            )
        )
        .robot_description_semantic(
            file_path=os.path.join(moveit_cfg_pkg, "config", "irb120.srdf.xacro")
        )
        .planning_pipelines(pipelines=["ompl"], default_planning_pipeline="ompl")
        .robot_description_kinematics(
            file_path=os.path.join(moveit_cfg_pkg, "config", "kinematics.yaml")
        )
        .trajectory_execution(
            file_path=os.path.join(moveit_cfg_pkg, "config", "moveit_controllers.yaml"),
            moveit_manage_controllers=False,
        )
        .joint_limits(file_path=os.path.join(moveit_cfg_pkg, "config", "joint_limits.yaml"))
        .to_moveit_configs()
    )

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict()],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", os.path.join(moveit_cfg_pkg, "rviz", "moveit.rviz")],
        parameters=[moveit_config.to_dict()],
    )

    world_to_base_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_to_base_static_tf",
        output="log",
        arguments=["0", "0", "0", "0", "0", "0", "world", "base_link"],
    )

    return LaunchDescription(
        [
            bringup_launch,
            world_to_base_tf,
            move_group_node,
            rviz_node,
        ]
    )
