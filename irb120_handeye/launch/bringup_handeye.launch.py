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
                [get_package_share_directory("irb120_control"), "launch", "test_bringup.launch.py"]
            )
        )
    )

    moveit_cfg_pkg = get_package_share_directory("irb120_moveit_config")

    moveit_config = (
        MoveItConfigsBuilder("irb120", package_name="irb120_moveit_config")
        .robot_description(
            file_path=os.path.join(
                get_package_share_directory("irb120_control"),
                "urdf",
                "irb120_with_finger.xacro",
            )
        )
        .robot_description_semantic(
            file_path=os.path.join(moveit_cfg_pkg, "config", "irb120_handeye.srdf.xacro")
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

    handeye_rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2_handeye",
        output="log",
        arguments=["-d", os.path.join(moveit_cfg_pkg, "rviz", "moveit_handeye.rviz")],
        parameters=[moveit_config.to_dict()],
    )

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("realsense2_camera"), "launch", "rs_launch.py"])
        ),
        launch_arguments={
            "camera_name": "realsense",
            "camera_namespace": "",
            "pointcloud.enable": "true",
            "colorizer.enable": "false",
            "depth_module.depth_profile": "848x480x30",
            "rgb_camera.color_profile": "640x480x30",
            "decimation_filter.enable": "false",
            "spatial_filter.enable": "false",
            "temporal_filter.enable": "false",
            "clip_distance": "2.2",
        }.items(),
    )

    return LaunchDescription([
        bringup_launch,
        move_group_node,
        handeye_rviz_node,
        realsense_launch,
    ])
