from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Matches ROS1 top-level order: bringup first, then EGM init handler.
    bringup_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("irb120_control"), "launch", "bringup_irb120_vel.launch.py"]
            )
        )
    )

    egm_startup_node = Node(
        package="irb120_control",
        executable="egm_handler",
        name="egm_handler_startup",
        output="screen",
        parameters=[
            {"rws_service_prefix": "/rws_client"},
            {"task": "T_ROB1"},
            {"stop_egm_after_startup": True},
            {"one_shot": True},
            {"startup_service_timeout_sec": 30.0},
            {"shutdown_on_exit": False},
        ],
    )

    return LaunchDescription([bringup_launch, egm_startup_node])
