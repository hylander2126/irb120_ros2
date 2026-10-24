"""
Bringup for ALL RealSense cameras (currently 3 as of 2026-10-23).
"""
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution


def generate_launch_description() -> LaunchDescription:
    realsense_1_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [get_package_share_directory("irb120_handeye"), "launch", "bringup_cam1.launch.py"]
            )
        ),
    )
    realsense_2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [get_package_share_directory("irb120_handeye"), "launch", "bringup_cam2.launch.py"]
            )
        ),
    )
    realsense_3_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [get_package_share_directory("irb120_handeye"), "launch", "bringup_cam3.launch.py"]
            )
        ),
    )


    return LaunchDescription([
        realsense_1_launch,
        realsense_2_launch,
        realsense_3_launch,
    ])
