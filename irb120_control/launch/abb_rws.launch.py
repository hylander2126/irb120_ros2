from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    rws_launch_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [get_package_share_directory("abb_bringup"), "launch", "abb_rws_client.launch.py"]
            )
        ),
        launch_arguments={"robot_ip": "192.168.125.1"}.items(),
    )

    return LaunchDescription([
        rws_launch_node,
    ])