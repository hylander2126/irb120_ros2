"""
Standalone bringup for the first RealSense camera.

Starts only:
  - The realsense2_camera driver, camera_name="realsense", pinned to its
    serial number so it can't grab the D435 #2 (or vice versa) when both are
    connected at once.
  - The eye-to-hand static transform (base -> realsense_link) already
    solved in camera_1_tf.launch.py.

Deliberately standalone — not wired into bringup_stack.launch.py, RViz, or
perception. Basic ROS-level functioning only:

  ros2 launch irb120_handeye bringup_cam1.launch.py

Verify it's alive:
  ros2 topic list | grep realsense
  ros2 topic hz /realsense/color/image_raw
  ros2 run tf2_ros tf2_echo base realsense_link
"""
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution

# rs_launch.py resolves each param through yaml.safe_load; a bare digit
# string would parse as an int and fail the driver's string-typed
# serial_no parameter, hence the literal single quotes inside the string.
CAM1_SERIAL = "'243522072478'"


def generate_launch_description() -> LaunchDescription:
    realsense_common_yaml = PathJoinSubstitution(
        [get_package_share_directory("irb120_handeye"), "config", "realsense_common.yaml"]
    )

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [get_package_share_directory("realsense2_camera"), "launch", "rs_launch.py"]
            )
        ),
        launch_arguments={
            # Only what differs per camera lives here — everything shared
            # (streams, filters) is in realsense_common.yaml.
            "camera_name": "realsense",
            "camera_namespace": "",
            "serial_no": CAM1_SERIAL,
            "clip_distance": "1.4",
            "config_file": realsense_common_yaml,
        }.items(),
    )

    cam1_tf = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [get_package_share_directory("irb120_handeye"), "launch", "camera_1_tf.launch.py"]
            )
        )
    )

    return LaunchDescription([
        realsense_launch,
        cam1_tf,
    ])
