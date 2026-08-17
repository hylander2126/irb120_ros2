"""
Standalone bringup for the second RealSense camera.

Starts only:
  - The realsense2_camera driver, camera_name="realsense2", pinned to its
    serial number so it can't grab the D435 (or vice versa) when both are
    connected at once.
  - The eye-to-hand static transform (base -> realsense2_link) already
    solved in cam2_tf.launch.py.

Deliberately standalone — not wired into bringup_stack.launch.py, RViz, or
perception. Basic ROS-level functioning only:

  ros2 launch irb120_handeye bringup_cam2.launch.py

Verify it's alive:
  ros2 topic list | grep realsense2
  ros2 topic hz /realsense2/color/image_raw
  ros2 run tf2_ros tf2_echo base realsense2_link
"""
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution

# rs_launch.py resolves each param through yaml.safe_load; a bare digit
# string would parse as an int and fail the driver's string-typed
# serial_no parameter, hence the literal single quotes inside the string.
CAM2_SERIAL = "'750612071219'"


def generate_launch_description() -> LaunchDescription:
    realsense2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [get_package_share_directory("realsense2_camera"), "launch", "rs_launch.py"]
            )
        ),
        launch_arguments={
            "camera_name": "realsense2",
            "camera_namespace": "",
            "serial_no": CAM2_SERIAL,
            "pointcloud.enable": "true",
            "align_depth.enable": "true",
            # "clip_distance": "1.1",
        }.items(),
    )

    cam2_tf = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [get_package_share_directory("irb120_handeye"), "launch", "camera_2_tf.launch.py"]
            )
        )
    )

    return LaunchDescription([
        realsense2_launch,
        cam2_tf,
    ])
