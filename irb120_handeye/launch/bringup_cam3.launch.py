"""
Standalone bringup for the third RealSense camera.

Starts only:
  - The realsense2_camera driver, camera_name="realsense3", pinned to its
    serial number so it can't grab either other D435 (or vice versa) when
    all three are connected at once.
  - The eye-to-hand static transform (base -> realsense3_link) already
    solved in cam_tf_realsense3_7mm.launch.py.

Deliberately standalone — not wired into RViz or perception. Basic
ROS-level functioning only:

  ros2 launch irb120_handeye bringup_cam3.launch.py

Verify it's alive:
  ros2 topic list | grep realsense3
  ros2 topic hz /realsense3/color/image_raw
  ros2 run tf2_ros tf2_echo base realsense3_link
"""
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import IfElseSubstitution, LaunchConfiguration, PathJoinSubstitution

# rs_launch.py resolves each param through yaml.safe_load; a bare digit
# string would parse as an int and fail the driver's string-typed
# serial_no parameter, hence the literal single quotes inside the string.
CAM3_SERIAL = "'213622073793'"


def generate_launch_description() -> LaunchDescription:
    calibration_arg = DeclareLaunchArgument(
        "calibration",
        default_value="false",
        description="Use the hand-eye calibration profile (1280x720 color, depth off).",
    )
    realsense_yaml = PathJoinSubstitution([
        get_package_share_directory("irb120_handeye"), "config",
        IfElseSubstitution(
            LaunchConfiguration("calibration"),
            "realsense_calibration.yaml",
            "realsense_common.yaml",
        ),
    ])

    realsense3_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [get_package_share_directory("realsense2_camera"), "launch", "rs_launch.py"]
            )
        ),
        launch_arguments={
            # Only what differs per camera lives here — everything shared
            # (streams, filters) is in realsense_common.yaml, or
            # realsense_calibration.yaml under calibration:=true.
            "camera_name": "realsense3",
            "camera_namespace": "",
            "serial_no": CAM3_SERIAL,
            "clip_distance": "1.4",
            "config_file": realsense_yaml,
        }.items(),
    )

    cam3_tf = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [get_package_share_directory("irb120_handeye"), "launch", "cam_tf_realsense3_9mm.launch.py"]
            )
        )
    )

    return LaunchDescription([
        calibration_arg,
        realsense3_launch,
        cam3_tf,
    ])
