"""
For testing BundleSDF and other mesh reconstruction algorithms. Record depth stream from two
calibrated cameras simultaneously.

Standalone — does NOT bring up the robot arm, robot_state_publisher, or MoveIt. The eye-to-hand
calibrations (camera_1_tf.launch.py / camera_2_tf.launch.py) are self-contained
tf2_ros static_transform_publisher nodes: they broadcast base -> realsense_link and
base -> realsense2_link straight onto /tf_static without depending on anything else already
publishing "base". No URDF or joint states are needed for that frame to exist — TF doesn't
require a parent frame to be "real"; it just becomes the implicit root of that tree.

Workflow:
  1. Cameras, TF, and an RViz window (both color images + both pointclouds, so you can check
     framing/registration before committing anything to disk) come up immediately.
  2. The launch terminal then prints a prompt and waits for Enter before starting the bag.
     Recording is NOT gated behind an ExecuteProcess/`read`: ros2 launch always wires a launched
     process's stdin to a pipe it privately controls (see launch.actions.ExecuteProcess docs,
     the ProcessStdin event) — it is never the real terminal — so a `read -p` inside one would
     hit EOF immediately and never actually wait. Instead a background thread calls Python's
     own input() against this launch process's real stdin, then starts `ros2 bag record` as a
     plain subprocess. That subprocess inherits this process's process group, so it is not one
     of ros2 launch's managed children — it sits in the terminal's foreground group like any
     plain command, and Ctrl-C reaches it directly and simultaneously with everything else.
  3. Ctrl-C at any point (before or after Enter) stops recording (if started) and tears down
     the whole launch tree exactly as usual.

Captures, per camera: color image, depth aligned to color, both cameras' camera_info
(intrinsics), and /tf_static (extrinsics — base->camera plus each driver's internal
color/depth-optical-frame statics). Raw pointclouds are deliberately excluded from the bag:
they're fully recoverable later from depth + camera_info + extrinsics and would roughly double
the recording size for information already captured (RViz still shows them live, unrecorded).

  ros2 launch irb120_perception record_depth_both.launch.py
  ros2 launch irb120_perception record_depth_both.launch.py bag_dir:=/data/captures
"""
import datetime
import os
import subprocess
import threading

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

# rs_launch.py resolves each param through yaml.safe_load; a bare digit
# string would parse as an int and fail the driver's string-typed
# serial_no parameter, hence the literal single quotes inside the string.
CAM1_SERIAL = "'243522072478'"
CAM2_SERIAL = "'750612071219'"

# What BundleSDF actually needs: raw color + depth aligned to color (so pixel (u,v) means the
# same thing in both), the intrinsics for each, and the static extrinsics tying the two cameras
# (and their own color/depth optical frames) into one tree.
BAG_TOPICS = [
    "/realsense/color/image_raw",
    "/realsense/color/camera_info",
    "/realsense/depth/image_rect_raw",
    "/realsense/depth/camera_info",
    "/realsense/aligned_depth_to_color/image_raw",
    "/realsense/aligned_depth_to_color/camera_info",
    "/realsense2/color/image_raw",
    "/realsense2/color/camera_info",
    "/realsense2/depth/image_rect_raw",
    "/realsense2/depth/camera_info",
    "/realsense2/aligned_depth_to_color/image_raw",
    "/realsense2/aligned_depth_to_color/camera_info",
    "/tf_static",
]


def _wait_for_enter_then_record(context, *args, **kwargs):
    bag_path = os.path.join(
        LaunchConfiguration("bag_dir").perform(context),
        "record_depth_both_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
    )

    def worker():
        input("\n>>> Cameras + RViz are up. Press Enter to start recording "
              "(Ctrl-C to exit without recording)...\n")
        print(f">>> Recording to {bag_path}", flush=True)
        # Deliberately plain subprocess.Popen, not launch.actions.ExecuteProcess — this way it
        # inherits our process group instead of being spawned into its own session, so it's a
        # normal foreground job like anything else you'd Ctrl-C in this terminal.
        subprocess.Popen([
            "ros2", "bag", "record",
            "--storage", "mcap",
            "-o", bag_path,
            *BAG_TOPICS,
        ])

    threading.Thread(target=worker, daemon=True).start()
    return []


def generate_launch_description() -> LaunchDescription:
    bag_dir_arg = DeclareLaunchArgument(
        "bag_dir",
        default_value=os.path.expanduser("~/bundlesdf_recordings"),
        description="Directory the timestamped bag folder is created under.",
    )

    realsense_common_yaml = PathJoinSubstitution(
        [get_package_share_directory("irb120_handeye"), "config", "realsense_common.yaml"]
    )

    realsense1_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [get_package_share_directory("realsense2_camera"), "launch", "rs_launch.py"]
            )
        ),
        launch_arguments={
            "camera_name": "realsense",
            "camera_namespace": "",
            "serial_no": CAM1_SERIAL,
            "clip_distance": "1.4",
            "config_file": realsense_common_yaml,
        }.items(),
    )
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
    cam2_tf = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [get_package_share_directory("irb120_handeye"), "launch", "camera_2_tf.launch.py"]
            )
        )
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2_record_depth_both",
        output="log",
        arguments=["-d", os.path.join(
            get_package_share_directory("irb120_perception"), "rviz", "record_depth_both.rviz"
        )],
    )

    return LaunchDescription([
        bag_dir_arg,
        realsense1_launch,
        realsense2_launch,
        cam1_tf,
        cam2_tf,
        rviz_node,
        OpaqueFunction(function=_wait_for_enter_then_record),
    ])
