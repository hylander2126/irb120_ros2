"""
Bringup for the RealSense cameras: each camera's realsense2_camera driver,
pinned to its serial number so drivers can't grab each other's device, plus
its eye-to-hand static transform (base -> <camera_name>_link).

  ros2 launch irb120_handeye bringup_cams.launch.py                     # all three
  ros2 launch irb120_handeye bringup_cams.launch.py cameras:=cam2       # just cam2
  ros2 launch irb120_handeye bringup_cams.launch.py cameras:=cam1,cam3 profile:=filtered

Cameras are independent: a camera that is missing (not found within
device_timeout) or whose driver dies is reported as an ERROR naming it, and
the others keep running.

Profiles (config/realsense_<file>.yaml, applied to every selected camera):
  raw_depth       1280x720 stereo/RGB at 6 Hz, emitter on, unfiltered aligned depth,
                  point cloud, infra1/infra2 (default; what perception and FFS use)
  filtered        realsense_common.yaml: decimated/spatial/temporal-filtered depth
  calibration     1280x720 color only, depth off (hand-eye calibration)
  offline_stereo  raw_depth's streams without point cloud/aligned depth, for the
                  stationary offline FFS capture; stop other camera drivers first

Verify:
  ros2 topic hz /realsense2/color/image_raw
  ros2 run tf2_ros tf2_echo base realsense2_link
"""
import math

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.logging import get_logger
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# key -> (camera_name, serial, static TF launch). Serials carry literal single
# quotes: rs_launch.py resolves each param through yaml.safe_load, and a bare
# digit string would parse as an int and fail the driver's string-typed
# serial_no parameter.
CAMERAS = {
    "cam1": ("realsense", "'243522072478'", "cam_tf_realsense_1.7px.launch.py"),
    "cam2": ("realsense2", "'750612071219'", "cam_tf_realsense2_2.1px.launch.py"),
    "cam3": ("realsense3", "'213622073793'", "cam_tf_realsense3_1.6px.launch.py"),
}

PROFILES = {
    "raw_depth": "realsense_raw_depth.yaml",
    "filtered": "realsense_common.yaml",
    "calibration": "realsense_calibration.yaml",
    "offline_stereo": "realsense_offline_stereo.yaml",
}


def _report_camera_exit(event, context):
    """Log a loud error when a camera driver exits; leave the other cameras running."""
    if context.is_shutdown:
        return None
    names = {f"/{name}": key for key, (name, _, _) in CAMERAS.items()}
    try:
        node = event.action.node_name
    except (AttributeError, RuntimeError):
        node = ""
    key = names.get(node, "?")
    get_logger("bringup_cams").error(
        f"\n{'=' * 72}\n  CAMERA {key} ({node or event.action.name}) DRIVER EXITED "
        f"(code {event.returncode}).\n  Not found within device_timeout, unplugged, or crashed; "
        f"see its log above.\n  The other cameras keep running.\n{'=' * 72}"
    )
    return None


def _launch_cameras(context):
    handeye_pkg = get_package_share_directory("irb120_handeye")
    rs_launch = f"{get_package_share_directory('realsense2_camera')}/launch/rs_launch.py"

    selected = [c.strip() for c in LaunchConfiguration("cameras").perform(context).split(",") if c.strip()]
    unknown = [c for c in selected if c not in CAMERAS]
    if unknown or not selected:
        raise ValueError(f"cameras must be a comma-separated subset of {list(CAMERAS)}, got {selected}")

    profile = LaunchConfiguration("profile").perform(context)
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {list(PROFILES)}, not {profile!r}")

    clip_distance = LaunchConfiguration("clip_distance").perform(context)
    device_timeout = LaunchConfiguration("device_timeout").perform(context)
    for arg, value in (("clip_distance", clip_distance), ("device_timeout", device_timeout)):
        try:
            valid = math.isfinite(float(value))
        except ValueError:
            valid = False
        if not valid:
            raise ValueError(f"{arg} must be a finite number, e.g. -1.0, not {value!r}")

    actions = []
    for key in selected:
        name, serial, tf_launch = CAMERAS[key]
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(rs_launch),
            launch_arguments={
                "camera_name": name,
                "camera_namespace": "",
                "serial_no": serial,
                "clip_distance": clip_distance,
                "wait_for_device_timeout": device_timeout,
                "config_file": f"{handeye_pkg}/config/{PROFILES[profile]}",
            }.items(),
        ))
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(f"{handeye_pkg}/launch/{tf_launch}")
        ))
    return actions


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument(
            "cameras", default_value="cam1,cam2,cam3",
            description="Comma-separated cameras to start: cam1 (realsense), cam2 (realsense2), "
                        "cam3 (realsense3)."),
        DeclareLaunchArgument(
            "profile", default_value="raw_depth", choices=list(PROFILES),
            description="Driver profile for every selected camera; see this file's docstring."),
        DeclareLaunchArgument(
            "clip_distance", default_value="1.4",
            description="Maximum depth in metres; a negative value disables this software cutoff."),
        DeclareLaunchArgument(
            "device_timeout", default_value="20.0",
            description="Seconds a driver waits for its camera before exiting with an error; "
                        "negative waits forever."),
        RegisterEventHandler(OnProcessExit(
            target_action=lambda action: isinstance(action, Node)
            and action.node_executable == "realsense2_camera_node",
            on_exit=_report_camera_exit,
        )),
        OpaqueFunction(function=_launch_cameras),
    ])
