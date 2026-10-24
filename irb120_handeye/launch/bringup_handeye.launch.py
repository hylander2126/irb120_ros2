import os
import sys
import tempfile

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    GroupAction,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


# Sets RSP's robot_description through the SetParameters service directly.
# `ros2 param set/load` re-parse string values as YAML, which breaks on the ": "
# inside URDF comments.
_SET_RSP_DESCRIPTION = """
import os
import sys
import rclpy
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters

rclpy.init()
node = rclpy.create_node("rsp_description_updater")
client = node.create_client(SetParameters, "/robot_state_publisher/set_parameters")
if not client.wait_for_service(timeout_sec=30.0):
    sys.exit("robot_state_publisher not found; did abb_control.launch.py start?")
with open(sys.argv[1]) as f:
    urdf = f.read()
os.remove(sys.argv[1])
value = ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=urdf)
future = client.call_async(SetParameters.Request(
    parameters=[Parameter(name="robot_description", value=value)]))
rclpy.spin_until_future_complete(node, future, timeout_sec=10.0)
result = future.result()
if result is None or not result.results[0].successful:
    sys.exit(f"robot_state_publisher rejected robot_description: {result}")
print("robot_state_publisher robot_description updated")
"""

def _push_robot_description(context, urdf_path):
    """Swap the finger for the ChArUco mount on abb_control's robot_state_publisher.

    abb_control.launch.py (Picknik's abb_bringup) hardcodes its xacro call, so its
    RSP always starts with the finger. RSP accepts robot_description updates at
    runtime, so push the ChArUco model to keep TF and RViz's RobotModel in sync
    with move_group's collision model.
    """
    urdf = xacro.process_file(urdf_path).toxml()
    with tempfile.NamedTemporaryFile("w", prefix="rsp_description_", suffix=".urdf", delete=False) as f:
        f.write(urdf)
    return [ExecuteProcess(
        cmd=[sys.executable, "-c", _SET_RSP_DESCRIPTION, f.name],
        name="rsp_description_updater",
        output="screen",
    )]


def generate_launch_description():
    bringup_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [get_package_share_directory("irb120_control"), "launch", "abb_control.launch.py"]
            )
        )
    )

    # Same EGM startup handler as bringup_stack.launch.py (egm_cond_time:= passes through).
    egm_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [get_package_share_directory("irb120_control"), "launch", "egm_handler.launch.py"]
            )
        )
    )

    moveit_cfg_pkg = get_package_share_directory("irb120_moveit_config")
    handeye_pkg = get_package_share_directory("irb120_handeye")
    urdf_path = os.path.join(get_package_share_directory("irb120_models"), "urdf", "irb120_with_charuco.xacro")

    moveit_config = (
        MoveItConfigsBuilder("irb120", package_name="irb120_moveit_config")
        .robot_description(file_path=urdf_path)
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
        # Calibration-only variant: same as irb120_moveit_config/config/joint_limits.yaml except
        # joint_6 gets the full +/-400 degree range instead of +/-60 -- that +/-60 degree limit protects the FT cable during
        # production motion, which calibration never routes through. See
        # irb120_handeye/config/joint_limits_calibration.yaml for the full rationale. Production
        # launch files (bringup_stack.launch.py, abb_control.launch.py) are untouched and keep
        # using irb120_moveit_config's own joint_limits.yaml, cable protection intact.
        .joint_limits(file_path=os.path.join(handeye_pkg, "config", "joint_limits_calibration.yaml"))
        .to_moveit_configs()
    )

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict()],
    )

    rsp_description_update = OpaqueFunction(
        function=_push_robot_description, args=[urdf_path]
    )

    # Shows all three cameras' color streams.
    handeye_rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2_handeye",
        output="log",
        arguments=["-d", os.path.join(moveit_cfg_pkg, "rviz", "moveit_handeye.rviz")],
        parameters=[moveit_config.to_dict()],
    )

    # Cameras (serial-pinned, with their static TFs) on the calibration profile:
    # 1280x720 color, depth off. Scoped so rs_launch.py's many arguments don't leak
    # into this launch's `--show-args`.
    camera_bringup_group = GroupAction(
        actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([handeye_pkg, "launch", "bringup_cams.launch.py"])
            )
        )],
        scoped=True,
        forwarding=False,
        # Evaluated before the scope is cleared; launch_arguments on the include would not be.
        launch_configurations={"cameras": LaunchConfiguration("cameras"), "profile": "calibration"},
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "cameras", default_value="cam1,cam2,cam3",
            description="Cameras to start, e.g. cam2 to calibrate only cam2 "
                        "(see bringup_cams.launch.py)."),
        bringup_launch,
        egm_launch,
        rsp_description_update,
        move_group_node,
        handeye_rviz_node,
        camera_bringup_group,
    ])
