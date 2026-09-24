import os

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from moveit_configs_utils import MoveItConfigsBuilder
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    GroupAction,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_param_builder import ParameterBuilder


def generate_launch_description():
    pkg_share = get_package_share_directory("irb120_control")
    moveit_cfg_pkg = get_package_share_directory("irb120_moveit_config")
    handeye_cfg_pkg = get_package_share_directory("irb120_handeye")
    perception_pkg = get_package_share_directory("irb120_perception")

    moveit_config = (
        MoveItConfigsBuilder("irb120", package_name="irb120_moveit_config")
        .robot_description(
            file_path=os.path.join(pkg_share, "urdf", "irb120_with_finger.xacro")
        )
        .robot_description_semantic(
            file_path=os.path.join(moveit_cfg_pkg, "config", "irb120.srdf.xacro")
        )
        .planning_pipelines(
            pipelines=["ompl"], default_planning_pipeline="ompl"
        )
        .robot_description_kinematics(
            file_path=os.path.join(moveit_cfg_pkg, "config", "kinematics.yaml")
        )
        .trajectory_execution(
            file_path=os.path.join(moveit_cfg_pkg, "config", "moveit_controllers.yaml"),
            moveit_manage_controllers=False,
        )
        .joint_limits(
            file_path=os.path.join(moveit_cfg_pkg, "config", "joint_limits.yaml")
        )
        .to_moveit_configs()
    )

    # EGM relies on whole ABB & RWS stack
    egm_handler_node = Node(
        package="irb120_control",
        executable="egm_handler",
        name="egm_handler_startup",
        output="screen",
        parameters=[
            {"rws_service_prefix": "/rws_client"},
            {"task": "T_ROB1"},
            {"startup_service_timeout_sec": 30.0},
            {"comm_timeout": 120.0},
            # EGM hard-stops after this many seconds no matter what (RobotWare-side
            # CondTime, not something ROS can override once set) — 180s (3 min) is
            # the safe interactive default; override with egm_cond_time:=<seconds>
            # for a long unattended batch (e.g. arc_static_batch) so it doesn't bail
            # mid-run. Kept as a launch arg rather than just raising the default so
            # everyday sessions keep the tighter backstop.
            {"cond_time": ParameterValue(LaunchConfiguration('egm_cond_time'), value_type=float)},
        ],
    )

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict()],
    )

    # Delay RViz by 5s so move_group receives real joint states before RViz
    # initializes the goal marker — prevents the marker snapping to all-zeros.
    rviz_node = TimerAction(
        period=5.0,
        actions=[Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="log",
            arguments=["-d", os.path.join(moveit_cfg_pkg, "rviz", "moveit.rviz")],
            parameters=[moveit_config.to_dict()],
        )],
    )

    ## RealSense Bringup (both cameras and both TFs)
    bringup_cam1 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([handeye_cfg_pkg, "launch", "bringup_cam1.launch.py"])
        )
    )
    bringup_cam2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([handeye_cfg_pkg, "launch", "bringup_cam2.launch.py"])
        )
    )
    bringup_cam3 = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([handeye_cfg_pkg, "launch", "bringup_cam3.launch.py"])
            )
        )

    # rs_launch.py declares every RealSense driver parameter.  A normal nested
    # include makes all of those implementation details appear in this
    # top-level launch's `--show-args` output, and forwards unrelated launch
    # configurations into rs_launch's validation loop.  The camera wrappers
    # already supply every setting they require, so isolate this private
    # implementation scope instead of exposing its arguments as our API.
    camera_bringup_group = GroupAction(
        actions=[bringup_cam1, bringup_cam2, bringup_cam3],
        scoped=True,
        forwarding=False,
    )

    perception_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([perception_pkg, "launch", "perception.launch.py"])
        ),
        launch_arguments={
            'active_at_start': LaunchConfiguration('perception_active_at_start'),
        }.items(),
    )

    # FT sensor nodes (REALLY wants to be run as executable, not as Node)
    net_ft_node = ExecuteProcess(
        cmd=[os.path.join(get_package_prefix("netft_utils"), "lib", "netft_utils", "netft_node",),
             "--address", "192.168.126.125", "--frame_id", "ft_link"],
        output="screen",
    )

    netft_preprocessor_node = Node(
        package="irb120_control",
        executable="netft_preprocessor",
        name="netft_preprocessor",
        output="screen",
    )

    # Just for recording video and saving convex hull — one instance per camera,
    # driven together by irb120_control.util.runtime_log_dir.start_recording()/
    # stop_recording() so cam1 and cam2 always start/stop/switch quality in lockstep.
    camera_hull_recorder_node = Node(
        package="irb120_control",
        executable="camera_hull_recorder",
        name="camera_hull_recorder",
        output="screen",
        parameters=[
            {"image_topic": "/realsense/color/image_raw"},
            {"camera_info_topic": "/realsense/color/camera_info"},
            {"marker_topic": "/object_detector/markers"},
            {"contact_marker_topic": "/contact_point_selector/markers"},
            {"annotated_image_topic": "/contact_point_selector/camera1_overlay"},
            {"recording_service": "/camera_hull_recorder/set_recording"},
            {"filename_prefix": "camera_hull_overlay"},
            {"auto_start_recording": False},
            {"show_hull": True},
            {"show_ft_hud": True},
        ],
    )
    camera_hull_recorder2_node = Node(
        package="irb120_control",
        executable="camera_hull_recorder",
        name="camera_hull_recorder2",
        output="screen",
        parameters=[
            {"image_topic": "/realsense2/color/image_raw"},
            {"camera_info_topic": "/realsense2/color/camera_info"},
            {"marker_topic": "/object_detector/markers"},
            {"contact_marker_topic": "/contact_point_selector/markers"},
            {"annotated_image_topic": "/contact_point_selector/camera2_overlay"},
            {"recording_service": "/camera_hull_recorder2/set_recording"},
            {"filename_prefix": "camera_hull_overlay_cam2"},
            {"auto_start_recording": False},
            {"show_hull": True},
            # No FT HUD on cam2 — this view is likely to get flipped/cropped in
            # post, and the readout would end up mirrored/misplaced or cut off.
            {"show_ft_hud": False},
        ],
    )
    # RQT plotter for netft. This doesn't work half the time.
    viz_netft_node = Node(
        package="rqt_plot",
        executable="rqt_plot",
        name="net_ft_viz",
        output="screen",
        arguments=["/netft_data_monitor/wrench/force/x", "/netft_data_monitor/wrench/force/y", "/netft_data_monitor/wrench/force/z"],
    )
    viz_netft_delayed = TimerAction(
        period=5.0,
        actions=[viz_netft_node],
    )

    # Servo nodes for keyboard jogging AND for press-and-pull velocity control.
    servo_params = {
        "moveit_servo": ParameterBuilder("irb120_moveit_config")
        .yaml("config/servo.yaml")
        .to_dict()
    }
    servo_node = Node(
        package='moveit_servo',
        executable='servo_node',
        name='servo_node',
        output='screen',
        parameters=[
            moveit_config.to_dict(),
            servo_params,
            {"update_period": 0.02},
            {"planning_group_name": "manipulator"},
        ],
        condition=IfCondition(LaunchConfiguration('start_servo')),
    )
    servo_set_twist_mode = TimerAction(
        period=3.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    'ros2', 'service', 'call',
                    '/servo_node/switch_command_type',
                    'moveit_msgs/srv/ServoCommandType',
                    '{command_type: 1}',
                ],
                output='screen',
            )
        ],
        condition=IfCondition(LaunchConfiguration('start_servo')),
    )

    # Declare the launch arguments

    egm_cond_time_arg = DeclareLaunchArgument(
        'egm_cond_time',
        default_value='180.0',
        description=(
            'EGM CondTime in seconds (RobotWare-side hard stop — see egm_handler_node '
            'comment). 180s (3 min) default is the safe interactive value; raise it for '
            'a long unattended batch (e.g. arc_static_batch) so EGM does not bail mid-run: '
            'egm_cond_time:=1200.0'
        ),
    )
    start_servo_arg = DeclareLaunchArgument(
        'start_servo',
        default_value='true',
        description=(
            'Start MoveIt Servo for arrow-key Cartesian jogging. '
            'Then run keyboard_jog in a second terminal: ros2 run irb120_control keyboard_jog. '
            'Arrow keys: ↑/↓ = +Z/-Z,  ←/→ = -X/+X.'
        ),
    )
    perception_active_at_start_arg = DeclareLaunchArgument(
        'perception_active_at_start',
        default_value='true',
        description=(
            'Whether object_detector starts out processing '
            'immediately (default) or idle until the first check_press_point() '
            'call activates them — see irb120_perception/perception.launch.py.'
        ),
    )

    return LaunchDescription([
        perception_active_at_start_arg,
        egm_cond_time_arg,
        start_servo_arg,

        egm_handler_node,

        move_group_node,
        rviz_node,
        camera_bringup_group,
        perception_launch,
        net_ft_node,
        netft_preprocessor_node,
        camera_hull_recorder_node,
        camera_hull_recorder2_node,
        # viz_netft_delayed,
        servo_node,
        servo_set_twist_mode,
    ])
