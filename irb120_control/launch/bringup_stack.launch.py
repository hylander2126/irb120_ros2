import os

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
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

    urdf_path = os.path.join(get_package_share_directory("irb120_models"), "urdf", "irb120_with_finger.xacro")
    # Experiments only run with the finger; the ChArUco calibration model
    # (irb120_with_charuco.xacro) is loaded by irb120_handeye's bringup_handeye.launch.py.

    moveit_config = (
        MoveItConfigsBuilder("irb120", package_name="irb120_moveit_config")
        .robot_description(file_path=urdf_path)
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

    egm_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_share, "launch", "egm_handler.launch.py"])
        )
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

    ## RealSense Bringup (all cameras and their TFs)
    # rs_launch.py declares every RealSense driver parameter.  A normal nested
    # include makes all of those implementation details appear in this
    # top-level launch's `--show-args` output, and forwards unrelated launch
    # configurations into rs_launch's validation loop, so isolate this private
    # implementation scope instead of exposing its arguments as our API.
    # No calibration mode here (see irb120_handeye's bringup_handeye.launch.py for that).
    camera_bringup_group = GroupAction(
        actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([handeye_cfg_pkg, "launch", "bringup_cams.launch.py"])
            )
        )],
        scoped=True,
        forwarding=False,
        # Evaluated before the scope is cleared; launch_arguments on the include would not be.
        launch_configurations={
            "cameras": LaunchConfiguration("cameras"),
            "profile": LaunchConfiguration("camera_profile"),
        },
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
        parameters=[{"calibration_file": LaunchConfiguration("ft_calibration_file")}],
        output="screen",
    )

    # Just for recording video and saving convex hull — one instance per camera,
    # driven together by irb120_control.util.runtime_log_dir.start_recording()/
    # stop_recording() so all three cameras always start/stop/switch quality in lockstep.
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
    camera_hull_recorder3_node = Node(
        package="irb120_control",
        executable="camera_hull_recorder",
        name="camera_hull_recorder3",
        output="screen",
        parameters=[
            {"image_topic": "/realsense3/color/image_raw"},
            {"camera_info_topic": "/realsense3/color/camera_info"},
            {"marker_topic": "/object_detector/markers"},
            {"contact_marker_topic": "/contact_point_selector/markers"},
            {"annotated_image_topic": "/contact_point_selector/camera3_overlay"},
            {"recording_service": "/camera_hull_recorder3/set_recording"},
            {"filename_prefix": "camera_hull_overlay_cam3"},
            {"auto_start_recording": False},
            {"show_hull": True},
            # Far-side view, faces the robot: same reasoning as cam2 for no FT HUD.
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

    cameras_arg = DeclareLaunchArgument(
        "cameras",
        default_value="cam1,cam2,cam3",
        description="Cameras to start (see irb120_handeye/launch/bringup_cams.launch.py).",
    )
    camera_profile_arg = DeclareLaunchArgument(
        "camera_profile",
        default_value="raw_depth",
        choices=["raw_depth", "filtered"],
        description="raw_depth: unfiltered depth + IR (default); filtered: post-processed depth.",
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
        default_value='false',
        description=(
            'Whether object_detector starts out processing '
            'immediately or idle (default) until the first perception snapshot '
            'switches it on — see irb120_perception/perception.launch.py.'
        ),
    )
    digital_twin_rviz_arg = DeclareLaunchArgument(
        'launch_digital_twin_rviz', default_value='false',
        description='Open a separate RViz window for the live editable object twin.',
    )
    digital_twin_rviz = TimerAction(
        period=7.0,
        actions=[Node(
            package='rviz2', executable='rviz2', name='object_digital_twin_rviz', output='log',
            arguments=['-d', os.path.join(perception_pkg, 'rviz', 'object_digital_twin.rviz')],
        )],
        condition=IfCondition(LaunchConfiguration('launch_digital_twin_rviz')),
    )

    return LaunchDescription([
        DeclareLaunchArgument("ft_calibration_file", default_value="",
                              description="Reviewed F/T calibration JSON; empty uses legacy constants"),
        perception_active_at_start_arg,
        cameras_arg,
        camera_profile_arg,
        start_servo_arg,
        digital_twin_rviz_arg,

        egm_launch,

        move_group_node,
        rviz_node,
        camera_bringup_group,
        perception_launch,
        digital_twin_rviz,
        net_ft_node,
        netft_preprocessor_node,
        camera_hull_recorder_node,
        camera_hull_recorder2_node,
        camera_hull_recorder3_node,
        # viz_netft_delayed,
        servo_node,
        servo_set_twist_mode,
    ])
