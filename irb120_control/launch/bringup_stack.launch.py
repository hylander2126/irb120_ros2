import os

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    TimerAction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_param_builder import ParameterBuilder


def generate_launch_description():
    moveit_cfg_pkg = get_package_share_directory("irb120_moveit_config")



    moveit_config = (
        MoveItConfigsBuilder("irb120", package_name="irb120_moveit_config")
        .robot_description(
            file_path=os.path.join(
                get_package_share_directory("irb120_control"),
                "urdf",
                "irb120_with_finger.xacro",
            )
        )
        .robot_description_semantic(
            file_path=os.path.join(moveit_cfg_pkg, "config", "irb120.srdf.xacro")
        )
        .planning_pipelines(pipelines=["ompl"], default_planning_pipeline="ompl")
        .robot_description_kinematics(
            file_path=os.path.join(moveit_cfg_pkg, "config", "kinematics.yaml")
        )
        .trajectory_execution(
            file_path=os.path.join(moveit_cfg_pkg, "config", "moveit_controllers.yaml"),
            moveit_manage_controllers=False,
        )
        .joint_limits(file_path=os.path.join(moveit_cfg_pkg, "config", "joint_limits.yaml"))
        .to_moveit_configs()
    )

    egm_handler_node = Node(
        package="irb120_control",
        executable="egm_handler",
        name="egm_handler_startup",
        output="screen",
        parameters=[
            {"rws_service_prefix": "/rws_client"},
            {"task": "T_ROB1"},
            {"startup_service_timeout_sec": 30.0},
            {"comm_timeout": 5.0},
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
        condition=UnlessCondition(LaunchConfiguration('debug_perception')),
    )

    rviz_debug_node = TimerAction(
        period=5.0,
        actions=[Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="log",
            arguments=["-d", os.path.join(moveit_cfg_pkg, "rviz", "moveit_debug_perception.rviz")],
            parameters=[moveit_config.to_dict()],
        )],
        condition=IfCondition(LaunchConfiguration('debug_perception')),
    )

    # RealSense Bringup
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [get_package_share_directory("realsense2_camera"), "launch", "rs_launch.py"]
            )
        ),
        launch_arguments={
            "camera_name": "realsense",
            "camera_namespace": "",
            "pointcloud.enable": "true",
            "colorizer.enable": "false",
            "depth_module.depth_profile": "640x480x30",
            "align_depth.enable": "true",
            "rgb_camera.color_profile": "640x480x30",
            "decimation_filter.enable": "true",
            "decimation_filter.filter_magnitude": "2",
            "spatial_filter.enable": "false",
            "temporal_filter.enable": "false",
            "clip_distance": "2.2",
        }.items(),
    )


    handeye_to_realsense_tf = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [get_package_share_directory("irb120_handeye"), "launch", "cam_tf_12mm.launch.py"]
            )
        )
    )

    perception_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [get_package_share_directory("irb120_perception"), "launch", "perception.launch.py"]
            )
        ),
        launch_arguments={
            'perception_method': LaunchConfiguration('perception_method'),
            'debug_perception': LaunchConfiguration('debug_perception'),
        }.items(),
    )

    netft_node_exe = os.path.join(
        get_package_prefix("netft_utils"),
        "lib",
        "netft_utils",
        "netft_node",
    )
    net_ft_node = ExecuteProcess(
        cmd=[netft_node_exe, "--address", "192.168.126.125"],
        output="screen",
    )

    netft_preprocessor_node = Node(
        package="irb120_control",
        executable="netft_preprocessor",
        name="netft_preprocessor",
        output="screen",
    )

    camera_hull_recorder_node = Node(
        package="irb120_control",
        executable="camera_hull_recorder",
        name="camera_hull_recorder",
        output="screen",
        parameters=[
            {"image_topic": "/realsense/color/image_raw"},
            {"camera_info_topic": "/realsense/color/camera_info"},
            {"marker_topic": "/object_detector/markers"},
            {"recording_service": "/camera_hull_recorder/set_recording"},
            {"auto_start_recording": False},
        ],
    )

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



    perception_method_arg = DeclareLaunchArgument(
        'perception_method',
        default_value='dbscan',
        description="Perception segmentation backend: 'dbscan' or 'sam'",
    )

    debug_perception_arg = DeclareLaunchArgument(
        'debug_perception',
        default_value='false',
        description=(
            'Launch the perception_debugger node and the debug RViz config. '
            'Trigger a snapshot at runtime with: '
            'ros2 topic pub --once /object_detector/debug_snapshot std_msgs/msg/Empty \'{}\''
        ),
    )

    start_servo_arg = DeclareLaunchArgument(
        'start_servo',
        default_value='false',
        description=(
            'Start MoveIt Servo for arrow-key Cartesian jogging. '
            'Then run keyboard_jog in a second terminal: ros2 run irb120_control keyboard_jog. '
            'Arrow keys: ↑/↓ = +Z/-Z,  ←/→ = -X/+X.'
        ),
    )

    return LaunchDescription([
        perception_method_arg,
        debug_perception_arg,
        start_servo_arg,

        egm_handler_node,

        move_group_node,
        rviz_node,
        rviz_debug_node,
        realsense_launch,
        handeye_to_realsense_tf,
        perception_launch,
        net_ft_node,
        netft_preprocessor_node,
        camera_hull_recorder_node,
        viz_netft_delayed,
        servo_node,
        servo_set_twist_mode,
    ])
