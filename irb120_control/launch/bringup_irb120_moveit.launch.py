import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction, ExecuteProcess
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_param_builder import ParameterBuilder


def generate_launch_description():
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

    keyboard_jog_arg = DeclareLaunchArgument(
        'keyboard_jog',
        default_value='false',
        description=(
            'Start MoveIt Servo for arrow-key Cartesian jogging. '
            'Then run keyboard_jog in a second terminal: ros2 run irb120_control keyboard_jog. '
            'Arrow keys: ↑/↓ = +Z/-Z,  ←/→ = -X/+X.'
        ),
    )

    # Bringup launch here
    bringup_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [get_package_share_directory("irb120_control"), "launch", "bringup_irb120.launch.py"]
            )
        )
    )

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

    # Realsense arguments for pointcloud octomap updates
    realsense_args = {
        "camera_name": "realsense",
        "camera_namespace": "",
        "pointcloud.enable": "true",
        "colorizer.enable": "false",
        # Match depth and color resolution so aligned depth needs no scaling artefacts
        "depth_module.depth_profile": "640x480x30",
        "align_depth.enable": "true",  # Align the depth image to the color (same shape)
        "rgb_camera.color_profile": "640x480x30",
        # Decimation averages NxN depth pixels before output — reduces noise, no meaningful lag
        "decimation_filter.enable": "true",
        "decimation_filter.filter_magnitude": "2",  # 2x2 average → 320x240 effective depth
        "spatial_filter.enable": "false",  # lag concern: handled in software with median blur instead
        "temporal_filter.enable": "false",  # lag concern: handled in software with EMA smoothing instead
        "clip_distance": "2.2",
    }

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([get_package_share_directory("realsense2_camera"), "launch", "rs_launch.py"])
        ),
        launch_arguments=realsense_args.items(),
    )

    handeye_to_realsense_tf = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [get_package_share_directory("irb120_control"), "handeye_calibrations", "cam_tf_12mm.launch.py"]
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
            'perception_method':            LaunchConfiguration('perception_method'),
            'debug_perception':  LaunchConfiguration('debug_perception'),
        }.items(),
    )

    # ---- MoveIt Servo node — only started when keyboard_jog:=true -----------
    # ParameterBuilder wraps the flat yaml under the "moveit_servo" namespace,
    # which is exactly how servo_node expects to find its parameters.
    servo_params = {
        "moveit_servo": ParameterBuilder("irb120_moveit_config")
        .yaml("config/servo.yaml")
        .to_dict()
    }
    acceleration_filter_update_period = {"update_period": 0.02}
    planning_group_name = {"planning_group_name": "manipulator"}

    servo_node = Node(
        package='moveit_servo',
        executable='servo_node',
        name='servo_node',
        output='screen',
        parameters=[
            moveit_config.to_dict(),
            servo_params,
            acceleration_filter_update_period,
            planning_group_name,
        ],
        condition=IfCondition(LaunchConfiguration('keyboard_jog')),
    )

    # Switch servo to TWIST mode (command_type=1) once servo_node is ready.
    # 3-second delay is enough for servo_node to finish initializing.
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
        condition=IfCondition(LaunchConfiguration('keyboard_jog')),
    )

    return LaunchDescription(
        [
            perception_method_arg,
            debug_perception_arg,
            keyboard_jog_arg,
            bringup_launch,
            move_group_node,
            rviz_node,
            rviz_debug_node,
            realsense_launch,
            handeye_to_realsense_tf,
            perception_launch,
            servo_node,
            servo_set_twist_mode,
        ]
    )
