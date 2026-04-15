from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch.substitutions import (
    Command,
    FindExecutable,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _create_nodes(context):
    # Hardcoded for this robot/cell setup.
    rws_ip = "192.168.125.1"
    rws_port = 80
    rws_username = "ROS"
    rws_password = "robotics"

    calibrate_raw = LaunchConfiguration("calibrate").perform(context).strip().lower()
    calibrate = calibrate_raw in ("1", "true", "yes", "on")
    robot_xacro = "irb120_3_58.xacro" if calibrate else "irb120_with_finger.xacro"

    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [FindPackageShare("irb120_control"), "urdf", robot_xacro]
            ),
            " ",
            "prefix:=\"\" ",
            "use_fake_hardware:=false ",
            "fake_sensor_commands:=false ",
            f"rws_ip:={rws_ip} ",
            f"rws_port:={rws_port} ",
            f"rws_username:={rws_username} ",
            f"rws_password:={rws_password} ",
            "configure_via_rws:=true ",
        ]
    )
    robot_description = {
        "robot_description": ParameterValue(robot_description_content, value_type=str)
    }

    robot_controllers = PathJoinSubstitution(
        [FindPackageShare("irb120_control"), "config", "irb120_controllers.yaml"]
    )

    control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[robot_description, robot_controllers],
        output="both",
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description],
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "/controller_manager",
        ],
    )

    initial_joint_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_trajectory_controller", "-c", "/controller_manager"],
    )

    rws_client_node = Node(
        package="abb_rws_client",
        executable="rws_client",
        name="rws_client",
        output="screen",
        parameters=[
            {"robot_ip": rws_ip},
            {"robot_port": rws_port},
            {"robot_nickname": "IRB120"},
            {"polling_rate": 5.0},
            {"no_connection_timeout": False},
        ],
    )

    egm_startup_node = Node(
        package="irb120_control",
        executable="egm_handler",
        name="egm_handler_startup",
        output="screen",
        parameters=[
            {"rws_service_prefix": "/rws_client"},
            {"task": "T_ROB1"},
            {"startup_service_timeout_sec": 30.0},
            # Keep EGM session alive longer during handeye planning pauses.
            {"comm_timeout": 120.0},
        ],
    )

    return [
        rws_client_node,
        control_node,
        robot_state_publisher_node,
        joint_state_broadcaster_spawner,
        initial_joint_controller_spawner,
        egm_startup_node,
    ]


def generate_launch_description():

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "calibrate",
                default_value="false",
                description="If true, load the pure robot model (without finger/sensor tooling).",
            ),
            OpaqueFunction(function=_create_nodes),
        ]
    )
