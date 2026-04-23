from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node


RWS_IP = "192.168.125.1"
RWS_PORT = "80"


def generate_launch_description():

    # Use Picknik's abb_control launch for the hardware stack, but:
    #  - pass our controllers yaml and URDF
    #  - disable their JTC spawner (initial_joint_controller:=none) so we
    #    gate it ourselves after EGM is live
    #  - disable RViz (we don't need it for this test)
    abb_control = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [get_package_share_directory("abb_bringup"), "launch", "abb_control.launch.py"]
            )
        ),
        launch_arguments={
            "description_package": "irb120_control",
            "description_file": "irb120_with_finger.xacro",
            "runtime_config_package": "irb120_control",
            "controllers_file": "irb120_controllers.yaml",
            "moveit_config_package": "irb120_moveit_config",
            "rws_ip": RWS_IP,
            "rws_port": RWS_PORT,
            "use_fake_hardware": "false",
            "configure_via_rws": "true",
            "launch_rviz": "false",
        }.items(),
    )

    # egm_startup_node = Node(
    #     package="irb120_control",
    #     executable="egm_handler",
    #     name="egm_handler_startup",
    #     output="screen",
    #     parameters=[
    #         {"rws_service_prefix": "/rws_client"},
    #         {"task": "T_ROB1"},
    #         {"startup_service_timeout_sec": 30.0},
    #         {"comm_timeout": 5.0},
    #     ],
    # )


    return LaunchDescription([
        abb_control,
        # egm_startup_node,
    ])
