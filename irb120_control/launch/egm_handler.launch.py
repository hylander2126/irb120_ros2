"""EGM startup handler, shared by bringup_stack.launch.py and irb120_handeye's
bringup_handeye.launch.py. Relies on the whole ABB & RWS stack (abb_rws.launch.py +
abb_control.launch.py)."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    egm_cond_time_arg = DeclareLaunchArgument(
        'egm_cond_time',
        default_value='360.0',
        description=(
            'EGM CondTime in seconds (RobotWare-side hard stop — see egm_handler.launch.py). '
            'Raise it for a long unattended batch (e.g. arc_static_batch) so EGM does not '
            'bail mid-run: egm_cond_time:=1200.0'
        ),
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
            {"comm_timeout": 120.0},
            # EGM hard-stops after this many seconds no matter what (RobotWare-side
            # CondTime, not something ROS can override once set). Kept as a launch arg
            # rather than just raising the default so everyday sessions keep a tighter
            # backstop.
            {"cond_time": ParameterValue(LaunchConfiguration('egm_cond_time'), value_type=float)},
        ],
    )

    return LaunchDescription([egm_cond_time_arg, egm_handler_node])
