"""Optional comparison backend: one box-prompted SAM cull of DBSCAN output."""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    checkpoint = DeclareLaunchArgument('sam_checkpoint', default_value='')
    device = DeclareLaunchArgument('sam_device', default_value='cpu')
    node = Node(
        package='irb120_perception', executable='object_detector_dbscan_sam_cull',
        name='object_detector_dbscan_sam_cull', output='screen',
        prefix=[os.path.expanduser('~/irb_venv/bin/python3')],
        additional_env={'OMP_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8'},
        parameters=[{
            'base_frame': 'base_link',
            'input_cloud': '/object_detector/object_points',
            'color_topic': '/realsense/color/image_raw',
            'camera_info_topic': '/realsense/color/camera_info',
            'sam_checkpoint': LaunchConfiguration('sam_checkpoint'),
            'sam_device': LaunchConfiguration('sam_device'),
            'box_padding_px': 12,
            'min_cull_interval_sec': 2.0,
        }],
    )
    return LaunchDescription([checkpoint, device, node])
