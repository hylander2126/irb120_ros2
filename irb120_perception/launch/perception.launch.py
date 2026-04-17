"""
Perception launch file.

Switch between segmentation backends via the 'method' argument:

  ros2 launch irb120_perception perception.launch.py method:=dbscan
  ros2 launch irb120_perception perception.launch.py method:=sam

Enable the debug pipeline (perception_debugger node) with:

  ros2 launch irb120_perception perception.launch.py method:=sam debug_perception:=true

DBSCAN: runs under system python, no GPU needed.
SAM:    runs under the venv python (~/.venvs/.venv_torch_SAM/bin/python3),
        requires CUDA GPU and SAM 2 weights.

Pipeline topology:

  RealSense ──▶ robot_mask_filter ──▶ object_detector
                    │ ~/points_masked   (DBSCAN input)
                    └ ~/depth_masked    (SAM input)
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import EqualsSubstitution, LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

PKG_SHARE      = get_package_share_directory('irb120_perception')
SAM_WEIGHTS    = os.path.join(PKG_SHARE, 'weights', 'sam2.1_hiera_tiny.pt')
VENV_SITE_PKGS = os.path.expanduser(
    '~/.venvs/.venv_torch_SAM/lib/python3.12/site-packages')

# Masked topic names published by robot_mask_filter
MASKED_CLOUD = '/robot_mask_filter/points_masked'
MASKED_DEPTH = '/robot_mask_filter/depth_masked'


def generate_launch_description() -> LaunchDescription:
    method_arg = DeclareLaunchArgument(
        'method',
        default_value='dbscan',
        description="Segmentation backend: 'dbscan' or 'sam'",
    )

    debug_arg = DeclareLaunchArgument(
        'debug_perception',
        default_value='false',
        description='Launch the perception_debugger node for on-demand SAM pipeline inspection.',
    )

    # ---- Robot mask filter (always running, both backends benefit) ----------
    mask_filter_node = Node(
        package='irb120_perception',
        executable='robot_mask_filter',
        name='robot_mask_filter',
        output='screen',
        parameters=[{
            'base_frame':  'base_link',
            'input_cloud': '/realsense/depth/color/points',
            'input_depth': '/realsense/aligned_depth_to_color/image_raw',
            'camera_info': '/realsense/color/camera_info',
            # Flattened (parent, child) pairs — each defines one capsule segment
            'robot_mask_capsules': [
                'base_link',   'link_1',
                'link_1',      'link_2',
                'link_2',      'link_3',
                'link_3',      'link_4',
                'link_4',      'link_5',
                'link_5',      'link_6',
                'link_6',      'ft_link',
                'ft_link',     'finger_link',
                'finger_link', 'finger_ball_center',
            ],
            'robot_mask_padding': 0.08,  # capsule radius — increase if arm still leaks through
        }],
    )

    # ---- DBSCAN parameters — reads from masked pointcloud -------------------
    dbscan_params = {
        'input_cloud_pc':      MASKED_CLOUD,
        'input_image':         '/realsense/color/image_raw',
        'base_frame':          'base_link',
        'segmentation_method': 'dbscan',
        'roi_x_min':  0.15,
        'roi_x_max':  0.80,
        'roi_y_min': -0.25,
        'roi_y_max':  0.25,
        'roi_z_min': -0.01,  # Table at Z≈-0.02; objects start above -0.01
        'roi_z_max':  0.50,
        'voxel_size':      0.005,
        'dbscan_eps':      0.02,
        'dbscan_min_pts':  20,
        'min_cluster_pts': 30,
        'max_cluster_pts': 50000,
        'smooth_alpha':    0.3,
    }

    # ---- SAM parameters — reads from masked depth image --------------------
    sam_params = {
        'input_cloud':    MASKED_DEPTH,
        'input_image':    '/realsense/color/image_raw',
        'camera_info':    '/realsense/color/camera_info',
        'base_frame':     'base_link',
        'segmentation_method': 'sam',
        'roi_x_min':  0.15,
        'roi_x_max':  0.80,
        'roi_y_min': -0.25,
        'roi_y_max':  0.25,
        'roi_z_min': -0.01,  # Table at Z≈-0.02; objects start above -0.01
        'roi_z_max':  0.50,
        'voxel_size':          0.005,
        'sam_weights':         SAM_WEIGHTS,
        'sam_config':          'configs/sam2.1/sam2.1_hiera_t.yaml',
        'sam_points_per_side':  8,
        'sam_iou_thresh':       0.85,
        'sam_min_mask_area':    1000,
        'sam_min_cluster_pts':  30,
        'sam_prominent_only':   True,
        'depth_median_ksize':   5,
        'outlier_std_ratio':    2.0,
        'smooth_alpha':         0.3,
    }

    # ---- DBSCAN node (system python) ----------------------------------------
    dbscan_node = Node(
        package='irb120_perception',
        executable='object_detector',
        name='object_detector',
        output='screen',
        parameters=[dbscan_params],
        condition=IfCondition(EqualsSubstitution(LaunchConfiguration('method'), 'dbscan')),
    )

    # ---- SAM node -----------------------------------------------------------
    sam_node = Node(
        package='irb120_perception',
        executable='object_detector',
        name='object_detector',
        output='screen',
        parameters=[sam_params],
        additional_env={'PYTHONPATH': VENV_SITE_PKGS + ':' + os.environ.get('PYTHONPATH', '')},
        condition=IfCondition(EqualsSubstitution(LaunchConfiguration('method'), 'sam')),
    )

    # ---- Debug node (optional) ----------------------------------------------
    debugger_node = Node(
        package='irb120_perception',
        executable='perception_debugger',
        name='perception_debugger',
        output='screen',
        condition=IfCondition(LaunchConfiguration('debug_perception')),
    )

    return LaunchDescription([
        method_arg,
        debug_arg,
        mask_filter_node,
        dbscan_node,
        sam_node,
        debugger_node,
    ])
