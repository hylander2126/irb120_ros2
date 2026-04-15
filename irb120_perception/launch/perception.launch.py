"""
Perception launch file.

Switch between segmentation backends via the 'method' argument:

  ros2 launch irb120_perception perception.launch.py method:=dbscan
  ros2 launch irb120_perception perception.launch.py method:=sam

DBSCAN: runs under system python, no GPU needed.
SAM:    runs under the venv python (~/.venvs/.venv_torch_SAM/bin/python3),
        requires CUDA GPU and SAM 2 weights.
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


def generate_launch_description() -> LaunchDescription:
    method_arg = DeclareLaunchArgument(
        'method',
        default_value='dbscan',
        description="Segmentation backend: 'dbscan' or 'sam'",
    )

    # ---- Shared parameters (both methods) -----------------------------------
    common_params = {
        'input_cloud':  '/realsense/depth/color/points',
        'input_image':  '/realsense/color/image_raw',
        'base_frame':   'base_link',

        # Workspace ROI (metres, base_link frame)
        'roi_x_min':  0.15,
        'roi_x_max':  0.80,
        'roi_y_min': -0.25,
        'roi_y_max':  0.25,
        'roi_z_min': -0.02,   # table surface ≈ Z=0 in base_link
        'roi_z_max':  0.50,

        'voxel_size': 0.005,  # 5 mm downsample
    }

    # ---- DBSCAN-specific parameters -----------------------------------------
    dbscan_params = {
        **common_params,
        'segmentation_method': 'dbscan',
        'dbscan_eps':          0.02,    # 2 cm neighbourhood radius
        'dbscan_min_pts':      20,
        'min_cluster_pts':     30,
        'max_cluster_pts':     50000,
    }

    # ---- SAM-specific parameters --------------------------------------------
    sam_params = {
        **common_params,
        'segmentation_method':  'sam',
        'sam_weights':          SAM_WEIGHTS,
        'sam_config':           'configs/sam2.1/sam2.1_hiera_t.yaml',
        'sam_points_per_side':  16,     # lower = faster, fewer small masks
        'sam_iou_thresh':       0.80,
        'sam_min_mask_area':    500,    # pixels — ignore tiny masks
        'sam_min_cluster_pts':  30,     # 3D points per mask to keep
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
    # Injects the venv site-packages (torch, sam2, cv2) via PYTHONPATH so the
    # ROS-launched process can import them without changing the executable.
    sam_node = Node(
        package='irb120_perception',
        executable='object_detector',
        name='object_detector',
        output='screen',
        parameters=[sam_params],
        additional_env={'PYTHONPATH': VENV_SITE_PKGS + ':' + os.environ.get('PYTHONPATH', '')},
        condition=IfCondition(EqualsSubstitution(LaunchConfiguration('method'), 'sam')),
    )

    return LaunchDescription([
        method_arg,
        dbscan_node,
        sam_node,
    ])
