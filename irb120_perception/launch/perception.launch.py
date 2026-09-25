"""
Perception launch file.

Launches DBSCAN segmentation followed by the SAM cull:

  ros2 launch irb120_perception perception.launch.py

  RealSense (cam1) ──┐
  RealSense (cam2) ──┼──▶ object_detector_dbscan ──▶ object_detector_dbscan_sam_cull (cam1 RGB)
  RealSense (cam3) ──┘      /object_detector/*        /object_detector_dbscan_sam_cull/*

DBSCAN finds the points most likely to be the object; the culler projects
them into cam1, prompts MobileSAM with their bounding box, and keeps only the
points inside the mask. The culler only works when DBSCAN publishes, and
irb120_control's perception snapshots switch DBSCAN on just for the snapshot
(active_at_start only sets its state before the first one).

The robot is deliberately out of the workspace for offline perception, so
the mask filter is not launched here.  Its executable and logic remain
available for workflows that need it. Set cam2_cloud_topic:='' /
cam3_cloud_topic:='' to disable an individual camera.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description() -> LaunchDescription:
    cam2_cloud_arg = DeclareLaunchArgument(
        'cam2_cloud_topic',
        default_value='/realsense2/depth/color/points',
        description=(
            "Second raw camera cloud, fused by object_detector. Set to '' to disable it."
        ),
    )

    cam3_cloud_arg = DeclareLaunchArgument(
        'cam3_cloud_topic',
        default_value='/realsense3/depth/color/points',
        description=(
            "Third raw camera cloud, fused by object_detector. Set to '' to disable it."
        ),
    )

    active_at_start_arg = DeclareLaunchArgument(
        'active_at_start',
        default_value='true',
        description=(
            "Whether object_detector starts out processing immediately (default) "
            "or idle until the first perception snapshot switches it on."
        ),
    )
    active_param = {'active': ParameterValue(LaunchConfiguration('active_at_start'), value_type=bool)}

    # ---- DBSCAN parameters — reads raw, camera-fused pointclouds ------------
    dbscan_params = {
        'input_cloud_pc':      '/realsense/depth/color/points',
        'input_cloud_pc2':     LaunchConfiguration('cam2_cloud_topic'),
        'input_cloud_pc3':     LaunchConfiguration('cam3_cloud_topic'),
        'base_frame':          'base_link',
        'roi_x_min':  0.15,
        'roi_x_max':  0.80,
        'roi_y_min': -0.25,
        'roi_y_max':  0.25,
        'roi_z_min': -0.01,  # Table at Z≈-0.02 (tilts up to ≈-0.015 at far x with the 2026-09-18 cam calibration)
        'roi_z_max':  0.50,
        'voxel_size':      0.005,
        'dbscan_eps':      0.02,
        'dbscan_min_pts':  20,
        'min_cluster_pts': 30,
        'max_cluster_pts': 50000,
        'outlier_k':          8,    # neighbours sampled per point for local-density check
        'outlier_std_ratio':  2.0,  # 0 disables; lower = more aggressive stray-point removal
        'table_plane_distance': 0.008,  # reject tabletop before DBSCAN can bridge through it
        # Offline contact selection has a stationary scene. Keep a voxel only
        # when it recurs in 12 of the last 20 fused observations: this removes
        # edge flicker and free-space specks that survive one-frame density
        # filtering. At normal direct-camera rates this warm-up is well below
        # the contact-check timeout; increase either value only if needed.
        'accum_frames':     20,
        'accum_min_hits':   12,
        'smooth_alpha':    0.3,
        # Union all surviving clusters into one object — only safe if the
        # workspace is scoped to a single physical item per detection cycle.
        # See object_detector_dbscan's docstring.
        'single_object_mode': True,
    }

    dbscan_node = Node(
        package='irb120_perception',
        executable='object_detector_dbscan',
        name='object_detector',
        output='screen',
        parameters=[dbscan_params, active_param],
    )

    # ---- SAM cull of DBSCAN's cloud -------------------------------------------
    # MobileSAM needs torch, so this one node runs on ~/irb_venv's Python with
    # BLAS threads capped (see README: an uncapped venv numpy once knocked out EGM).
    sam_checkpoint_arg = DeclareLaunchArgument(
        'sam_checkpoint',
        default_value=os.path.expanduser('~/irb120_ws_models/mobile_sam/mobile_sam.pt'),
    )
    sam_cull_node = Node(
        package='irb120_perception',
        executable='object_detector_dbscan_sam_cull',
        name='object_detector_dbscan_sam_cull',
        output='screen',
        prefix=[os.path.expanduser('~/irb_venv/bin/python3')],
        additional_env={'OMP_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8'},
        parameters=[{
            'base_frame': 'base_link',
            'input_cloud': '/object_detector/object_points',
            'color_topic': '/realsense/color/image_raw',
            'camera_info_topic': '/realsense/color/camera_info',
            'sam_checkpoint': LaunchConfiguration('sam_checkpoint'),
            'sam_device': 'cpu',
            'box_padding_px': 12,
            'min_cull_interval_sec': 2.0,
        }],
    )

    # Contact selection isn't a node: irb120_control's perception snapshot calls
    # select_contact_points() on the culled cloud. See CONTACT_SELECTION.md.

    return LaunchDescription([
        cam2_cloud_arg,
        cam3_cloud_arg,
        active_at_start_arg,
        sam_checkpoint_arg,
        dbscan_node,
        sam_cull_node,
    ])
