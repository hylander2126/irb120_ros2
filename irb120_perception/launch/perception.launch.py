"""
Perception launch file.

Launches DBSCAN segmentation followed by the SAM cull:

  ros2 launch irb120_perception perception.launch.py

  RealSense (cam1) ──┐
  RealSense (cam2) ──┼──▶ object_detector_dbscan ──▶ object_detector_dbscan_sam_cull (cam1 RGB; cam3 review mask)
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

from ament_index_python.packages import get_package_share_directory
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
        # Table removal: each camera's fixed view of the table, z = a*x + b*y + c in
        # base_link, the median depth_plane_audit fit over the 2026-10-13 snapshots
        # (cam1/cam3 p95 residual 2.2 mm, cam2 5.0 mm). Re-measure after recalibrating a
        # camera: python3 -m irb120_perception.depth_plane_audit SNAPSHOT --camera camN.
        'table_plane_cam1': [0.0112, -0.0009, -0.0279],
        'table_plane_cam2': [0.0108, 0.0036, -0.0191],
        'table_plane_cam3': [0.0039, 0.0006, -0.0208],
        'table_clearance': 0.006,  # above cam2's 5 mm p95 depth noise on the table
        'max_camera_age_sec': 1.0,  # every camera's latest cloud must be this recent to fuse
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
        # sklearn isn't installed for the system interpreter irb120_perception is built
        # under (a gap from the 2026-10 lab PC migration); it lives in the venv, so route
        # this node through it too, same prefix/thread-cap pattern as the SAM cull below.
        prefix=[os.path.expanduser('~/.virtualenvs/irb_venv/bin/python3')],
        additional_env={'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'},
        parameters=[dbscan_params, active_param],
    )

    # ---- SAM cull of DBSCAN's cloud -------------------------------------------
    # MobileSAM needs torch, so this one node runs on ~/.virtualenvs/irb_venv's Python
    # with BLAS threads capped (see README: an uncapped venv numpy once knocked out EGM).
    sam_checkpoint_arg = DeclareLaunchArgument(
        'sam_checkpoint',
        default_value=os.path.expanduser('~/Documents/irb120_ws_models/mobile_sam/mobile_sam.pt'),
    )
    sam_device_arg = DeclareLaunchArgument(
        'sam_device', default_value='cuda',
        description=(
            "MobileSAM device. Verified working on this RTX 4060 (~60 MB VRAM, warm inference "
            "~0.06-0.15s/frame vs ~0.85s/frame on CPU; see scripts/STEREO_OFFLINE.md). "
            "Set to 'cpu' to restore the old default."
        ),
    )
    sam_cull_node = Node(
        package='irb120_perception',
        executable='object_detector_dbscan_sam_cull',
        name='object_detector_dbscan_sam_cull',
        output='screen',
        prefix=[os.path.expanduser('~/.virtualenvs/irb_venv/bin/python3')],
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
            # cam3 mask for the forward-tip (far-edge) pivot fit; culls nothing.
            'review_camera': 'cam3',
            'review_color_topic': '/realsense3/color/image_raw',
            'review_depth_topic': '/realsense3/aligned_depth_to_color/image_raw',
            'review_camera_info_topic': '/realsense3/color/camera_info',
        }],
    )

    twin_node = Node(
        package='irb120_perception', executable='object_digital_twin',
        name='object_digital_twin', output='screen',
        parameters=[{'input_cloud': '/object_detector_dbscan_sam_cull/object_points',
                     'model_file': os.path.join(get_package_share_directory('irb120_perception'),
                                                'config', 'object_digital_twin.json')}],
    )
    confidence_map_node = Node(
        package='irb120_perception', executable='object_confidence_map',
        name='object_confidence_map', output='screen',
    )

    # Contact selection isn't a node: irb120_control's perception snapshot calls
    # select_contact_points() on the culled cloud. See CONTACT_SELECTION.md.

    return LaunchDescription([
        cam2_cloud_arg,
        cam3_cloud_arg,
        active_at_start_arg,
        sam_checkpoint_arg,
        sam_device_arg,
        dbscan_node,
        sam_cull_node,
        twin_node,
        confidence_map_node,
    ])
