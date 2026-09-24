"""
Perception launch file.

Launches the DBSCAN segmentation backend:

  ros2 launch irb120_perception perception.launch.py

robot_mask_filter and object_detector are compute-heavy but only actually
needed briefly (see their own docstrings' "On/off gate" section) —
`press_point_check.check_press_point()` already toggles them on/off around
each check regardless of this launch arg. This one just controls what state
they're in before the first check ever runs (or if you want to watch live
detections in RViz without running a control script):

  ros2 launch irb120_perception perception.launch.py active_at_start:=false

Pipeline topology:

  RealSense (cam1) ──┐
  RealSense (cam2) ──┼──▶ object_detector_dbscan (raw clouds fused in-node)
  RealSense (cam3) ──┘

The robot is deliberately out of the workspace for offline perception, so
the mask filter is not launched here.  Its executable and logic remain
available for workflows that need it. Set cam2_cloud_topic:='' /
cam3_cloud_topic:='' to disable an individual camera.
"""

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
            "Whether object_detector starts out processing "
            "immediately (default, matches historical always-on behaviour) or "
            "idle until the first check_press_point() call activates them."
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

    # ---- Contact point selection ---------------------------------------------
    # NOT launched here: `contact_point_selector` is invoked directly by
    # `irb120_control/util/press_point_check.py` (select_contact_points()) as
    # part of a press check, not run as a standalone persistent node. See
    # CONTACT_SELECTION.md.

    return LaunchDescription([
        cam2_cloud_arg,
        cam3_cloud_arg,
        active_at_start_arg,
        dbscan_node,
    ])
