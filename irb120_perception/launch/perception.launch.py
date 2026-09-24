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
  RealSense (cam2) ──┼──▶ robot_mask_filter ──▶ object_detector_dbscan
  RealSense (cam3) ──┘        ~/points_masked_dbscan  (all three cameras fused)

  robot_mask_filter fuses all three cameras' point clouds (see its own
  docstring) to reduce occlusion and increase point density. Set
  cam2_cloud_topic:='' / cam3_cloud_topic:='' to disable fusion for that
  camera individually.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# Masked topic published by robot_mask_filter
MASKED_CLOUD = '/robot_mask_filter/points_masked_dbscan'


def generate_launch_description() -> LaunchDescription:
    cam2_cloud_arg = DeclareLaunchArgument(
        'cam2_cloud_topic',
        default_value='/realsense2/depth/color/points',
        description=(
            "Second camera's point cloud, fused into the DBSCAN input by "
            "robot_mask_filter. Set to '' to disable fusion for this camera."
        ),
    )

    cam3_cloud_arg = DeclareLaunchArgument(
        'cam3_cloud_topic',
        default_value='/realsense3/depth/color/points',
        description=(
            "Third camera's point cloud, fused into the DBSCAN input by "
            "robot_mask_filter. Set to '' to disable fusion for this camera."
        ),
    )

    active_at_start_arg = DeclareLaunchArgument(
        'active_at_start',
        default_value='true',
        description=(
            "Whether robot_mask_filter + object_detector start out processing "
            "immediately (default, matches historical always-on behaviour) or "
            "idle until the first check_press_point() call activates them."
        ),
    )
    active_param = {'active': ParameterValue(LaunchConfiguration('active_at_start'), value_type=bool)}

    # ---- Robot mask filter ---------------------------------------------------
    mask_filter_node = Node(
        package='irb120_perception',
        executable='robot_mask_filter',
        name='robot_mask_filter',
        output='screen',
        parameters=[{
            'base_frame':  'base_link',
            'input_cloud': '/realsense/depth/color/points',
            'input_cloud2': LaunchConfiguration('cam2_cloud_topic'),
            'input_cloud3': LaunchConfiguration('cam3_cloud_topic'),
            'input_depth': '/realsense/aligned_depth_to_color/image_raw',
            'camera_info': '/realsense/color/camera_info',
            'robot_mask_padding': 0.08,  # arm mesh links — increase if arm still leaks through
            # Capsule segments (wrist sensor stack + finger) are fixed in
            # RobotMaskFilter.CAPSULE_SEGMENTS, not a launch param — see that
            # node's docstring. (A 'robot_mask_capsules' key used to be passed
            # here, but the node never declared/read it — dead config left
            # over from before the finger/sensor assembly was rebuilt, and it
            # still named the now-nonexistent ft_link/finger_link frames.)
        }, active_param],
    )

    # ---- DBSCAN parameters — reads from masked, camera-fused pointcloud -----
    dbscan_params = {
        'input_cloud_pc':      MASKED_CLOUD,
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
        # Off (1) by default. Would fuse this many recent frames via
        # voxel-occupancy consensus before clustering (see
        # object_detector_dbscan's docstring, "Temporal accumulation") — but
        # the naive latency estimate of accum_frames/publish_rate assumed
        # robot_mask_filter sustains something near camera rate (30-90 Hz).
        # Measured live instead: robot_mask_filter running continuously
        # (active_at_start default) sustains only ~1-1.5 Hz with heavy
        # jitter — it's the most expensive node in the chain (full-resolution
        # mesh/capsule masking on every point, three cameras, one thread) and
        # was never meant to run flat-out continuously, see its own
        # docstring's "On/off gate". At that real rate, accum_frames=10 means
        # 10+ seconds of total silence on ~/object_points before the first
        # detection — looks exactly like segmentation being broken. Only
        # raise this for a short, deliberate active window (e.g. around one
        # press_point_check call) where you can afford to wait and know the
        # input rate for that window; do not raise it for continuous bringup
        # viewing without re-measuring `ros2 topic hz
        # /robot_mask_filter/points_masked_dbscan` first.
        'accum_frames':     1,
        'accum_min_hits':   0,  # 0 = auto (~60% of accum_frames)
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
        mask_filter_node,
        dbscan_node,
    ])
